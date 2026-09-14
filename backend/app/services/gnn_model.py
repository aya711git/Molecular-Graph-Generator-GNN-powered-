"""Modular GNN encoder for molecular graphs.

Two interchangeable message-passing schemes are provided behind a single
``MolecularGNN`` class, selected via ``GNNConfig.layer_type``:

* ``gcn``  — stacked :class:`~torch_geometric.nn.GCNConv` layers. Fast,
  simple, uses only the graph topology (bond features are ignored beyond
  connectivity).
* ``mpnn`` — a Message Passing Neural Network in the style of Gilmer et al.
  (2017), implemented with :class:`~torch_geometric.nn.NNConv`, where an
  edge-conditioned MLP generates the message-passing weight matrix from
  bond features. This lets bond type/conjugation/ring membership directly
  influence how atom representations are updated.

Both variants share the same interface: raw atom/bond feature tensors in,
per-atom embeddings + a pooled graph embedding out. This lets the API layer
stay agnostic to which architecture is active.

Note: weights are randomly initialized (seeded for reproducibility). This
module ships the *architecture* and inference pipeline; swapping in
pretrained checkpoints only requires calling ``model.load_state_dict(...)``
before inference — see the "Training & checkpoints" section of
``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, NNConv, global_add_pool, global_max_pool, global_mean_pool

from app.core.exceptions import ModelInferenceError
from app.schemas.models import GNNConfig, LayerType, MolecularGraph

_POOLING_FNS = {
    "mean": global_mean_pool,
    "max": global_max_pool,
    "add": global_add_pool,
}


class _MPNNLayer(nn.Module):
    """One edge-conditioned message-passing layer (Gilmer et al. MPNN)."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        # Edge network maps bond features -> a full (in_dim x out_dim)
        # weight matrix used to transform each neighbor's message.
        edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, in_dim * out_dim),
        )
        self.conv = NNConv(in_dim, out_dim, edge_mlp, aggr="mean")
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        out = self.conv(x, edge_index, edge_attr)
        out = self.norm(out)
        out = F.relu(out)
        return self.dropout(out)


class _GCNLayer(nn.Module):
    """One standard GCN layer (Kipf & Welling)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        out = self.conv(x, edge_index)
        out = self.norm(out)
        out = F.relu(out)
        return self.dropout(out)


class MolecularGNN(nn.Module):
    """Stacked message-passing encoder producing atom- and graph-level
    embeddings for a single molecule (or a batch, via PyG's ``Batch``)."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        config: GNNConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(node_feature_dim, config.hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            if config.layer_type == LayerType.MPNN:
                self.layers.append(
                    _MPNNLayer(config.hidden_dim, config.hidden_dim, edge_feature_dim, config.dropout)
                )
            else:
                self.layers.append(_GCNLayer(config.hidden_dim, config.hidden_dim, config.dropout))

        self.readout_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.pool = _POOLING_FNS[config.pooling]

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (node_embeddings, graph_embedding)."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if data.batch is not None else torch.zeros(x.size(0), dtype=torch.long)

        h = F.relu(self.input_proj(x))
        for layer in self.layers:
            if isinstance(layer, _MPNNLayer):
                h = layer(h, edge_index, edge_attr)
            else:
                h = layer(h, edge_index)

        node_embeddings = self.readout_proj(h)
        graph_embedding = self.pool(node_embeddings, batch)
        return node_embeddings, graph_embedding


def graph_to_pyg_data(graph: MolecularGraph) -> Data:
    """Convert the API's ``MolecularGraph`` schema into a PyG ``Data``
    object with bidirectional edges (required for undirected message
    passing in PyTorch Geometric)."""
    if not graph.nodes:
        raise ModelInferenceError("Cannot build a graph tensor with zero atoms.")

    x = torch.tensor([n.features for n in graph.nodes], dtype=torch.float)

    if graph.edges:
        src = [e.source for e in graph.edges] + [e.target for e in graph.edges]
        dst = [e.target for e in graph.edges] + [e.source for e in graph.edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_feats = [e.features for e in graph.edges] * 2
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    else:
        # Single-atom molecules (e.g. "O", "[Na+]") have no bonds.
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, graph.edge_feature_dim), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def run_inference(graph: MolecularGraph, config: GNNConfig) -> Tuple[list, list]:
    """Build the model + data, run a forward pass, return plain Python lists
    ready for JSON serialization.

    Raises:
        ModelInferenceError: on any tensor-construction or forward-pass failure.
    """
    try:
        if config.seed is not None:
            torch.manual_seed(config.seed)

        data = graph_to_pyg_data(graph)
        model = MolecularGNN(
            node_feature_dim=graph.node_feature_dim,
            edge_feature_dim=graph.edge_feature_dim,
            config=config,
        )
        model.eval()
        with torch.no_grad():
            node_emb, graph_emb = model(data)

        return node_emb.tolist(), graph_emb.squeeze(0).tolist()
    except ModelInferenceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelInferenceError(f"GNN forward pass failed: {exc}") from exc
