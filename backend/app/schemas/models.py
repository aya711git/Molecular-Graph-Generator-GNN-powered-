"""Pydantic schemas shared across the API layer.

Keeping these separate from ORM/graph internals means the wire format is
stable even if internal representations change.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class InputType(str, Enum):
    """How the caller wants their `query` string interpreted.

    `auto` tries SMILES first (cheapest, offline), then falls back to a
    PubChem name/formula lookup.
    """

    AUTO = "auto"
    SMILES = "smiles"
    NAME = "name"
    FORMULA = "formula"


class LayerType(str, Enum):
    MPNN = "mpnn"
    GCN = "gcn"


class MoleculeRequest(BaseModel):
    """Payload for /parse, /graph, and /embed."""

    query: str = Field(
        ..., min_length=1, max_length=300,
        description="A SMILES string, a drug/common name, or a molecular formula.",
        examples=["CC(=O)OC1=CC=CC=C1C(=O)O", "Aspirin", "C9H8O4"],
    )
    input_type: InputType = Field(default=InputType.AUTO)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty or whitespace-only")
        return v


class GNNConfig(BaseModel):
    """Optional overrides for the GNN encoder, exposed to power users."""

    layer_type: LayerType = Field(default=LayerType.MPNN)
    hidden_dim: int = Field(default=128, ge=8, le=1024)
    num_layers: int = Field(default=3, ge=1, le=12)
    latent_dim: int = Field(default=64, ge=2, le=512)
    dropout: float = Field(default=0.1, ge=0.0, le=0.9)
    pooling: str = Field(default="mean", pattern="^(mean|max|add)$")
    seed: Optional[int] = Field(
        default=42, description="Random seed for reproducible (untrained) weights."
    )


class EmbedRequest(MoleculeRequest):
    gnn_config: GNNConfig = Field(default_factory=GNNConfig)


class MoleculeInfo(BaseModel):
    """Canonicalized chemical identity, independent of how it was queried."""

    canonical_smiles: str
    molecular_formula: str
    molecular_weight: float
    num_atoms: int
    num_heavy_atoms: int
    num_bonds: int
    num_rings: int
    resolved_from: InputType
    iupac_name: Optional[str] = None
    pubchem_cid: Optional[int] = None


class AtomNode(BaseModel):
    """One graph node, with both the raw feature vector and human-readable
    metadata for the frontend to render labels/tooltips."""

    index: int
    symbol: str
    atomic_number: int
    degree: int
    formal_charge: int
    hybridization: str
    is_aromatic: bool
    is_in_ring: bool
    num_hydrogens: int
    features: List[float]
    x: Optional[float] = Field(default=None, description="2D layout x-coordinate")
    y: Optional[float] = Field(default=None, description="2D layout y-coordinate")


class BondEdge(BaseModel):
    """One graph edge (bond). Graphs are represented as directed edge pairs
    (i -> j and j -> i) to match PyTorch Geometric conventions, but this
    schema represents the undirected chemical bond once."""

    source: int
    target: int
    bond_type: str
    is_aromatic: bool
    is_conjugated: bool
    is_in_ring: bool
    features: List[float]


class MolecularGraph(BaseModel):
    molecule: MoleculeInfo
    nodes: List[AtomNode]
    edges: List[BondEdge]
    node_feature_dim: int
    edge_feature_dim: int


class EmbeddingResult(BaseModel):
    graph: MolecularGraph
    node_embeddings: List[List[float]] = Field(
        description="Per-atom latent vectors after the final GNN layer."
    )
    graph_embedding: List[float] = Field(
        description="Pooled whole-molecule latent vector."
    )
    embedding_dim: int
    layer_type: LayerType
    num_layers: int


class ErrorResponse(BaseModel):
    error_type: str
    message: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    rdkit_available: bool
    torch_available: bool
