<!--Copyright (c) 2026 Aya Khaled Khuris. All rights reserved.
Licensed under the MIT License. See LICENSE file in the project root for details.

Molecular Graph Generator (GNN-Powered)
Author: Aya Khaled Khuris <aya.khuris@gmail.com>
Architecture & Graph Construction Logic -->

This document explains, precisely, how a raw chemical input becomes a
tensor that a Graph Neural Network can consume, and how the GNN itself is
structured. It is meant to be read alongside `backend/app/services/`.

---

## 1. Pipeline overview

```
┌──────────────┐   ┌───────────────────┐   ┌──────────────────┐   ┌────────────────┐
│  Raw input   │──▶│ molecule_parser.py│──▶│ graph_builder.py │──▶│  gnn_model.py   │
│ SMILES/name/ │   │ RDKit + PubChemPy │   │ Featurization    │   │ PyG MPNN / GCN  │
│ formula      │   │ → sanitized Mol   │   │ → MolecularGraph │   │ → embeddings    │
└──────────────┘   └───────────────────┘   └──────────────────┘   └────────────────┘
```

Each stage has a single, testable responsibility and only depends on typed
outputs from the previous stage — no stage reaches "backward" into raw
strings or "forward" into tensors.

---

## 2. Input resolution (`molecule_parser.py`)

| Input mode | Resolution strategy | Library |
|---|---|---|
| `smiles` | Direct parse: `Chem.MolFromSmiles` → `Chem.SanitizeMol` | RDKit |
| `name`   | PubChem name search → canonical SMILES → RDKit parse | PubChemPy → RDKit |
| `formula`| PubChem formula search → canonical SMILES → RDKit parse | PubChemPy → RDKit |
| `auto`   | Try SMILES first (offline, instant); on failure try PubChem name, then PubChem formula | both |

**Why SMILES-first in `auto` mode?** SMILES parsing is a local, offline,
O(atoms) operation. Attempting it before any network call means the common
case (a user pastes a SMILES string) never waits on PubChem's API or is
vulnerable to it being down.

**Sanitization.** RDKit's sanitization step performs valence checking,
aromaticity perception (via its own model, not naive alternating
single/double bonds), ring perception, and conjugation flagging. A
molecule that fails sanitization (e.g. an atom with impossible valence) is
rejected with `InvalidMoleculeError` rather than silently producing a
malformed graph.

**Safety bound.** `MAX_HEAVY_ATOMS` (default 150) prevents pathologically
large inputs (e.g. a polymer SMILES) from generating graphs so large that
GNN inference becomes a denial-of-service vector. This is a defensive
default that deployers can raise for legitimate large-molecule use cases.

---

## 3. Graph construction (`graph_builder.py`)

An RDKit `Mol` is a rich object graph; we project it down to two flat
tensors — one per atom, one per bond — so the representation is uniform
across every possible input molecule.

### 3.1 Node (atom) features — `ATOM_FEATURE_DIM = 45`

| Slice | Meaning | Encoding |
|---|---|---|
| `[0:43]` | Atomic symbol | One-hot over a 42-element common-element vocabulary (`C, N, O, S, F, Si, P, Cl, Br, ...`) plus a trailing `UNK` bucket for anything outside the vocabulary |
| `[43]` | Aromaticity | Single boolean (0/1) |

Design choices:

* **Fixed vocabulary, not full periodic table.** A curated ~42-element
  vocabulary (covering essentially all drug-like and common organic/
  inorganic chemistry) keeps the one-hot block small and the embedding
  table well-conditioned, while the `UNK` bucket guarantees the pipeline
  never crashes on an exotic element — it degrades gracefully instead.
* **Purely categorical / boolean vector.** Every entry in the 45-dim vector
  is exactly 0.0 or 1.0. This avoids scale mismatches between, say, a
  formal charge of `-1` and a degree of `4` dominating the aromaticity
  bit — a common failure mode when mixing raw integers into a GNN's first
  layer without normalization.
* **Richer descriptors still travel with the node.** Degree, formal
  charge, hybridization, ring membership, and hydrogen count are computed
  and returned on `AtomNode` (see `schemas/models.py`) for the frontend to
  render as tooltips/labels, even though they aren't baked into the
  learned feature vector in this reference implementation. Extending
  `featurize_atom` to fold these in as additional one-hot/scalar blocks is
  a straightforward, isolated change — bump `ATOM_FEATURE_DIM` in
  `config.py` to match.

### 3.2 Edge (bond) features — `BOND_FEATURE_DIM = 10`

| Slice | Meaning | Encoding |
|---|---|---|
| `[0:4]` | Bond type | One-hot: `SINGLE, DOUBLE, TRIPLE, AROMATIC` |
| `[4]` | Conjugation | Boolean |
| `[5]` | Ring membership | Boolean |
| `[6:10]` | Stereochemistry | One-hot: `STEREONONE, STEREOZ, STEREOE, STEREOANY` |

Bonds are chemically undirected, but PyTorch Geometric's message passing
operates on directed edges. `gnn_model.graph_to_pyg_data` therefore
duplicates every bond as `(i→j)` and `(j→i)` with identical edge features,
which is the standard way to represent undirected graphs in PyG.

### 3.3 2D layout coordinates

`AllChem.Compute2DCoords` generates a depiction layout (the same algorithm
class used by chemical drawing software) so the frontend can render a
recognizable structure instead of an arbitrary force-directed blob. This
is purely cosmetic — it has no bearing on the GNN's input — and falls back
silently to `(0, 0)` coordinates (frontend auto-layout) if 2D embedding
fails for any reason.

### 3.4 Defensive consistency checks

After featurizing every atom/bond, `build_graph` asserts the first node's
and first edge's feature-vector lengths match the configured
`ATOM_FEATURE_DIM` / `BOND_FEATURE_DIM`. This catches silent featurizer
bugs (e.g. a one-hot block resized without updating the config) at graph
construction time rather than as a cryptic shape-mismatch deep inside the
GNN.

---

## 4. GNN architecture (`gnn_model.py`)

### 4.1 Two interchangeable message-passing schemes

**GCN (`GCNConv`, Kipf & Welling 2017).** Each atom's new representation is
a normalized average of its neighbors' representations, transformed by a
shared weight matrix. Fast and effective, but treats every bond
identically — it cannot distinguish a single bond from a triple bond.

**MPNN (`NNConv`, Gilmer et al. 2017).** An auxiliary "edge network" (a
small MLP) maps each bond's 10-dim feature vector to a full
`hidden_dim × hidden_dim` transformation matrix, which is then applied to
the neighbor's message before aggregation. This means bond chemistry
(type, conjugation, ring membership, stereochemistry) directly modulates
how information flows across that bond — a double bond and a single bond
between the same two atom types propagate different messages.

Both are wrapped in a layer that adds `LayerNorm → ReLU → Dropout`, which
stabilizes training (or, for this untrained reference model, keeps
activations well-scaled for inspection) across stacked layers.

### 4.2 Forward pass

```
atom features (N × 45)
   │  Linear projection
   ▼
hidden state (N × hidden_dim)
   │  L × [message passing layer]     (L = num_layers, default 3)
   ▼
per-atom hidden state (N × hidden_dim)
   │  2-layer MLP readout
   ▼
per-atom embedding (N × latent_dim)      ──▶  returned as `node_embeddings`
   │  pooling (mean / max / sum) over atoms
   ▼
graph embedding (1 × latent_dim)          ──▶  returned as `graph_embedding`
```

### 4.3 Configurability

Every hyperparameter that materially changes the architecture — layer
type, hidden width, depth, latent dimension, dropout, pooling function,
and random seed — is exposed through `GNNConfig` and accepted per-request
via the `/api/v1/embed` endpoint. This lets a caller compare, e.g., a
64-dim MPNN embedding against a 128-dim GCN embedding for the same
molecule without redeploying the service.

### 4.4 Untrained weights, and how to plug in a trained checkpoint

This reference implementation seeds the model's random weights (via
`GNNConfig.seed`) for reproducibility, but does **not** ship a pretrained
checkpoint — the embeddings it produces reflect the graph's *structure*
(a molecule's topology strongly determines which random projection it
lands near) but are not tuned for any downstream task like property
prediction.

To use a trained model in production:

1. Train `MolecularGNN` (or a subclass) against a labeled dataset (e.g.
   QM9, ZINC, ESOL) using a standard PyG training loop with an appropriate
   loss head on top of `graph_embedding`.
2. Save with `torch.save(model.state_dict(), "checkpoint.pt")`.
3. In `gnn_model.run_inference`, load the checkpoint once at process
   startup (e.g. in a FastAPI `lifespan` handler) instead of constructing
   a fresh randomly-initialized model per request, and reuse that instance
   across requests for both correctness (consistent weights) and latency.

### 4.5 Batching multiple molecules

`MolecularGNN.forward` already accepts a PyG `Batch` (multiple `Data`
objects concatenated with a `batch` index vector), so extending the API
with a `/embed-batch` endpoint that processes a list of molecules in one
forward pass — rather than looping — is a natural, low-effort extension:
build each molecule's `Data` via `graph_to_pyg_data`, combine with
`torch_geometric.data.Batch.from_data_list`, and run one forward pass.

---

## 5. Error handling philosophy

Every failure mode maps to a specific, typed exception in
`app/core/exceptions.py`, which the API layer (`app/api/routes.py`)
translates into a precise HTTP status code:

| Exception | HTTP status | Example trigger |
|---|---|---|  
| `InputResolutionError` | 404 | Drug name not found on PubChem |
| `InvalidMoleculeError` | 422 | Malformed SMILES, failed sanitization |
| `MoleculeTooLargeError` | 413 | Heavy-atom count exceeds `MAX_HEAVY_ATOMS` |
| `GraphConstructionError` | 500 | Internal featurization inconsistency |
| `ModelInferenceError` | 500 | GNN forward pass failure |

A global `MolGNNError` handler in `main.py` is a last-resort safety net
that guarantees no domain exception ever surfaces as an unhandled 500
traceback to the client, even if a route forgets to catch it explicitly.

---

## 6. Frontend rendering notes

* Graph layout uses the same 2D coordinates RDKit would use to *draw* the
  molecule, so the visual matches what a chemist expects to see — this is
  why coordinates travel all the way from `graph_builder.py` through the
  API to `MoleculeGraph.tsx`, rather than the frontend computing its own
  force-directed layout from scratch.
* Double/triple bonds are rendered as 2/3 parallel lines (standard
  chemical drawing convention), computed client-side from the bond-type
  string — no extra data is needed from the backend for this.
* The embedding heatmap intentionally uses a diverging teal/magenta scale
  (rather than a sequential scale) because embedding values are
  zero-centered (no activation function on the final readout layer), so a
  diverging palette makes sign meaningful at a glance.
