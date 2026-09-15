# Molecular Graph Generator (GNN-powered)

A production-grade, full-stack application that converts chemical input
(SMILES strings, drug names, or molecular formulas) into an interactive
molecular graph and computes latent-space embeddings using a Graph Neural
Network (Message Passing Neural Network / GCN).

```
User Input (SMILES / drug name / formula)
        │
        ▼
 RDKit / PubChemPy  →  Molecule Validation & Parsing
        │
        ▼
 Graph Constructor   →  Atom nodes + Bond edges + Feature Vectors
        │
        ▼
 PyTorch Geometric   →  MPNN / GCN Encoder  →  Latent Embedding
        │
        ▼
 FastAPI JSON API    →  React/Next.js Frontend  →  Interactive Graph + Embedding Viewer
```

## Repository layout

```
molgnn/
├── backend/                 FastAPI service, RDKit parsing, GNN model
│   ├── app/
│   │   ├── core/            Configuration & app-wide constants
│   │   ├── schemas/         Pydantic request/response models
│   │   ├── services/        Business logic (parsing, graph building, GNN, embeddings)
│   │   ├── api/             REST route definitions
│   │   └── main.py          FastAPI application entrypoint
│   ├── tests/                Unit tests (pytest)
│   └── requirements.txt
├── frontend/                 Next.js + TypeScript + Tailwind UI
│   ├── src/
│   │   ├── components/       MoleculeGraph, EmbeddingPanel, InputForm
│   │   ├── lib/               API client
│   │   └── styles/
│   └── package.json
├── docs/
│   └── ARCHITECTURE.md       Detailed graph-construction & GNN documentation
└── README.md                 (this file)
```

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`. Interactive OpenAPI docs
are auto-generated at `http://localhost:8000/docs`.

Key endpoints:

| Method | Path                | Description                                             |
|--------|---------------------|----------------------------------------------------------|
| POST   | `/api/v1/parse`      | Validate & parse SMILES / name / formula into a molecule  |
| POST   | `/api/v1/graph`       | Build the molecular graph (nodes, edges, features)        |
| POST   | `/api/v1/embed`       | Run the GNN and return the latent embedding + graph        |
| GET    | `/api/v1/health`      | Liveness/readiness probe                                   |

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000`. Set `NEXT_PUBLIC_API_URL` (defaults to
`http://localhost:8000`) if your backend runs elsewhere.

## Design notes

* **Input flexibility** — a single `/parse` endpoint accepts raw SMILES,
  a common/IUPAC drug name (resolved via PubChemPy → PubChem REST API), or a
  molecular formula (resolved via PubChemPy formula search). All paths
  converge on an RDKit `Mol` object, which is the single source of truth for
  everything downstream.
* **Deterministic featurization** — atom/bond featurizers are pure functions
  with fixed-size, documented output vectors (see `docs/ARCHITECTURE.md`),
  so the same molecule always produces the same graph tensor, and the model
  is agnostic to *how* the molecule was originally specified.
* **Modular GNN** — the encoder is assembled from interchangeable message
  passing layers (`GCNConv`, `NNConv`/MPNN) via a config object, so new
  layer types can be added without touching the API or frontend.
* **Separation of concerns** — RDKit logic never touches PyTorch; the GNN
  never touches HTTP; the frontend never touches chemistry — it only
  renders whatever JSON the API returns. This keeps each layer testable in
  isolation.

// Copyright (c) 2026 Aya Khaled Khuris. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for details.

// Molecular Graph Generator (GNN-Powered)
// Author: Aya Khaled Khuris <aya.khuris@gmail.com>