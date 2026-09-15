// Copyright (c) 2026 Aya Khaled Khuris. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for details.

// Molecular Graph Generator (GNN-Powered)
// Author: Aya Khaled Khuris <aya.khuris@gmail.com>
/**
 * Typed client for the Molecular Graph Generator backend.
 *
 * Every function throws an `ApiError` on non-2xx responses, carrying the
 * backend's structured error payload so components can render meaningful
 * messages instead of a generic "something went wrong".
 */

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export type InputType = "auto" | "smiles" | "name" | "formula";
export type LayerType = "mpnn" | "gcn";

export interface MoleculeInfo {
  canonical_smiles: string;
  molecular_formula: string;
  molecular_weight: number;
  num_atoms: number;
  num_heavy_atoms: number;
  num_bonds: number;
  num_rings: number;
  resolved_from: InputType;
  iupac_name?: string | null;
  pubchem_cid?: number | null;
}

export interface AtomNode {
  index: number;
  symbol: string;
  atomic_number: number;
  degree: number;
  formal_charge: number;
  hybridization: string;
  is_aromatic: boolean;
  is_in_ring: boolean;
  num_hydrogens: number;
  features: number[];
  x?: number | null;
  y?: number | null;
}

export interface BondEdge {
  source: number;
  target: number;
  bond_type: string;
  is_aromatic: boolean;
  is_conjugated: boolean;
  is_in_ring: boolean;
  features: number[];
}

export interface MolecularGraph {
  molecule: MoleculeInfo;
  nodes: AtomNode[];
  edges: BondEdge[];
  node_feature_dim: number;
  edge_feature_dim: number;
}

export interface GNNConfig {
  layer_type: LayerType;
  hidden_dim: number;
  num_layers: number;
  latent_dim: number;
  dropout: number;
  pooling: "mean" | "max" | "add";
  seed?: number | null;
}

export interface EmbeddingResult {
  graph: MolecularGraph;
  node_embeddings: number[][];
  graph_embedding: number[];
  embedding_dim: number;
  layer_type: LayerType;
  num_layers: number;
}

export class ApiError extends Error {
  status: number;
  detail?: string;

  constructor(status: number, message: string, detail?: string) {
    super(message);
    this.status = status;
    this.detail = detail;
    this.name = "ApiError";
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    let detail = "";
    try {
      const errJson = await res.json();
      detail = errJson.detail || errJson.message || "";
    } catch {
      detail = await res.text();
    }
    throw new ApiError(res.status, `Request to ${path} failed (${res.status})`, detail);
  }

  return res.json() as Promise<T>;
}

export const api = {
  parseMolecule: (query: string, input_type: InputType = "auto") =>
    post<MoleculeInfo>("/api/v1/parse", { query, input_type }),

  buildGraph: (query: string, input_type: InputType = "auto") =>
    post<MolecularGraph>("/api/v1/graph", { query, input_type }),

  embedMolecule: (query: string, input_type: InputType = "auto", gnn_config?: Partial<GNNConfig>) =>
    post<EmbeddingResult>("/api/v1/embed", {
      query,
      input_type,
      gnn_config: {
        layer_type: "mpnn",
        hidden_dim: 128,
        num_layers: 3,
        latent_dim: 64,
        dropout: 0.1,
        pooling: "mean",
        seed: 42,
        ...gnn_config,
      },
    }),

  healthCheck: () =>
    fetch(`${API_BASE_URL}/api/v1/health`).then((r) => r.json()),
};
