"use client";

import { useState } from "react";
import { api, ApiError, EmbeddingResult, InputType } from "@/lib/api";
import InputForm from "@/components/InputForm";
import MoleculeGraph from "@/components/MoleculeGraph";
import MoleculeInfoCard from "@/components/MoleculeInfoCard";
import EmbeddingPanel from "@/components/EmbeddingPanel";

export default function HomePage() {
  const [result, setResult] = useState<EmbeddingResult | null>(null);
  const [selectedAtom, setSelectedAtom] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(query: string, inputType: InputType) {
    setLoading(true);
    setError(null);
    setSelectedAtom(null);
    try {
      const res = await api.embedMolecule(query, inputType);
      setResult(res);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.detail || err.message);
      } else {
        setError("Unexpected error contacting the backend. Is it running?");
      }
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="max-w-6xl mx-auto px-4 py-10 space-y-8">
      <header className="space-y-2">
        <h1 className="text-2xl font-bold tracking-tight">
          Molecular Graph <span className="text-accent-400">Generator</span>
        </h1>
        <p className="text-sm text-slate-400 max-w-2xl">
          Enter a SMILES string, drug name, or molecular formula. The backend
          resolves it with RDKit/PubChemPy, builds a typed molecular graph,
          and runs it through a message-passing GNN to produce atom- and
          molecule-level latent embeddings.
        </p>
      </header>

      <InputForm onSubmit={handleSubmit} loading={loading} />

      {error && (
        <div className="bg-red-950/40 border border-red-800 text-red-300 text-sm rounded-lg px-4 py-3">
          {error}
        </div>
      )}

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          <div className="lg:col-span-3 space-y-4">
            <MoleculeGraph
              graph={result.graph}
              highlightedAtom={selectedAtom}
              onSelectAtom={setSelectedAtom}
            />
            <MoleculeInfoCard info={result.graph.molecule} />
          </div>
          <div className="lg:col-span-2 bg-graphite-800/50 border border-graphite-700 rounded-xl p-4">
            <h2 className="text-sm font-semibold text-slate-300 mb-4">
              Latent Embeddings
            </h2>
            <EmbeddingPanel result={result} selectedAtom={selectedAtom} />
          </div>
        </div>
      )}

      {!result && !error && (
        <div className="text-center text-slate-500 text-sm py-20 border border-dashed border-graphite-700 rounded-xl">
          Enter a molecule above (or click an example) to generate its graph.
        </div>
      )}
    </main>
  );
}
