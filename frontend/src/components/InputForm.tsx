// Copyright (c) 2026 Aya Khaled Khuris. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for details.

// Molecular Graph Generator (GNN-Powered)
// Author: Aya Khaled Khuris <aya.khuris@gmail.com>
"use client";

import { useState, FormEvent } from "react";
import type { InputType } from "@/lib/api";

interface Props {
  onSubmit: (query: string, inputType: InputType) => void;
  loading: boolean;
}

const EXAMPLES: { label: string; query: string; type: InputType }[] = [
  { label: "Aspirin (name)", query: "Aspirin", type: "name" },
  { label: "Caffeine (SMILES)", query: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", type: "smiles" },
  { label: "Ibuprofen (formula)", query: "C13H18O2", type: "formula" },
  { label: "Benzene (SMILES)", query: "c1ccccc1", type: "smiles" },
];

export default function InputForm({ onSubmit, loading }: Props) {
  const [query, setQuery] = useState("");
  const [inputType, setInputType] = useState<InputType>("auto");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    onSubmit(query.trim(), inputType);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <div className="flex flex-col sm:flex-row gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="SMILES, drug name, or formula (e.g. Aspirin, CCO, C9H8O4)"
          className="flex-1 bg-graphite-800 border border-graphite-600 rounded-lg px-4 py-2.5
                     text-sm text-slate-100 placeholder:text-slate-500 font-mono
                     focus:outline-none focus:ring-2 focus:ring-accent-500/60"
        />
        <select
          value={inputType}
          onChange={(e) => setInputType(e.target.value as InputType)}
          className="bg-graphite-800 border border-graphite-600 rounded-lg px-3 py-2.5
                     text-sm text-slate-300 focus:outline-none focus:ring-2 focus:ring-accent-500/60"
        >
          <option value="auto">Auto-detect</option>
          <option value="smiles">SMILES</option>
          <option value="name">Drug / common name</option>
          <option value="formula">Molecular formula</option>
        </select>
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="px-5 py-2.5 rounded-lg bg-accent-600 hover:bg-accent-500 disabled:opacity-40
                     disabled:cursor-not-allowed text-graphite-950 font-semibold text-sm
                     transition-colors whitespace-nowrap"
        >
          {loading ? "Generating…" : "Generate Graph"}
        </button>
      </div>

      <div className="flex flex-wrap gap-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            onClick={() => {
              setQuery(ex.query);
              setInputType(ex.type);
            }}
            className="text-xs px-2.5 py-1 rounded-full bg-graphite-800 border border-graphite-600
                       text-slate-400 hover:text-accent-400 hover:border-accent-500/50 transition-colors"
          >
            {ex.label}
          </button>
        ))}
      </div>
    </form>
  );
}
