// Copyright (c) 2026 Aya Khaled Khuris. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for details.

// Molecular Graph Generator (GNN-Powered)
// Author: Aya Khaled Khuris <aya.khuris@gmail.com>
"use client";

import type { MoleculeInfo } from "@/lib/api";

export default function MoleculeInfoCard({ info }: { info: MoleculeInfo }) {
  const rows: [string, string | number][] = [
    ["Formula", info.molecular_formula],
    ["Molecular weight", `${info.molecular_weight} g/mol`],
    ["Heavy atoms", info.num_heavy_atoms],
    ["Total atoms (+H)", info.num_atoms],
    ["Bonds", info.num_bonds],
    ["Rings", info.num_rings],
    ["Resolved via", info.resolved_from],
  ];
  if (info.iupac_name) rows.push(["IUPAC name", info.iupac_name]);
  if (info.pubchem_cid) rows.push(["PubChem CID", info.pubchem_cid]);

  return (
    <div className="bg-graphite-800 border border-graphite-700 rounded-xl p-4 space-y-3">
      <div>
        <div className="text-xs text-slate-500 uppercase tracking-wide mb-1">Canonical SMILES</div>
        <code className="text-sm text-accent-400 break-all">{info.canonical_smiles}</code>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
        {rows.map(([label, value]) => (
          <div key={label} className="contents">
            <span className="text-slate-500">{label}</span>
            <span className="text-slate-200 text-right font-mono">{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
