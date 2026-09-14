"use client";

import type { EmbeddingResult } from "@/lib/api";

interface Props {
  result: EmbeddingResult;
  selectedAtom: number | null;
}

/** Maps a value in [-1, 1]-ish range to a teal-magenta diverging color. */
function valueToColor(v: number, maxAbs: number): string {
  const t = maxAbs === 0 ? 0 : Math.max(-1, Math.min(1, v / maxAbs));
  if (t >= 0) {
    // teal for positive
    const alpha = 0.15 + 0.85 * t;
    return `rgba(45, 212, 191, ${alpha.toFixed(2)})`;
  }
  // magenta for negative
  const alpha = 0.15 + 0.85 * -t;
  return `rgba(232, 121, 249, ${alpha.toFixed(2)})`;
}

function Heatmap({ vector, label }: { vector: number[]; label: string }) {
  const maxAbs = Math.max(1e-6, ...vector.map((v) => Math.abs(v)));
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs uppercase tracking-wide text-slate-400">{label}</span>
        <span className="text-xs text-slate-500 font-mono">{vector.length} dims</span>
      </div>
      <div
        className="grid gap-[2px] p-2 bg-graphite-900 rounded-lg border border-graphite-700"
        style={{ gridTemplateColumns: "repeat(16, minmax(0, 1fr))" }}
      >
        {vector.map((v, i) => (
          <div
            key={i}
            title={`dim ${i}: ${v.toFixed(4)}`}
            className="aspect-square rounded-[2px]"
            style={{ backgroundColor: valueToColor(v, maxAbs) }}
          />
        ))}
      </div>
    </div>
  );
}

export default function EmbeddingPanel({ result, selectedAtom }: Props) {
  const atomVector =
    selectedAtom != null ? result.node_embeddings[selectedAtom] : undefined;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-3 text-xs font-mono">
        <Badge label="layer" value={result.layer_type.toUpperCase()} />
        <Badge label="layers" value={String(result.num_layers)} />
        <Badge label="dim" value={String(result.embedding_dim)} />
      </div>

      <Heatmap vector={result.graph_embedding} label="Graph-level embedding" />

      {atomVector ? (
        <Heatmap
          vector={atomVector}
          label={`Atom #${selectedAtom} (${result.graph.nodes[selectedAtom]?.symbol}) embedding`}
        />
      ) : (
        <p className="text-xs text-slate-500 italic">
          Click an atom in the graph to inspect its individual latent vector.
        </p>
      )}
    </div>
  );
}

function Badge({ label, value }: { label: string; value: string }) {
  return (
    <span className="px-2 py-1 rounded-md bg-graphite-800 border border-graphite-700 text-slate-300">
      <span className="text-slate-500">{label}:</span> {value}
    </span>
  );
}
