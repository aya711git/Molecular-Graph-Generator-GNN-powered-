"use client";

import { useMemo, useState } from "react";
import type { MolecularGraph } from "@/lib/api";

const ATOM_COLORS: Record<string, string> = {
  C: "#94a3b8",
  N: "#60a5fa",
  O: "#f87171",
  S: "#fbbf24",
  F: "#4ade80",
  Cl: "#4ade80",
  Br: "#c084fc",
  P: "#fb923c",
  H: "#e5e7eb",
};

function atomColor(symbol: string): string {
  return ATOM_COLORS[symbol] ?? "#2dd4bf";
}

interface Props {
  graph: MolecularGraph;
  highlightedAtom?: number | null;
  onSelectAtom?: (index: number | null) => void;
}

/**
 * Renders an interactive SVG depiction of the molecular graph using the
 * 2D coordinates computed server-side by RDKit. Falls back to a simple
 * circular layout if coordinates are missing (e.g. layout failed).
 *
 * No external charting library is required — nodes/edges are drawn as
 * plain SVG primitives so this component has zero extra runtime deps.
 */
export default function MoleculeGraph({ graph, highlightedAtom, onSelectAtom }: Props) {
  const [hovered, setHovered] = useState<number | null>(null);

  const positions = useMemo(() => {
    const hasCoords = graph.nodes.every((n) => n.x != null && n.y != null);
    if (hasCoords) {
      return graph.nodes.map((n) => ({ x: n.x as number, y: n.y as number }));
    }
    // Fallback: circular layout.
    const n = graph.nodes.length;
    return graph.nodes.map((_, i) => ({
      x: Math.cos((2 * Math.PI * i) / n),
      y: Math.sin((2 * Math.PI * i) / n),
    }));
  }, [graph.nodes]);

  const { viewBox, scaled } = useMemo(() => {
    const xs = positions.map((p) => p.x);
    const ys = positions.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const pad = 1.5;
    const w = Math.max(maxX - minX, 0.1) + pad * 2;
    const h = Math.max(maxY - minY, 0.1) + pad * 2;
    const scale = 60; // px per graph unit
    const scaled = positions.map((p) => ({
      x: (p.x - minX + pad) * scale,
      y: (maxY - p.y + pad) * scale, // flip Y for screen coords
    }));
    return {
      viewBox: `0 0 ${w * scale} ${h * scale}`,
      scaled,
    };
  }, [positions]);

  const activeAtom = highlightedAtom ?? hovered;

  return (
    <div className="w-full h-full flex items-center justify-center bg-graphite-900 rounded-xl border border-graphite-700 overflow-hidden">
      <svg viewBox={viewBox} className="w-full h-full max-h-[520px]" role="img" aria-label="Molecular graph">
        {/* Edges (bonds) */}
        {graph.edges.map((edge, i) => {
          const a = scaled[edge.source];
          const b = scaled[edge.target];
          if (!a || !b) return null;
          const isHighlighted = activeAtom === edge.source || activeAtom === edge.target;
          const strokeWidth = edge.bond_type.includes("DOUBLE")
            ? 0
            : edge.bond_type.includes("TRIPLE")
            ? 0
            : 2;

          // Render double/triple bonds as parallel lines for clarity.
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const len = Math.sqrt(dx * dx + dy * dy) || 1;
          const nx = (-dy / len) * 3;
          const ny = (dx / len) * 3;

          const lineCount = edge.bond_type.includes("TRIPLE")
            ? 3
            : edge.bond_type.includes("DOUBLE")
            ? 2
            : 1;

          const offsets =
            lineCount === 1
              ? [0]
              : lineCount === 2
              ? [-1, 1]
              : [-1.4, 0, 1.4];

          return (
            <g key={`edge-${i}`}>
              {offsets.map((off, j) => (
                <line
                  key={j}
                  x1={a.x + nx * off}
                  y1={a.y + ny * off}
                  x2={b.x + nx * off}
                  y2={b.y + ny * off}
                  stroke={isHighlighted ? "#5eead4" : edge.is_aromatic ? "#94a3b8" : "#475569"}
                  strokeWidth={isHighlighted ? 3 : 2}
                  strokeDasharray={edge.is_aromatic ? "0" : undefined}
                  opacity={edge.is_aromatic ? 0.85 : 1}
                />
              ))}
            </g>
          );
        })}

        {/* Nodes (atoms) */}
        {graph.nodes.map((node, i) => {
          const p = scaled[i];
          if (!p) return null;
          const isActive = activeAtom === node.index;
          const radius = node.symbol === "C" ? 14 : 16;
          return (
            <g
              key={`node-${i}`}
              transform={`translate(${p.x}, ${p.y})`}
              className="cursor-pointer"
              onMouseEnter={() => setHovered(node.index)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => onSelectAtom?.(isActive ? null : node.index)}
            >
              <circle
                r={radius}
                fill={node.symbol === "C" ? "#181c22" : atomColor(node.symbol)}
                stroke={isActive ? "#5eead4" : "#334155"}
                strokeWidth={isActive ? 3 : 1.5}
              />
              <text
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={13}
                fontWeight={600}
                fill={node.symbol === "C" ? "#94a3b8" : "#0b0d10"}
                className="select-none pointer-events-none font-mono"
              >
                {node.symbol}
              </text>
              {node.formal_charge !== 0 && (
                <text
                  x={radius * 0.7}
                  y={-radius * 0.7}
                  fontSize={10}
                  fill="#fbbf24"
                  className="select-none pointer-events-none"
                >
                  {node.formal_charge > 0 ? `+${node.formal_charge}` : node.formal_charge}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
