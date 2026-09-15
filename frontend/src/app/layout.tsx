// Copyright (c) 2026 Aya Khaled Khuris. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for details.

// Molecular Graph Generator (GNN-Powered)
// Author: Aya Khaled Khuris <aya.khuris@gmail.com>
import type { Metadata } from "next";
import "@/styles/globals.css";

export const metadata: Metadata = {
  title: "Molecular Graph Generator",
  description: "GNN-powered molecular graph generation and embedding visualization.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-graphite-950 text-slate-100 font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
