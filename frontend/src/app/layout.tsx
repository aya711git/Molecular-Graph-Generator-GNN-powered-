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
