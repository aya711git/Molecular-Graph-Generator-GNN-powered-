import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        graphite: {
          950: "#0b0d10",
          900: "#111418",
          800: "#181c22",
          700: "#22272f",
          600: "#2e3540",
        },
        accent: {
          400: "#5eead4",
          500: "#2dd4bf",
          600: "#14b8a6",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Menlo", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
