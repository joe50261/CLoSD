import { defineConfig } from "vite";

// Base path for the deployed site. GitHub Pages serves under /<repo-name>/,
// so for joe50261/CLoSD the base is "/CLoSD/". Override via VITE_BASE for
// other hosts (e.g. "/" for local dev or a custom domain).
const base = process.env.VITE_BASE ?? "/";

export default defineConfig({
  base,
  server: { port: 5173 },
  test: {
    globals: true,
    environment: "node",
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web", "@huggingface/transformers"],
  },
});
