import { defineConfig } from "vite";

export default defineConfig({
  server: { port: 5173 },
  test: {
    globals: true,
    environment: "node",
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web", "@huggingface/transformers"],
  },
});
