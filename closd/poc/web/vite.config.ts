import { defineConfig } from "vite";
import { execSync } from "node:child_process";

// Base path for the deployed site. GitHub Pages serves under /<repo-name>/,
// so for joe50261/CLoSD the base is "/CLoSD/". Override via VITE_BASE for
// other hosts (e.g. "/" for local dev or a custom domain).
const base = process.env.VITE_BASE ?? "/";

// Stamp the bundle with the current commit + timestamp so a stale cache
// is visible to anyone reading the harness log.
function buildStamp(): string {
  const sha = (() => {
    try {
      return execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim();
    } catch {
      return "unknown";
    }
  })();
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  return `${sha}@${ts}`;
}

export default defineConfig({
  base,
  define: {
    __BUILD_STAMP__: JSON.stringify(buildStamp()),
  },
  server: { port: 5173 },
  test: {
    globals: true,
    environment: "node",
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web", "@huggingface/transformers"],
  },
});
