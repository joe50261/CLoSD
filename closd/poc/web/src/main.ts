// Browser entry point. Minimal UI: a Run button, a console-style log area,
// and a results panel. Plan §"UI": "textarea + Run button" — no styling work
// beyond legibility, no error handling beyond "show the error".

import { formatReport, runParity } from "./parity.js";

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} not in DOM`);
  return el as T;
};

const logEl = $<HTMLPreElement>("log");
const resultEl = $<HTMLPreElement>("result");
const runBtn = $<HTMLButtonElement>("run");

function log(msg: string): void {
  const line = document.createElement("div");
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

runBtn.addEventListener("click", async () => {
  runBtn.disabled = true;
  resultEl.textContent = "";
  logEl.textContent = "";
  log("starting parity run");
  try {
    const report = await runParity({
      // Vite serves static files from /public; we drop fixtures/ + artifacts/
      // there at build time (or symlink locally for dev).
      fixturesBaseUrl: "/fixtures/phase1_core",
      modelUrl: "/artifacts/dip_no_target.fp16.onnx",
      seed: 10,
      threshold: 5e-3,
      onProgress: log,
    });
    resultEl.textContent = formatReport(report);
  } catch (e) {
    log(`ERROR: ${e instanceof Error ? e.message : String(e)}`);
    if (e instanceof Error && e.stack) {
      resultEl.textContent = e.stack;
    }
  } finally {
    runBtn.disabled = false;
  }
});
