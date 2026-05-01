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
    // BASE_URL is "/" in dev and "/CLoSD/" on GitHub Pages — Vite injects it
    // from the `base` config at build time.
    const base = import.meta.env.BASE_URL;
    const report = await runParity({
      fixturesBaseUrl: `${base}fixtures/phase1_core`,
      modelUrl: `${base}artifacts/dip_no_target.fp16.onnx`,
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
