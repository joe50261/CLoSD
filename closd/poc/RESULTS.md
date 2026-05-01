# Phase 1 PoC — Results

End-of-Phase-1 deliverable. **This document is a template.** It's filled in by running the harness on the target hardware and pasting the report below. The go/no-go decision at the end gates whether Phase 2 starts.

> Status as of last commit: **NOT YET RUN**. All code and tests pass; nothing has executed against the actual ONNX trunk on WebGPU.

## How to reproduce

### One-time artifact production (developer machine, Python, ~20 min)

Requires the `closd` conda env with torch + clip + onnx + onnxruntime + onnxconverter-common + huggingface_hub installed.

```bash
conda activate closd
cd <repo root>
python -m closd.poc.export_onnx --fp16
python -m closd.poc.generate_fixtures --phase core
```

Produces:
- `closd/poc/artifacts/dip_no_target.onnx` (FP32, ~80–100 MB)
- `closd/poc/artifacts/dip_no_target.fp16.onnx` (FP16, ~40–50 MB)
- `closd/poc/artifacts/dip_no_target.args.json`
- `closd/poc/fixtures/phase1_core/` (~1 MB of `.npy` reference tensors)

### Browser parity run (Mac Mini M4, Chrome stable, WebGPU enabled)

```bash
cd closd/poc/web
mkdir -p public
ln -sf ../../fixtures public/fixtures
ln -sf ../../artifacts public/artifacts
npm run dev
# open http://localhost:5173 → click "Run parity check"
```

The browser fetches fixtures + the ONNX file from the static dev server, runs the full TS pipeline, and prints the report. Paste the report below.

## Success-criteria checklist

Per `/root/.claude/plans/mac-demo-scalable-iverson.md` §"Success criteria":

- [ ] **Loads**: ONNX loads on WebGPU EP, no fallback to WASM. (`executionProvider: webgpu` in the report header.)
- [ ] **Parity (final motion)**: `motion_iter0` MAE < **1e-3** per element (FP16 tolerance).
- [ ] **Parity (intermediate)**: every `iter0/x_t_step*` and `iter0/pred_xstart_step*` MAE < **5e-3**.
- [ ] **Autoregressive**: `iter0` and `iter1` both pass the same thresholds (verified end-to-end via the full AR sample).
- [ ] **CLIP**: `text_embed` (CLIP browser vs Python) MAE < **1e-3** (FP32) or **5e-3** (FP16).
- [ ] **Performance (soft)**: full AR sample (5 iters × 10 steps × 2 CFG passes = 100 trunk calls) completes in < **2 s** wall time. Anything > 10 s is a hard fail per plan §"Decision points".

## Report

> Paste the output of `formatReport(report)` from the browser harness.

```
Execution provider: <fill>
Full AR sample: <fill>ms (per-step avg <fill>ms over 50 steps)

  [PASS/FAIL] text_embed (CLIP): MAE=… (< 5e-3)
  [PASS/FAIL] iter0/x_t_step00: MAE=… (< 5e-3)
  …
  [PASS/FAIL] iter0/pred_xstart_step09: MAE=… (< 5e-3)
  [PASS/FAIL] motion_iter0: MAE=… (< 5e-3)

Summary: <N> passed, <M> failed
```

## WebGPU environment

> Capture once on the test machine.

- [ ] `chrome://gpu` confirms WebGPU enabled (paste version + driver).
- [ ] Browser version: **<fill>**
- [ ] OS / hardware: **Mac Mini M4 / macOS <fill>**
- [ ] ORT Web version (from `npm ls onnxruntime-web`): **<fill>**

## Bug post-mortems

> One bullet per non-trivial issue encountered, with the fix and the decision.

- _(none yet)_

## Go / no-go decision

| Outcome | Action |
|---|---|
| **All gates green** | Proceed to Phase 2 Day 8: port goal-conditioned generation. Paste a one-liner to that effect below. |
| **Some intermediate parity drift but final motion < 1e-3** | Proceed; note the drift in the next phase's risk log. |
| **Final motion fails parity but is rootcause-able in < 2 days** | Spend Day 6 → Day 8 debugging; do not start Phase 2 features until parity is restored. |
| **WebGPU fallback to WASM** | **Stop.** Tier A (browser DiP) is dead in this configuration. Re-evaluate: distillation, smaller model, or different runtime (TFJS, candle, etc.). |
| **Perf > 10 s per sample** | **Stop.** Tier A is user-hostile. Distillation or quantization required before continuing. |

**Decision (date, signer):** _(fill in)_

**Reasoning (one paragraph):** _(fill in — what was learned, what's the next step.)_
