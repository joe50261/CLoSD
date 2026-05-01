# closd/poc — WebGPU port proof-of-concept

Goal: prove DiP can run in a browser (WebGPU via ONNX Runtime Web) with output parity to the Python reference, then port the full DiP tool suite (generate / edit / invert / SMPL) on top.

Plan: `/root/.claude/plans/mac-demo-scalable-iverson.md` (14-day, two-phase: 7-day core PoC + 7-day tool-suite extension).

Spec: [SAMPLER_NOTES.md](./SAMPLER_NOTES.md) — algorithmic contract for the TypeScript port. Read this first.

## Layout

```
closd/poc/
├── SAMPLER_NOTES.md          Day 1 — sampler spec (READ FIRST)
├── export_onnx.py            Day 2 — export DiP trunk to ONNX
├── generate_fixtures.py      Day 2 — produce parity reference tensors
├── artifacts/                (gitignored) export outputs: .onnx, .args.json
├── fixtures/                 (gitignored) parity reference tensors per phase
└── web/                      Day 3-6 — TypeScript browser app (not yet created)
```

## Phase 1 (Days 1–7): core text-to-motion PoC

### Step 1 — produce ONNX + fixtures (one-time, requires conda env)

```bash
conda activate closd
python -m closd.poc.export_onnx --fp16
python -m closd.poc.generate_fixtures --phase core
```

Outputs: `closd/poc/artifacts/dip_no_target.onnx` (FP32 + FP16) and `closd/poc/fixtures/phase1_core/` (~1 MB of `.npy` tensors).

If the conda env doesn't have onnx tooling installed, the export script falls back to skipping the optional CPU sanity check and FP16 conversion — both can be re-run later.

### Step 2 — browser-side parity tests (Days 3–6)

Not yet implemented. Will be in `closd/poc/web/`.

## Phase 2 (Days 8–14): tool-suite extension

Each tool reuses Phase 1's runtime and adds a `--phase <name>` to `generate_fixtures.py`:

| Day | Tool | `--phase` | `--ckpt` |
|---|---|---|---|
| 8–9 | Goal-conditioned generation | `goal` | `multi-target` |
| 10–11 | In-between editing | `inbetween` | `no-target` |
| 12 | Upper-body editing | `upper` | `no-target` |
| 13 | DDIM inversion | `invert` | `no-target` |
| 14 | SMPL extraction (post-process) | `smpl` | `no-target` |

The Phase 2 functions in `generate_fixtures.py` currently `raise NotImplementedError` and will be filled in on the relevant days.

## Final deliverable (end of Day 14)

A static site (`closd/poc/web/dist/`) plus the exported `.onnx` weights — served by any static host. End users open a URL and run all Phase 1 + Phase 2 tools in their browser. **No Python at runtime.**
