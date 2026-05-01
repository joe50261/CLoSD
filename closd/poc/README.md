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

## Deployment to GitHub Pages

`.github/workflows/deploy-poc.yml` builds and deploys `closd/poc/web/` to GitHub Pages on every push to `main` (or the active feature branch) that touches `closd/poc/web/**`, `closd/poc/artifacts/**`, or `closd/poc/fixtures/**`. The deployed URL is `https://joe50261.github.io/CLoSD/`.

### One-time repo setup

In **Settings → Pages → Build and deployment**: set **Source = "GitHub Actions"**. The workflow uses `actions/deploy-pages@v4` and won't deploy until this is enabled.

### Providing artifacts and fixtures

The workflow expects:

- `closd/poc/artifacts/dip_no_target.fp16.onnx` — exported model
- `closd/poc/fixtures/phase1_core/*.npy` — parity reference tensors

Both directories are listed in `closd/poc/.gitignore`, so the workflow won't find them unless one of:

1. **Commit them directly** (override `.gitignore` for those paths). FP16 ONNX is ~50 MB which fits under GitHub's 100 MB single-file limit; fixtures are ~5 MB total. Simple, but adds permanent repo size.
2. **Use Git LFS** for the `.onnx` file. Better for repo hygiene; needs LFS quota.
3. **Add a CI step** that downloads the ONNX from a HuggingFace release or GitHub release before the build. Cleanest separation, but adds complexity.

For the PoC, option 1 is fine. After running `export_onnx.py` + `generate_fixtures.py` locally, force-add and commit:

```bash
git add -f closd/poc/artifacts/dip_no_target.fp16.onnx closd/poc/artifacts/dip_no_target.args.json
git add -f closd/poc/fixtures/phase1_core/
git commit -m "poc: add Phase 1 artifacts and fixtures"
git push
```

Without those files committed, the workflow still succeeds and the page deploys, but the parity harness will 404 on the model fetch.

### Manual deploy

`Actions → Deploy PoC to GitHub Pages → Run workflow` triggers a deploy without a commit.

## Final deliverable (end of Day 14)

A static site (`closd/poc/web/dist/`) plus the exported `.onnx` weights — served by GitHub Pages or any static host. End users open the URL and run all Phase 1 + Phase 2 tools in their browser. **No Python at runtime.**
