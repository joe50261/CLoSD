# Plan: CLoSD WebGPU PoC (Technical Validation)

## Context

Full WebGPU port of CLoSD has been evaluated (previous revision of this doc): Tier A (DiP-only viewer) is viable, Tier C/D (closed-loop with physics) is multi-month. Before committing to any tier, we want a **1-week technical validation** that answers one question: **can DiP actually run in a browser via WebGPU with output parity to the Python reference?**

Everything downstream (rendering, UI, goal conditioning, physics) depends on this yes/no. If the answer is no, we know before spending weeks on three.js and asset pipelines.

## Success criteria (explicit go / no-go)

The PoC is **successful** iff all four hold on a Mac Mini M4, Chrome stable, WebGPU enabled:

1. **Loads**: the exported DiP ONNX loads in ORT Web with the `webgpu` execution provider, no fallback to WASM.
2. **Parity**: given a fixed text prompt, fixed initial noise tensor, and fixed diffusion schedule, the browser's final 263-dim motion vector matches the Python reference within **mean abs err < 1e-3** per element (FP16 inference tolerance). Intermediate per-step tensors match within **< 5e-3** mean abs err.
3. **Autoregressive**: 2 autoregressive iterations (context carries over between them) still match Python within the same tolerance.
4. **Performance**: single 10-step CFG'd sample completes in **< 2 s** wall time on M4 (soft target — informational, not a gate unless it's > 10 s, which would kill Tier A).

The PoC **fails** if parity diverges and cannot be root-caused within 2 days of debugging (see decision points below).

**Explicitly out of scope for this PoC:** rendering, UI beyond a textarea + Run button, physics, PHC policy, multi-repetition, mobile.

## Scope — two phases

**Phase 1 (core PoC, days 1–7):** prove the core DiP sampler works in the browser with parity to Python, using the simplest mode (autoregressive text-to-motion). This is the only phase with a go/no-go gate. If it fails, all tools below are also dead.

**Phase 2 (tool-suite extension, days 8–14):** once core parity holds, port each of the remaining DiP "tools" on top of the same runtime. This is where "工具也移植" gets delivered. Each tool is a thin add-on over Phase 1 infrastructure — different conditioning, different masking, same model + same scheduler. Parity-tested one by one.

### Phase 1 scope (core)

- Export **one** DiP checkpoint: `DiP_no-target_10steps_context20_predict40/model000200000.pt` (autoregressive, no target).
- Text encoder: use **transformers.js** CLIP ViT-B/32 (WebGPU) — avoids re-implementing the tokenizer. Verify it matches the Python CLIP output bit-for-bit at load time.
- Diffusion: re-implement the sampler in TypeScript. ~100 LoC. Keep the full loop in JS, not in ONNX — standard and easier to debug.
- CFG: re-implement in TS. Two-pass denoise + linear blend.
- Output: raw 263-dim HumanML3D motion vector, logged to console / dumped as JSON. **No visualization.**

### Phase 2 scope (tools)

Each bullet = one additional tool to port. Listed with reference file, delta over Phase 1, and parity strategy.

1. **Goal-conditioned generation** — ref: `closd/diffusion_planner/sample/generate.py` with `--sampling_mode goal --target_joint_source random`.
   Delta: second checkpoint export (`DiP_multi-target_10steps_context20_predict40/model000300000.pt`); additional model inputs for target joint (traj / heading / wrist / foot); target packing in TS.
   Parity: same harness, fixed random target tensor, MAE vs Python fixture.
   Est: ~2 days.

2. **In-between editing** — ref: `closd/diffusion_planner/sample/edit.py` with `--edit_mode in_between`.
   Delta: inpainting-style sampler (mask + known prefix/suffix frames conditioned in the denoise loop). Model weights unchanged. Scheduler gains a `q_sample` forward-noising helper.
   Parity: supply fixed prefix + suffix frames, check middle-frame reconstruction MAE.
   Est: ~2 days.

3. **Upper-body editing** — ref: `closd/diffusion_planner/sample/edit.py` with `--edit_mode upper_body`.
   Delta: per-joint feature mask (263-dim partitioned by joint index) applied during the denoise loop. Same sampler as #2 but with a spatial mask instead of temporal.
   Est: ~1 day (once #2 lands).

4. **DDIM inversion** — ref: `closd/diffusion_planner/sample/ddim_invert.py`.
   Delta: reverse-direction sampler (data → noise), needed if later we want to edit existing motion clips from the user. Adds a `ddim_invert_step` function mirroring the forward DDIM step. No new model exports.
   Parity: round-trip test — invert a known motion to noise, re-sample forward, MAE against the original.
   Est: ~1–2 days.

5. **SMPL parameter extraction** — ref: `closd/utils/extract_smpl.py`.
   Delta: this is a **post-processing** tool (joints → SMPL pose/shape for mesh rendering). Not model inference — just numerical optimization / inverse kinematics. Port is needed for future SMPL-mesh rendering (Tier A's nicer variant). Can run in a WebWorker; no WebGPU needed.
   Risk: if `extract_smpl.py` depends on `pytorch3d` or a CUDA-only joints2smpl optimizer, porting means reimplementing the optimizer in TS — budget may balloon.
   Est: ~2 days if pure PyTorch; **flag as blocker** if `pytorch3d` / CUDA optimizer is in the hot path.

6. **Blender / IsaacGym-recording tools** — ref: `closd/blender/record2anim.py`, `closd/utils/extract_smpl.py` with IsaacGym recordings.
   Decision: **not ported**. Blender is a desktop app; IsaacGym recordings require IsaacGym to produce. These are authoring-side tools, not runtime-side; they stay in Python on Linux.

7. **Eval harness** (`closd/diffusion_planner/eval/eval_humanml.py`) — **not ported**. Eval runs on a 1.3 GB dataset and computes FID-style metrics against a reference model; it's a research workflow, not a user-facing tool. Keep in Python.

Phase 2 therefore ports tools 1–5, skips 6–7 with explicit rationale.

## Work breakdown (14 working days: 7 core + 7 tools)

### Phase 1 — core PoC (days 1–7)

### Day 1 — Characterize the sampler (read-only)

Read and document the exact algorithm:
- `closd/diffusion_planner/sample/generate.py` — top-level call sequence
- `closd/diffusion_planner/utils/sampler_util.py` — `AutoRegressiveSampler`, `ClassifierFreeSampleModel`
- `closd/diffusion_planner/diffusion/` — exact sampler (DDPM `p_sample` vs DDIM), beta schedule, `posterior_mean_coef` formulas
- `closd/diffusion_planner/model/mdm.py` — `forward(x, timesteps, y)` signature; what `y` contains (text emb, mask, prefix, lengths)

Output: `closd/poc/SAMPLER_NOTES.md` — a ≤ 1 page reference so the JS port doesn't guess. Include the exact DDIM/DDPM step equation, the CFG blend formula, the AR context-concat rule, and the shape of every model input.

### Day 2 — Export ONNX + fixtures

New file `closd/poc/export_onnx.py`:
- Load DiP via `get_dependencies()` (`closd/utils/hf_handler.py`) — no new download plumbing.
- Export `model.forward(x_t, timesteps, y_dict)` to ONNX. Dynamic axes on batch + seq len. FP32 first (FP16 later if time permits).
- Strip CLIP from the graph — export DiP's trunk only. Text embeddings become an input.
- Verify the ONNX model runs under `onnxruntime` CPU EP and matches PyTorch output within 1e-5 (ONNX export sanity check).

New file `closd/poc/generate_fixtures.py`:
- Fix `torch.manual_seed(0)` and prompt = `"a person walks forward and waves"`.
- Save: initial noise `x_T.npy`, CLIP text embedding `text_emb.npy`, per-step intermediate `x_t` for all 10 steps, final motion `motion_ref.npy`. JSON sidecar with shapes/dtypes.
- Also save the CLIP text embedding from Python, so Day 3 can verify transformers.js reproduces it before we trust it.

### Day 3 — Browser runtime skeleton

New dir `closd/poc/web/`:
- `package.json` — deps: `onnxruntime-web`, `@xenova/transformers` (transformers.js), `vite`, `typescript`.
- `vite.config.ts` — COOP/COEP headers (required for WebGPU + SharedArrayBuffer in some browsers), static serving of `/fixtures/`.
- `index.html` — `<textarea>` + `<button>` + `<pre>` for output.
- `src/scheduler.ts` — DDPM/DDIM step, beta schedule, from Day 1's notes. Pure functions over `Float32Array`.
- `src/cfg.ts` — classifier-free guidance blend.
- `src/ar_loop.ts` — autoregressive context update.
- `src/main.ts` — wires CLIP (transformers.js, WebGPU) + DiP (ORT Web, WebGPU) + scheduler + CFG + AR, logs raw 263-dim output.

Get the path running end-to-end without parity checks first — just "run produces numbers of the right shape".

### Day 4 — CLIP parity

Before testing the sampler end-to-end, confirm **CLIP text embedding parity** between transformers.js and Python:
- Load the Day-2 reference `text_emb.npy`.
- Tokenize and encode the same prompt in the browser via transformers.js CLIP ViT-B/32 WebGPU.
- Compute mean abs err.

If this exceeds 1e-3, the root cause is almost always the tokenizer (special tokens, truncation, lowercase). Fix by loading the Python tokenizer output as input instead. This is a known gotcha — budget for it.

### Day 5 — End-to-end parity

- Feed Day-2 `x_T.npy` noise in browser (override `Math.random`; do not generate noise in JS for the parity run).
- Run full 10-step + CFG loop.
- Compare every intermediate `x_t` against the Python fixture.
- If divergence happens at step `k`, the bug is localized to that step.

Likely failure modes and where to look first:
- **Step 1 diverges**: wrong beta schedule, wrong timestep encoding, FP precision at `sqrt(alpha_cumprod)`.
- **CFG-scaled output wrong**: CFG formula sign or weight mismatch vs `sampler_util.py`.
- **Later steps diverge but first step OK**: noise schedule accumulating drift — acceptable up to ~5e-3, escalate if more.
- **AR divergence**: context-carry rule in `closd/diffusion_planner/utils/sampler_util.py` `AutoRegressiveSampler` class transcribed wrong into TS.

### Day 6 — Autoregressive + perf

- Run 2 AR iterations, parity-check each.
- Measure: total wall time, time-per-step, WebGPU EP confirmation via `chrome://gpu`.
- Try FP16 ONNX export. If parity still within tolerance, keep FP16 (halves download).

### Day 7 — Phase 1 writeup + go/no-go

- `closd/poc/RESULTS.md` — parity numbers, perf numbers, bug post-mortems, WebGPU compatibility observed.
- Decision: does the core sampler work? If no → stop, do not start Phase 2. If yes → continue.

### Phase 2 — tool suite (days 8–14)

Each tool below is additive on top of Phase 1's runtime. Parity harness from Phase 1 is reused for each.

### Day 8–9 — Goal-conditioned generation
- Export second checkpoint via `closd/poc/export_onnx.py --ckpt multi-target` (extend the export script to accept a ckpt id).
- Generate fixtures for random-target mode (fixed seed → fixed sampled target → reference output).
- Add target-joint packing in `src/conditioning.ts`: encode `target_joint_names` + joint index + target xyz/rotmat into the model input dict.
- Parity-check against Python.

### Day 10–11 — In-between editing
- Extend `closd/poc/export_onnx.py` to also export the base DiP trunk (no-target ckpt is reused; `edit.py` uses the same weights).
- Add `src/inpaint.ts`: mask + `q_sample`-forward-noised known frames injected at each denoise step (transcribe from `closd/diffusion_planner/diffusion/*` inpaint paths).
- Fixtures: fixed prefix + suffix frames → reference middle-frame output.
- Parity-check.

### Day 12 — Upper-body editing
- Add joint-feature mask (263-dim partitioned — consult `closd/diffusion_planner/data_loaders/humanml/common/skeleton.py` for the partition indices).
- Reuse `src/inpaint.ts` but swap the temporal mask for a feature mask.
- Parity-check.

### Day 13 — DDIM inversion
- Add `ddim_invert_step` to `src/scheduler.ts` (reverse direction of the DDIM step from Phase 1 Day 3).
- Round-trip test: known motion → invert to noise → sample forward → MAE against original.

### Day 14 — SMPL extraction + final writeup
- First look at `closd/utils/extract_smpl.py`: if it's a pure-PyTorch optimizer, port to TS using a small LBFGS / Adam in `src/smpl_extract.ts`.
- **If it depends on `pytorch3d` or CUDA-specific joints2smpl code, stop and flag** — we'll either re-implement joints→SMPL from scratch (another 3–5 days) or ship Phase 2 without this tool.
- Update `closd/poc/RESULTS.md` with tool-by-tool parity numbers and any open blockers.

## Files to create (all net-new, no existing files modified)

```
closd/poc/
├── SAMPLER_NOTES.md               # Day 1 spec
├── export_onnx.py                 # Day 2, extended on Day 8 for multi-target
├── generate_fixtures.py           # Day 2, extended for each tool
├── fixtures/                      # Day 2+ outputs (gitignored if > 50MB)
│   ├── phase1_core/               # x_T, text_emb, x_t_step{00..09}, motion_ref
│   ├── phase2_goal/               # goal-conditioned reference tensors
│   ├── phase2_inbetween/          # prefix, suffix, middle_ref
│   ├── phase2_upper/              # feature-mask reference tensors
│   ├── phase2_invert/             # round-trip reference tensors
│   └── phase2_smpl/               # joints_in, smpl_params_ref
├── web/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
│       ├── main.ts                # Phase 1 entry
│       ├── scheduler.ts           # Phase 1 + Day 13 (ddim_invert_step)
│       ├── cfg.ts                 # Phase 1
│       ├── ar_loop.ts             # Phase 1
│       ├── ort_setup.ts           # Phase 1 — WebGPU EP config, model load
│       ├── conditioning.ts        # Day 8–9 — target-joint packing
│       ├── inpaint.ts             # Day 10–12 — temporal + feature masking
│       └── smpl_extract.ts        # Day 14 — joints → SMPL (if not blocked)
├── RESULTS.md                     # Day 7 + Day 14
└── README.md                      # how to reproduce, per-tool invocation
```

## Existing utilities — dev-time reuse only (NOT shipped to browser)

Clarification: the final browser deliverable is a static bundle (`closd/poc/web/dist/` + exported `.onnx` weights). **No Python code ships to end users.** The references below are used in one of two ways:

**(a) Dev-time Python script dependencies** — called by `closd/poc/export_onnx.py` and `closd/poc/generate_fixtures.py` on a developer machine to produce artifacts (ONNX files, `.npy` fixtures). Run once per checkpoint revision, never in the browser.

- `closd/utils/hf_handler.py` `get_dependencies()` — checkpoint auto-download inside the export script.
- `closd/diffusion_planner/utils/model_util.py` `create_model_and_diffusion()` — model instantiation for ONNX export.

**(b) Reference-only (read, transcribe, do not import)** — we read the source to understand the algorithm, then re-implement it in TypeScript. The Python file is never called, even at dev time.

- `closd/diffusion_planner/utils/sampler_util.py` `ClassifierFreeSampleModel`, `AutoRegressiveSampler` — read → re-implement in TS.
- `closd/diffusion_planner/diffusion/` — sampling math reference for `scheduler.ts` (including inpaint + DDIM invert).
- `closd/diffusion_planner/sample/edit.py` — mask construction for in-between + upper-body editing (Day 10–12 reference).
- `closd/diffusion_planner/sample/ddim_invert.py` — inversion step math (Day 13 reference).
- `closd/diffusion_planner/data_loaders/humanml/common/skeleton.py` — joint-index partition for Day 12 upper-body mask (values get hardcoded as a TS const).
- `closd/utils/extract_smpl.py` — Day 14; port to TS if pure-PyTorch, escalate if it pulls in `pytorch3d` / CUDA.

## Final deliverable (what actually ships)

- Static site: `closd/poc/web/dist/index.html` + bundled JS (ORT Web, transformers.js, scheduler/cfg/ar_loop/conditioning/inpaint/smpl_extract TS).
- Exported weights: one or two `.onnx` files (DiP no-target trunk; optionally DiP multi-target trunk for Phase 2 tool 1).
- No Python, no CUDA, no server-side inference.

## Verification

Two distinct workflows. Python is needed **once** to produce artifacts; after that, verification is browser-only.

### (A) One-time artifact production (developer machine, Python, run once per checkpoint revision)

These commands produce `.onnx` weights and `.npy` parity fixtures. Output is committed/bundled into `closd/poc/` so subsequent verification does not need Python.

```bash
conda activate closd
python -m closd.poc.export_onnx                       # Phase 1 core
python -m closd.poc.generate_fixtures --phase core    # Phase 1 core
# Phase 2, per tool (run only once, outputs checked in):
python -m closd.poc.export_onnx --ckpt multi-target
python -m closd.poc.generate_fixtures --phase goal
python -m closd.poc.generate_fixtures --phase inbetween
python -m closd.poc.generate_fixtures --phase upper
python -m closd.poc.generate_fixtures --phase invert
python -m closd.poc.generate_fixtures --phase smpl
```

After this step, the repo contains everything the browser needs: `.onnx` weights + `.npy` reference tensors. If fixture files exceed the repo size budget (> 50 MB), they're uploaded to the checkpoint release asset bucket and fetched by the browser at runtime — still no Python involved.

### Artifact size estimates

Rough numbers (to be confirmed on Day 2 export):

| Artifact | Count | Per-file | Total |
|---|---|---|---|
| DiP ONNX (no-target, FP16) | 1 | ~40–60 MB | ~50 MB |
| DiP ONNX (multi-target, FP16) | 1 (Phase 2) | ~40–60 MB | ~50 MB |
| Parity fixtures `.npy` (Phase 1 core) | ~15 files | ~40 KB each | ~1 MB |
| Parity fixtures `.npy` (Phase 2 × 5 tools) | ~50 files | ~40 KB each | ~5 MB |
| **Static artifacts subtotal** | | | **~100–120 MB** |
| ORT Web runtime + WASM glue (bundled JS) | — | — | ~10 MB |
| transformers.js CLIP ViT-B/32 (HF CDN, first-load cached) | — | — | ~80 MB |
| App JS (TS compiled) | — | — | ~200 KB |
| **First page load subtotal** | | | **~200 MB** |

Assumptions: DiP is a transformer with latent_dim=512, 8 layers, ~20–25M params. FP16 export halves FP32 size. Confirm on Day 2 by exporting and running `du -h closd/poc/fixtures/ closd/poc/*.onnx`; if FP32 ONNX overshoots GitHub's 100 MB single-file limit, we use FP16 (preferred anyway for WebGPU) or Git LFS / release assets.

If total static artifact size > 150 MB after Day 2 measurement, switch to hosting `.onnx` on the HuggingFace repo that already hosts the checkpoints (`closd/utils/hf_handler.py` points at it) and fetch at browser init — same mechanism as CLIP.

### (B) Ongoing verification (browser only, no Python)

This is what's re-run on every code change, every contributor machine, every CI job:

```bash
cd closd/poc/web
npm install
npm run dev
# open http://localhost:5173
```

The browser:
1. Loads the bundled `.onnx` + fetches reference fixtures (static files).
2. Runs each tool tab (Phase 1 core, plus Phase 2 tools 1–5).
3. Computes MAE against the reference fixtures in JS and prints PASS/FAIL per tool to the UI and console.

Then open `closd/poc/RESULTS.md` and confirm:
- Phase 1: all four core success criteria checked with numbers.
- Phase 2: per-tool parity MAE logged for tools 1–5; any skipped tool has explicit reason.
- `chrome://gpu` screenshot showing WebGPU enabled.
- `about:tracing` or `performance.mark()` log showing ORT Web used WebGPU EP (not WASM).

### End-user deployment (no Python ever)

The final demo is the built `closd/poc/web/dist/` plus the static `.onnx` files — served by any static host (GitHub Pages, S3, netlify). End users open a URL and run the tools in their browser. They never install Python, PyTorch, or CUDA.

## Decision points (hard rules)

- **By end of Day 3**, if ONNX export itself fails (unsupported op, shape inference breaks), try `torch.onnx.dynamo_export` — if that also fails, the model uses an op WebGPU/ONNX can't run → **stop, escalate, reconsider Tier A**.
- **By end of Day 5**, if CLIP parity is solved but DiP step-1 output still diverges > 1e-2, spend Day 6 root-causing; if still unresolved, **stop, write post-mortem, reconsider**.
- **By end of Day 6**, if perf > 10 s per sample on M4 WebGPU, Tier A is user-hostile — consider distillation or smaller model before committing to Tier A implementation.
- **By end of Day 7**, if Phase 1 gates not met → **do not start Phase 2**. The tool suite rests entirely on the core sampler.
- **By Day 14**, if SMPL extraction depends on `pytorch3d` or joints2smpl-CUDA, **do not re-implement on the fly**. Flag it, ship the other four tools, and treat SMPL extraction as a separate scoped task.

## Out of scope (still out of scope even with Phase 2)

- three.js or any rendering (Tier A work, follows a successful PoC)
- PHC policy, MuJoCo, physics of any kind (Tier C/D)
- UI polish, error handling, loading states
- mobile browsers
- weight download optimization (let browser cache unoptimized files)
- cross-browser testing — Chrome stable only for this PoC
- Blender visualization scripts and IsaacGym-recording consumers — authoring-side, stay in Python
- Evaluation harness (FID metrics, HumanML3D test set) — research workflow, stays in Python
