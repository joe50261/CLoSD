# DiP Sampler — TypeScript port spec

Day 1 deliverable. Audience: the developer (or Claude) doing the TS implementation in Days 3–6. Every `file:line` reference is the authoritative source.

## TL;DR

DiP is a **transformer text-to-motion diffusion model** that predicts `x_start` (not `epsilon`) over **10 cosine-scheduled diffusion timesteps** in a 263-dim HumanML3D feature space. Inference loops are **DDIM (deterministic, eta=0)**, wrapped in **classifier-free guidance**, wrapped in an **autoregressive sampler** that produces 196 frames in 5 iterations of 40-frame predictions conditioned on a 20-frame rolling prefix. The TS port re-implements all three loops; only the trunk (transformer + heads) is exported to ONNX.

## 1. Checkpoint we're targeting

`DiP_no-target_10steps_context20_predict40/model000200000.pt`

Config (from checkpoint `args.json`, confirmed against parser defaults and model_util):

| Key | Value | Why it matters for TS |
|---|---|---|
| `diffusion_steps` | **10** (not 1000) | Beta schedule has only 10 entries; `timestep_map` is identity `[0..9]`; **no respacing collapse**. (`model_util.py:81`, `respace.py:31-39`) |
| `noise_schedule` | `"cosine"` | Beta computation per `betas_for_alpha_bar` (`gaussian_diffusion.py:49-66`). |
| `sigma_small` | `True` (parser default `parser_util.py:93`) | `model_var_type = FIXED_SMALL`; **DDIM with eta=0 doesn't use it** — port can ignore. |
| `latent_dim` | `512` | Transformer width. |
| `layers` | `8` | Transformer depth. |
| `cond_mode` | `"text"` | Text-only (no action, no target). |
| `cond_mask_prob` | `0.1` | Training-time only; CFG asserts > 0 (`sampler_util.py:16`). |
| `context_len` | `20` | AR rolling prefix length. |
| `pred_len` | `40` | AR predict-window length. |
| `multi_target_cond` | `False` | No target-joint head; the no-target ckpt skips that branch entirely. |
| `dataset` | `"humanml"` | → `njoints=263, nfeats=1, data_rep='hml_vec'` (`model_util.py:43-47`). |
| `arch` | `"trans_enc"` (typical) | Transformer encoder-only (vs `trans_dec`). Confirm at export time. |
| `emb_policy` | `"concat"` (parser default) | How `time_emb + text_emb` are fused (concat vs add). Confirm at export time. |

**Inference defaults** (`generate_args` / `parser_util.py:229`):
- `guidance_param = 7.5`
- `autoregressive = True` (set on CLI; checkpoint expects it)
- `autoregressive_include_prefix = False` (default; output drops the seed prefix)
- `num_repetitions = 1`

## 2. Model forward signature (what gets exported to ONNX)

Class: `MDM` (`closd/diffusion_planner/model/mdm.py:13`).

```python
def forward(self, x, timesteps, y={}):
    # x:         [B, 263, 1, T_pred]    float32   (T_pred = pred_len = 40)
    # timesteps: [B]                    int64     (values 0..9 for the 10-step model)
    # y: dict, keys actually used by the no-target ckpt:
    #   text_embed:  [1, B, 512] float32  (cached CLIP output; pre-encoded — see §3)
    #   text_uncond: bool                 (CFG uncond pass flag; see §4)
    #   mask:        [B, 1, 1, T_pred]    bool   (1=valid frame; padding=0)
    #   prefix:      [B, 263, 1, 20]      float32  (rolling context from AR loop)
    #   lengths:     [B] int              (motion length; only used in inv_transform path)
    # Returns: [B, 263, 1, 20+40]  float32   (model concats prefix internally — see below)
```

**Internal behavior we must mirror in TS** (`mdm.py:200-291`):
- `target_cond` branch — **skipped** (multi_target_cond=False).
- `is_prefix_comp` branch (`mdm.py:205-208`) — **active**. Model concatenates `y['prefix']` to `x` along the time dim and pads the mask with 20 ones on the left, so **inside** the transformer the sequence is 60 frames. After the transformer, the model strips the prefix back off (`mdm.py:290-291`: `output = output[self.context_len:]`), so **the returned tensor is `[1, 263, 1, 40]`**, not 60.
- Text branch (`mdm.py:221-237`) — uses `y['text_embed']` if present (cached). If `y.get('text_uncond')==True`, `mask_cond` zeros the text emb (`mdm.py:156-164`).

**Output is x_start**, not epsilon, not (mean, var). `model_mean_type = ModelMeanType.START_X` is hardcoded by `predict_xstart=True` (`model_util.py:80, 102`). FIXED_SMALL/LARGE only changes `posterior_variance` reporting, which DDIM with eta=0 ignores.

**Export shape contract for ONNX** (Day 2):
- Inputs: `x` `[1, 263, 1, 40]`, `timesteps` `[1]`, `text_embed` `[1, 1, 512]`, `mask` `[1, 1, 1, 40]`, `prefix` `[1, 263, 1, 20]`, `text_uncond_mask` `[1]` float32.
- Output: `pred_xstart` `[1, 263, 1, 40]` — already trimmed to the predict region.
- For the CFG uncond pass, set `text_uncond_mask = 1.0` (zeros the text branch after the embed Linear, mirroring `mask_cond(force_mask=True)`); for the cond pass, set it to `0.0`. **No re-tokenization needed for the uncond pass.** This is wired up via a monkey-patched `mask_cond` at export time — see `closd/poc/export_onnx.py`.

## 3. Text encoder — separate from the trunk

Lives in CLIP ViT-B/32 (`mdm.py:107-181`). For the port:

- **Use transformers.js** (`Xenova/clip-vit-base-patch32`) for tokenize + encode. WebGPU EP. Outputs the 512-dim text projection.
- The Python path is `clip.tokenize(text, context_length=22, truncate=True)` then `clip_model.encode_text(...).float()` → `[B, 512]` → unsqueeze to `[1, B, 512]`. transformers.js produces the same 512-dim projection (verified once at load — see Day 5 task).
- The 512→512 `embed_text` Linear and the `mask_cond` zeroing **stay inside the exported ONNX trunk**. So the TS pipeline is: `text → CLIP (transformers.js) → [1,1,512] → ONNX trunk input`.
- For the **CFG uncond pass**: `mask_cond` zeros the text embedding when `text_uncond=True`. Equivalent in TS: pass a zero tensor of shape `[1,1,512]` as `text_embed` for the second pass. **No need to re-tokenize an empty string.**

Tokenization context_length is 22 (`mdm.py:172` for HumanML3D). transformers.js default is 77; we need to either truncate to 22 or accept that CLIP itself zero-pads — they should produce identical pooled output because the [EOS] token's index drives the projection. Verify on Day 5.

## 4. Classifier-free guidance

`ClassifierFreeSampleModel.forward` (`sampler_util.py:21-31`):

```python
y_uncond = deepcopy(y); y_uncond['text_uncond'] = True
out_cond   = model(x, t, y)              # pass A
out_uncond = model(x, t, y_uncond)       # pass B
return out_uncond + scale * (out_cond - out_uncond)
```

**TS implementation** (`cfg.ts`):
```ts
const xCond   = await session.run({ x, timesteps, text_embed: clipEmb, mask, prefix });
const xUncond = await session.run({ x, timesteps, text_embed: zerosLike(clipEmb), mask, prefix });
return uncond + scale * (cond - uncond);   // elementwise on the [1,263,1,60] tensor
```

`scale` is broadcast as `[B,1,1,1]` in Python (`sampler_util.py:31`) — for B=1 it's a scalar. Default `7.5`.

**Two ONNX runs per denoise step.** With 10 steps × 5 AR iters × 2 = **100 trunk forward passes per 196-frame motion.** Day 6 perf will measure this.

## 5. DDIM step (the inner sampler loop)

Source: `gaussian_diffusion.py:806-856`. The eta=0 path simplifies dramatically.

**Precomputed schedule** (computed **once** at TS init, all length 10):
```ts
// betas: cosine schedule
// alpha_bar(s) = cos((s + 0.008) / 1.008 * pi/2)^2,  for s in [0,1]
// betas[i] = min(1 - alpha_bar((i+1)/N) / alpha_bar(i/N), 0.999)   (gaussian_diffusion.py:49-66)
const alphas = betas.map(b => 1 - b);
const alphasCumprod = cumprod(alphas);                            // [10]
const alphasCumprodPrev = [1.0, ...alphasCumprod.slice(0, -1)];   // [10]
const sqrtRecipAlphasCumprod   = alphasCumprod.map(a => Math.sqrt(1 / a));
const sqrtRecipm1AlphasCumprod = alphasCumprod.map(a => Math.sqrt(1 / a - 1));
```

**One DDIM step** (eta=0; deterministic; verified against `gaussian_diffusion.py:837-855`):

```ts
// Inputs: x_t [1,263,1,60], pred_xstart [1,263,1,60] (model output after CFG), t (int 0..9)
function ddimStep(x_t, pred_xstart, t):
    // _predict_eps_from_xstart, gaussian_diffusion.py:463-467
    const eps = (sqrtRecipAlphasCumprod[t] * x_t - pred_xstart) / sqrtRecipm1AlphasCumprod[t];

    const alpha_bar_prev = alphasCumprodPrev[t];
    // eta=0 → sigma=0 → noise term vanishes (line 855: sample = mean_pred + 0 * sigma * noise)
    const x_tm1 = pred_xstart * Math.sqrt(alpha_bar_prev)
                + eps * Math.sqrt(1 - alpha_bar_prev);   // sigma=0 simplification of line 850
    return x_tm1;
```

**Loop direction** (`gaussian_diffusion.py:p_sample_loop` / `ddim_sample_loop`):
```
indices = [9, 8, 7, ..., 1, 0]   // descending
x = randn([1,263,1,60])           // initial noise
for t in indices:
    pred_xstart_full = cfg(model, x, t, text_embed, ...)   // model output is x_start in last 40 frames
    # the model returns [1,263,1,60]; the prefix slice [..., :20] gets re-noised by the model itself
    # to keep prefix consistent — see §6 below
    x = ddimStep(x, pred_xstart_full, t)
return x[..., -40:]                # only the last 40 are the new prediction
```

**clip_denoised**: `False` for DiP (`generate.py:201`, `clip_denoised=False` is passed to the sampler). So **no `clamp(-1,1)` on pred_xstart**. Important — clipping would silently corrupt the output for valid HumanML3D ranges that exceed [-1,1].

## 6. Prefix handling inside the denoise loop

The model concatenates `y['prefix']` (clean) with `x` (noisy predict region) internally, runs the transformer over the full 60-frame sequence, then strips the prefix off before returning (`mdm.py:205-208` and `mdm.py:290-291`). So the **caller never sees 60 frames**:

- We pass `x` of shape `[1, 263, 1, 40]` (noisy predict region).
- We pass `prefix` of shape `[1, 263, 1, 20]` (clean rolling context).
- Model returns `pred_xstart` of shape `[1, 263, 1, 40]`. No slicing needed in TS.

**TS pseudocode** (verified against `gaussian_diffusion.ddim_sample` and `mdm.forward`):

```ts
// x is [1,263,1,40] throughout the denoise loop
let x = randn([1, 263, 1, 40]);
for (let t = 9; t >= 0; t--) {
    const predXstart = await cfgRun(x, t, prefix, textEmb);     // [1,263,1,40]
    x = ddimStep(x, predXstart, t);                              // operates on 40-frame x
}
return x;   // [1,263,1,40] — one AR iteration's prediction
```

The mask passed to the model is for the 40-frame predict region only; the model pads 20 ones onto the left internally.

## 7. Autoregressive loop

`AutoRegressiveSampler.sample` (`sampler_util.py:44-61`):

```ts
const requiredFrames = 196;
const nIterations = Math.floor(196 / 40) + 1;   // = 5
let prefix = initialPrefix.clone();              // [1,263,1,20] from data or zeros
const samplesBuf = [];
if (autoregressive_include_prefix) samplesBuf.push(prefix);

for (let i = 0; i < nIterations; i++) {
    const sample = await ddimLoop(x_init=randn([1,263,1,40]), prefix);  // [1,263,1,40]
    samplesBuf.push(sample);                                             // last 40 only
    prefix = sample.slice(-20, axis=time);                               // for next iteration
}
const full = concat(samplesBuf, axis=time).slice(0, 196);                // [1,263,1,196]
```

**Initial prefix source** (`generate.py` / `autoregressive_init`):
- `'data'` (default): pulled from a real HumanML3D motion. **For PoC, we just save a fixed `[1,263,1,20]` tensor as a fixture** so the browser doesn't need the dataset.
- `'isaac'`: from IsaacGym recordings. Out of scope.

## 8. Denormalization (for visualization only — not for parity)

After the 196-frame motion is produced, `generate.py:218-220`:

```python
sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
sample = recover_from_ric(sample, n_joints, hml_type)   # 263-dim → joint xyz
```

`inv_transform`: `x * std + mean` where `std, mean ∈ ℝ^263`, loaded from:
- `closd/diffusion_planner/dataset/t2m_mean.npy`
- `closd/diffusion_planner/dataset/t2m_std.npy`

For Phase 1 PoC parity, we compare the **normalized** 263-dim output (no inv_transform, no recover_from_ric) to the Python reference. Denormalization adds float drift and isn't on the model's correctness path.

For Phase 2 SMPL extraction (Day 14), we'll need both inv_transform (trivial: 263-dim mul+add) and `recover_from_ric` (non-trivial: cumulative root rotation/translation reconstruction — port from `closd/diffusion_planner/data_loaders/humanml/scripts/motion_process.py`).

## 9. Exact constants for `scheduler.ts`

```ts
export const CONFIG = {
    nDiffusionSteps: 10,
    noiseSchedule: 'cosine',           // see betasForAlphaBar() below
    nFrames: { context: 20, predict: 40, total: 196 },
    featureDim: 263,                    // HumanML3D
    latentDim: 512,
    clipDim: 512,
    guidanceScale: 7.5,                 // CFG default
    arIterations: 5,                    // (196 / 40) + 1
    eta: 0.0,                           // deterministic DDIM
    clipDenoised: false,                // generate.py:201
} as const;

// Cosine beta schedule (gaussian_diffusion.py:49-66, betas_for_alpha_bar with cosine alpha_bar)
function alphaBar(s: number) {
    const x = (s + 0.008) / 1.008;
    return Math.cos(x * Math.PI / 2) ** 2;
}
function cosineBetas(N: number, maxBeta = 0.999): number[] {
    const betas = [];
    for (let i = 0; i < N; i++) {
        const t1 = i / N, t2 = (i + 1) / N;
        betas.push(Math.min(1 - alphaBar(t2) / alphaBar(t1), maxBeta));
    }
    return betas;
}
```

## 10. Phase 2 reference (read on the relevant day, not yet)

| Phase 2 tool | Algorithmic delta vs Phase 1 | Reference file:line |
|---|---|---|
| Goal-conditioned (Day 8–9) | Add `target_cond` `[B, NJ+2, 3]` and `target_uncond` flag to `y`. Re-export model with `multi_target_cond=True` weights. | `mdm.py:200-202`, `closd/diffusion_planner/sample/generate.py` (`--sampling_mode goal`) |
| In-between editing (Day 10–11) | Inpainting: at each denoise step, replace `pred_xstart` in the known regions with `q_sample(x_known, t)` so they re-noise correctly; loop unchanged. Uses `q_sample` formula from `gaussian_diffusion.py:255-258`. The mask path is in `p_mean_variance` lines 359-370 (`inpainting_mask` / `inpainted_motion`). | `gaussian_diffusion.py:239-259, 359-370`, `closd/diffusion_planner/sample/edit.py` |
| Upper-body editing (Day 12) | Same as in-between but mask is over **feature dim** (263 partition by joint), not time dim. Joint partition: `closd/diffusion_planner/data_loaders/humanml/common/skeleton.py`. | `edit.py` `--edit_mode upper_body` |
| DDIM inversion (Day 13) | Reverse DDIM: increment t instead of decrement, and use `alphas_cumprod_next` instead of `_prev` to step forward (data → noise). | `closd/diffusion_planner/sample/ddim_invert.py`, `gaussian_diffusion.py` (look for `ddim_reverse_sample`) |
| SMPL extract (Day 14) | Pure post-process: 263-dim → SMPL pose params. **Risk:** if `extract_smpl.py` uses `pytorch3d`, port becomes a research task. | `closd/utils/extract_smpl.py` |

## 11. Decision-ready notes for export (Day 2)

- Export with `torch.onnx.export(..., opset_version=17, do_constant_folding=True, dynamic_axes=None)`. Fixed shapes: `B=1, T_pred=40`. ORT Web's WebGPU EP supports opset 17 well.
- If `torch.onnx.export` fails on a multi-head attention op, fall back to `torch.onnx.dynamo_export` (PyTorch 2.x). If that also fails — **Day 3 hard stop, escalate, see plan §"Decision points"**.
- FP16: export FP32, then convert with `onnxconverter_common.float16_converter.convert_float_to_float16`. Tolerance for parity goes from `1e-4` (FP32) to `1e-3` (FP16) per the success criteria.
- **Do not** export the CLIP model. transformers.js handles it.
- **Do** verify the exported ONNX with `onnxruntime` (Python) before checking it into the repo: same input → same output as the PyTorch model within `1e-5` (CPU FP32 path).

## 12. Gotchas (don't lose a day to these)

1. **Model strips the prefix internally** — output is `[1,263,1,40]`, not 60. No slicing in TS.
2. **`text_uncond` is implemented inside the ONNX trunk via the `text_uncond_mask` input** (a `[1]` float). Set to `0.0` for the cond pass, `1.0` for the uncond pass. The exported graph monkey-patches `mask_cond` so the runtime tensor drives the zeroing — `text_embed` is the real CLIP output in both passes.
3. **DDIM eta=0 means no noise term** — the line `mean_pred + nonzero_mask * sigma * noise` reduces to just `mean_pred` with `sigma=0`. Don't forget this when transcribing.
4. **`clip_denoised=False`** for DiP — do NOT clamp `pred_xstart` to [-1,1]. The HumanML3D feature space exceeds that range and clamping silently corrupts.
5. **Timestep tensor is int64** in PyTorch ONNX exports. ORT Web requires `BigInt64Array` for int64 inputs — use `new BigInt64Array([BigInt(t)])`, not `Int32Array`.
6. **Mask is bool**, but ONNX bool tensors take `Uint8Array` (`new Uint8Array([1,1,...])`) — don't pass `Float32Array` here, will silently broadcast wrong.
7. **CLIP tokenization context_length=22** in HumanML3D path; transformers.js may default to 77. Either truncate explicitly or verify the projection is invariant (the `[EOS]` token determines the pooled output, so truncation past the prompt's actual length is usually a no-op — verify on Day 5).
8. **`autoregressive_include_prefix=False`** for the standard demo — first 20 frames in the output are from the data prefix; user should NOT see them.
9. **CFG `scale` shape is `[B,1,1,1]`** in PyTorch broadcast. For B=1 it's just a scalar in TS; keep it scalar.
10. **No learned null token** — `mask_cond` zeros the text embedding on the **output** side of the `embed_text` Linear (per `mdm.py:233`, default `emb_before_mask=False`). The export script monkey-patches `mask_cond` to apply the runtime `text_uncond_mask` tensor at that exact spot, preserving the original semantics regardless of `emb_before_mask`.

## 13. References (quick links)

- Model: `closd/diffusion_planner/model/mdm.py`
- Sampler: `closd/diffusion_planner/utils/sampler_util.py`
- Diffusion math: `closd/diffusion_planner/diffusion/gaussian_diffusion.py`
- Schedule respacing: `closd/diffusion_planner/diffusion/respace.py`
- Model construction: `closd/diffusion_planner/utils/model_util.py`
- Generate entrypoint: `closd/diffusion_planner/sample/generate.py`
- Edit entrypoint (Phase 2): `closd/diffusion_planner/sample/edit.py`
- DDIM invert (Phase 2): `closd/diffusion_planner/sample/ddim_invert.py`
- Mean/std: `closd/diffusion_planner/dataset/{t2m_mean,t2m_std}.npy`
- Joint partition (Phase 2 upper-body): `closd/diffusion_planner/data_loaders/humanml/common/skeleton.py`
- Recover-from-RIC: `closd/diffusion_planner/data_loaders/humanml/scripts/motion_process.py`
