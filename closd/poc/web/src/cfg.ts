// Classifier-free guidance wrapper around the ONNX trunk.
//
// Mirrors closd/diffusion_planner/utils/sampler_util.py:21-31:
//   out_cond   = model(x, t, y)
//   out_uncond = model(x, t, y_uncond)   // y_uncond['text_uncond'] = True
//   return out_uncond + scale * (out_cond - out_uncond)
//
// In our ONNX trunk the cond/uncond branch is selected by `text_uncond_mask`:
//   0.0 → cond pass (text passes through embed_text)
//   1.0 → uncond pass (text embedding zeroed after embed_text Linear, matching mask_cond)
// See closd/poc/SAMPLER_NOTES.md §4 and the export script for why we wire it this way.

import type { TrunkSession } from "./ort_setup.js";
import { axpby, T4 } from "./tensor.js";

export interface CfgInputs {
  x: T4;                  // [1, 263, 1, 40]
  timestep: number;       // 0..9
  textEmbed: T4;          // [T_text, 1, 768] DistilBERT last_hidden_state, seq-first
  textMask: T4;           // [1, T_text] bool, True = padding
  mask: T4;               // [1, 1, 1, 40] validity mask, treated as bool by ORT
  prefix: T4;             // [1, 263, 1, 20] AR rolling prefix
}

/**
 * Run two trunk passes (cond + uncond) and blend per CFG.
 * `scale` is the guidance_param (default 7.5 for DiP).
 *
 * Both passes share x, timestep, mask, prefix, and textEmbed — only
 * `text_uncond_mask` differs. We pay the cost of two forward passes per
 * denoise step; with 10 steps × 5 AR iters this is 100 trunk calls per 196-frame motion.
 *
 * Optional `tap` callback receives the pre-CFG cond/uncond outputs for
 * the parity harness on Day 6.
 */
export async function runCfgStep(
  session: TrunkSession,
  inputs: CfgInputs,
  scale: number,
  tap?: (cond: T4, uncond: T4) => void,
): Promise<T4> {
  // ORT-Web sessions can't be invoked concurrently — `session.run()` is
  // serialized internally and a second concurrent call throws
  // "Session already started". So even though cond + uncond are
  // independent we have to await them sequentially. Total wall time
  // is ~2x but correctness > parallelism.

  // Cond pass: real text_embed, text_uncond_mask=0 (no zeroing).
  const cond = await session.run({
    x: inputs.x,
    timestep: inputs.timestep,
    textEmbed: inputs.textEmbed,
    textMask: inputs.textMask,
    mask: inputs.mask,
    prefix: inputs.prefix,
    textUncondMask: 0,
  });

  // Uncond pass: zero text_embed in TS instead of relying on the trunk's
  // text_uncond_mask path (which depends on a monkey-patched mask_cond
  // captured at export time and may not have been traced reliably). Per
  // mdm.py:233 with emb_before_mask=False (verified True in args.json):
  //   text_emb = embed_text(mask_cond(enc_text, force_mask=True))
  //            = embed_text(zeros)
  //            = bias of the embed_text Linear
  // Feeding all-zero text_embed reproduces this exactly because the Linear
  // is the very next op after mask_cond. text_mask stays unchanged across
  // cond/uncond per mdm.py:228-229. We also set text_uncond_mask=1 belt-and-
  // suspenders so if the patch IS in the graph it's idempotent (zeros * 0 = 0).
  const zerosEmbed: T4 = {
    data: new Float32Array(inputs.textEmbed.data.length),
    shape: inputs.textEmbed.shape,
  };
  const uncond = await session.run({
    x: inputs.x,
    timestep: inputs.timestep,
    textEmbed: zerosEmbed,
    textMask: inputs.textMask,
    mask: inputs.mask,
    prefix: inputs.prefix,
    textUncondMask: 1,
  });

  if (tap) tap(cond, uncond);

  // out = uncond + scale * (cond - uncond) = scale*cond + (1-scale)*uncond
  return axpby(scale, cond, 1 - scale, uncond);
}
