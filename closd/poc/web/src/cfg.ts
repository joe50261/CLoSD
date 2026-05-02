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

import type { TrunkSession, TrunkInputs } from "./ort_setup.js";
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
  const baseInputs: Omit<TrunkInputs, "textUncondMask"> = {
    x: inputs.x,
    timestep: inputs.timestep,
    textEmbed: inputs.textEmbed,
    textMask: inputs.textMask,
    mask: inputs.mask,
    prefix: inputs.prefix,
  };

  const condPromise = session.run({ ...baseInputs, textUncondMask: 0 });
  const uncondPromise = session.run({ ...baseInputs, textUncondMask: 1 });
  const [cond, uncond] = await Promise.all([condPromise, uncondPromise]);

  if (tap) tap(cond, uncond);

  // out = uncond + scale * (cond - uncond) = scale*cond + (1-scale)*uncond
  return axpby(scale, cond, 1 - scale, uncond);
}
