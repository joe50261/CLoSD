// In-between editing and upper-body editing for the DiP no-target trunk.
//
// Approach (matches closd/diffusion_planner/sample/edit.py + the inpainting
// branch in gaussian_diffusion.py:359-370): at each DDIM step, after the
// model produces pred_xstart, replace the values at indices where the
// inpainting mask is True with the corresponding values from the
// reference motion. The DDIM step then uses the mixed pred_xstart, so x_{t-1}
// at preserved regions stays close to the reference, and at the unmasked
// regions the model fills in.
//
// We adapt to AR mode: the user provides a 196-frame reference + 196-frame
// mask, and within each 40-frame AR iteration we slice the corresponding
// 40 frames of both for the mix. This keeps the same trunk + same scheduler
// as Phase 1; only the predict-region post-processing changes.

import { runCfgStep } from "./cfg.js";
import type { TrunkSession } from "./ort_setup.js";
import {
  buildSchedule,
  ddimStepEta0,
  ddimTimesteps,
  type DiPConfig,
} from "./scheduler.js";
import {
  concatLastAxis,
  copy,
  inpaintMix,
  randnT4,
  SeededRng,
  sliceLastAxis,
  T4,
  zeros,
} from "./tensor.js";

export interface InpaintInputs {
  textEmbed: T4;                  // [1, 1, 512]
  initialPrefix: T4;              // [1, 263, 1, context_len]
  /**
   * Reference motion. Two supported shapes:
   *   - Time-axis inpainting (in-between): [1, 263, 1, 196]
   *     mask should also be [1, 263, 1, 196]; mask=true frames preserved.
   *   - Feature-axis inpainting (upper-body): [1, 263, 1, 196]
   *     mask should be [1, 263, 1, 1]; mask=true features preserved
   *     (broadcast over batch + time).
   */
  referenceMotion: T4;
  inpaintingMask: T4;
}

export interface InpaintOptions {
  config: DiPConfig;
  seed: number;
  onStep?: (iter: number, step: number, t: number, xT: T4) => void;
  onTap?: (iter: number, step: number, cond: T4, uncond: T4) => void;
  includePrefix?: boolean;
}

/**
 * Slice a 196-frame reference along the time axis to match a single AR
 * iteration's 40-frame predict region. For feature-mask inpainting the mask
 * doesn't depend on time and we return it as-is.
 */
function sliceForIter(
  reference: T4,
  mask: T4,
  iterIdx: number,
  predLen: number,
  totalLen: number,
): { ref: T4; m: T4 } {
  const start = iterIdx * predLen;
  const end = Math.min(start + predLen, totalLen);
  // If the iter goes past the reference length (last iter when total/predLen
  // is non-integer), clamp to the last predLen frames.
  const safeStart =
    end - start < predLen ? Math.max(0, totalLen - predLen) : start;
  const safeEnd = safeStart + predLen;
  const ref = sliceLastAxis(reference, safeStart, safeEnd);

  if (mask.shape[3] === 1) {
    return { ref, m: mask }; // feature mask, no time slicing
  }
  const m = sliceLastAxis(mask, safeStart, safeEnd);
  return { ref, m };
}

async function denoiseLoopWithInpaint(
  session: TrunkSession,
  textEmbed: T4,
  prefix: T4,
  iterReference: T4,
  iterMask: T4,
  config: DiPConfig,
  rng: SeededRng,
  onStep: InpaintOptions["onStep"],
  onTap: InpaintOptions["onTap"],
  iterIdx: number,
): Promise<T4> {
  const schedule = buildSchedule(config.nDiffusionSteps);
  const shape = [1, config.featureDim, 1, config.nFramesPredict] as const;

  let x: T4 = randnT4(shape, rng);
  const mask: T4 = (() => {
    const m = zeros(shape);
    m.data.fill(1);
    return m;
  })();

  let stepIdx = 0;
  for (const t of ddimTimesteps(schedule)) {
    if (onStep) onStep(iterIdx, stepIdx, t, x);

    let predXstart = await runCfgStep(
      session,
      { x, timestep: t, textEmbed, mask, prefix },
      config.guidanceScale,
      onTap ? (cond, uncond) => onTap(iterIdx, stepIdx, cond, uncond) : undefined,
    );

    // Inpainting post-process: replace pred_xstart at known regions with
    // the reference. Mirrors gaussian_diffusion.py:363.
    predXstart = inpaintMix(predXstart, iterReference, iterMask);

    x = ddimStepEta0(x, predXstart, t, schedule);
    stepIdx++;
  }
  return x;
}

/**
 * AR sample with inpainting at every step. Reference + mask are sliced per
 * iteration and applied inside the denoise loop.
 */
export async function autoregressiveSampleWithInpainting(
  session: TrunkSession,
  inputs: InpaintInputs,
  opts: InpaintOptions,
): Promise<T4> {
  const cfg = opts.config;
  const nIterations = Math.floor(cfg.nFramesTotal / cfg.nFramesPredict) + 1;
  const rng = new SeededRng(opts.seed);
  let prefix = copy(inputs.initialPrefix);

  const buf: T4[] = [];
  if (opts.includePrefix) buf.push(copy(prefix));

  for (let i = 0; i < nIterations; i++) {
    const { ref, m } = sliceForIter(
      inputs.referenceMotion,
      inputs.inpaintingMask,
      i,
      cfg.nFramesPredict,
      cfg.nFramesTotal,
    );

    const sample = await denoiseLoopWithInpaint(
      session,
      inputs.textEmbed,
      prefix,
      ref,
      m,
      cfg,
      rng,
      opts.onStep,
      opts.onTap,
      i,
    );
    buf.push(sample);
    prefix = sliceLastAxis(sample, -cfg.nFramesContext);
  }

  let full = buf[0]!;
  for (let i = 1; i < buf.length; i++) {
    full = concatLastAxis(full, buf[i]!);
  }
  if (full.shape[3]! > cfg.nFramesTotal) {
    full = sliceLastAxis(full, 0, cfg.nFramesTotal);
  }
  return full;
}

/**
 * Helper: build an in-between mask (preserve frames [0..start) ∪ [end..total),
 * inpaint frames [start..end)). Returns a [1, 263, 1, total] tensor with 1.0
 * where frames are preserved and 0.0 where they're inpainted.
 */
export function buildInBetweenMask(
  start: number,
  end: number,
  totalFrames: number,
  featureDim = 263,
): T4 {
  const shape: readonly number[] = [1, featureDim, 1, totalFrames];
  const data = new Float32Array(featureDim * totalFrames);
  for (let f = 0; f < featureDim; f++) {
    for (let t = 0; t < totalFrames; t++) {
      const preserved = t < start || t >= end ? 1 : 0;
      data[f * totalFrames + t] = preserved;
    }
  }
  return { data, shape };
}

/**
 * Helper: build an upper-body mask (preserve lower-body features, inpaint upper).
 * The actual feature partition is in
 *   closd/diffusion_planner/data_loaders/humanml_utils.py (HML_LOWER_BODY_MASK).
 *
 * For the PoC we accept the partition as a boolean array — the caller must
 * supply the correct one (typically loaded from a fixture or imported as a
 * 263-element constant).
 */
export function buildFeatureMask(
  partition: ReadonlyArray<boolean>,
  featureDim = 263,
): T4 {
  if (partition.length !== featureDim) {
    throw new Error(`partition length ${partition.length} != ${featureDim}`);
  }
  const data = new Float32Array(featureDim);
  for (let f = 0; f < featureDim; f++) data[f] = partition[f] ? 1 : 0;
  return { data, shape: [1, featureDim, 1, 1] };
}
