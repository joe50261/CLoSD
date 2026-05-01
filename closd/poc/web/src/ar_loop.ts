// Autoregressive sampling loop, mirroring closd/diffusion_planner/utils/sampler_util.py:38-61.
//
// Pseudocode (also in SAMPLER_NOTES.md §7):
//   nIterations = floor(196 / pred_len) + 1     // = 5 for pred_len=40
//   for i in range(nIterations):
//       sample = ddimSampleLoop(model, prefix=cur_prefix)   // [1,263,1,40]
//       buf.append(sample)
//       cur_prefix = sample[..., -context_len:]              // last 20 frames
//   return concat(buf)[:196]

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
  randnT4,
  SeededRng,
  sliceLastAxis,
  T4,
  zeros,
} from "./tensor.js";

export interface ArInputs {
  textEmbed: T4;        // [1, 1, 512] cached CLIP encoding
  initialPrefix: T4;    // [1, 263, 1, context_len] seed prefix
}

export interface ArOptions {
  config: DiPConfig;
  /** Seed for the noise RNG, so two runs match. */
  seed: number;
  /** Optional progress callback: (iter, step, x_t) — for the parity harness on Day 6. */
  onStep?: (iter: number, step: number, t: number, xT: T4) => void;
  /** Optional fixture taps for cond/uncond pre-CFG outputs. */
  onTap?: (iter: number, step: number, cond: T4, uncond: T4) => void;
  /** If true, the seed prefix is included as the first 20 frames of the output.
   *  Defaults to false to match generate.py with autoregressive_include_prefix=False. */
  includePrefix?: boolean;
}

/**
 * Run a single denoise loop (10 DDIM steps with CFG) for one AR iteration.
 * Conditioned on `prefix`. Returns the [1, 263, 1, pred_len] predicted region.
 */
async function denoiseLoop(
  session: TrunkSession,
  textEmbed: T4,
  prefix: T4,
  config: DiPConfig,
  rng: SeededRng,
  onStep: ArOptions["onStep"],
  onTap: ArOptions["onTap"],
  iterIdx: number,
): Promise<T4> {
  const schedule = buildSchedule(config.nDiffusionSteps);
  const shape = [1, config.featureDim, 1, config.nFramesPredict] as const;

  // Initial noise.
  let x: T4 = randnT4(shape, rng);
  // Validity mask: all 40 predict frames are valid (we don't pad in the PoC).
  const mask: T4 = (() => {
    const m = zeros(shape);
    m.data.fill(1);
    return m;
  })();

  let stepIdx = 0;
  for (const t of ddimTimesteps(schedule)) {
    if (onStep) onStep(iterIdx, stepIdx, t, x);

    const predXstart = await runCfgStep(
      session,
      { x, timestep: t, textEmbed, mask, prefix },
      config.guidanceScale,
      onTap ? (cond, uncond) => onTap(iterIdx, stepIdx, cond, uncond) : undefined,
    );
    x = ddimStepEta0(x, predXstart, t, schedule);
    stepIdx++;
  }

  return x;
}

/**
 * Full AR sample: 5 iterations of 40-frame predictions, with the last 20 frames
 * of each iteration used as the prefix for the next. Returns the trimmed
 * 196-frame motion (the same length generate.py would produce).
 */
export async function autoregressiveSample(
  session: TrunkSession,
  inputs: ArInputs,
  opts: ArOptions,
): Promise<T4> {
  const cfg = opts.config;
  const nIterations = Math.floor(cfg.nFramesTotal / cfg.nFramesPredict) + 1;

  const rng = new SeededRng(opts.seed);
  let prefix = copy(inputs.initialPrefix);

  const buf: T4[] = [];
  if (opts.includePrefix) buf.push(copy(prefix));

  for (let i = 0; i < nIterations; i++) {
    const sample = await denoiseLoop(
      session,
      inputs.textEmbed,
      prefix,
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

  // Trim to total frames.
  const T = full.shape[3]!;
  if (T > cfg.nFramesTotal) {
    full = sliceLastAxis(full, 0, cfg.nFramesTotal);
  }
  return full;
}
