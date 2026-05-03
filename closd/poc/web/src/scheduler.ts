// DDIM scheduler for DiP. Mirrors closd/diffusion_planner/diffusion/gaussian_diffusion.py.
//
// Why this lives in TS, not ONNX: the diffusion loop is trivially small (10 steps,
// scalar coefficients, two array ops per step). Putting it in TS makes parity
// debugging straightforward — you can log intermediate x_t at every step.
//
// References (every line cited can be checked against the Python source):
//  - Cosine beta schedule: gaussian_diffusion.py:49-66 (betas_for_alpha_bar)
//  - alphas_cumprod et al: gaussian_diffusion.py:173-184
//  - DDIM step (eta=0):    gaussian_diffusion.py:806-855 (ddim_sample)
//  - _predict_eps_from_xstart: gaussian_diffusion.py:463-467
//  - clip_denoised=False:  closd/diffusion_planner/sample/generate.py:193

import { axpby, epsFromXstart, T4 } from "./tensor.js";

export interface DiPConfig {
  readonly nDiffusionSteps: number;     // 10 for the DiP_*_10steps_* checkpoints
  readonly noiseSchedule: "cosine";     // only schedule we ship
  readonly nFramesContext: number;      // 20
  readonly nFramesPredict: number;      // 40
  readonly nFramesTotal: number;        // 196
  readonly featureDim: number;          // 263 (HumanML3D)
  readonly clipDim: number;             // 512
  readonly guidanceScale: number;       // 7.5
  readonly eta: number;                 // 0.0 → deterministic DDIM
  readonly clipDenoised: boolean;       // false for DiP
}

export const DEFAULT_CONFIG: DiPConfig = {
  nDiffusionSteps: 10,
  noiseSchedule: "cosine",
  nFramesContext: 20,
  nFramesPredict: 40,
  nFramesTotal: 196,
  featureDim: 263,
  clipDim: 512,
  guidanceScale: 7.5,
  eta: 0.0,
  clipDenoised: false,
};

// gaussian_diffusion.py:54-58 — alpha_bar(t) for cosine schedule.
// Note: t here is the continuous "fraction of total steps" in [0, 1].
function cosineAlphaBar(s: number): number {
  const x = (s + 0.008) / 1.008;
  return Math.cos((x * Math.PI) / 2) ** 2;
}

// gaussian_diffusion.py:49-66 — discrete betas from alpha_bar.
export function cosineBetas(N: number, maxBeta = 0.999): number[] {
  const betas: number[] = [];
  for (let i = 0; i < N; i++) {
    const t1 = i / N;
    const t2 = (i + 1) / N;
    const beta = Math.min(1 - cosineAlphaBar(t2) / cosineAlphaBar(t1), maxBeta);
    betas.push(beta);
  }
  return betas;
}

// Precomputed schedule. All arrays have length N (= nDiffusionSteps).
export interface Schedule {
  readonly betas: number[];
  readonly alphas: number[];
  readonly alphasCumprod: number[];        // ᾱ_t
  readonly alphasCumprodPrev: number[];    // ᾱ_{t-1}, with ᾱ_{-1} := 1
  readonly sqrtRecipAlphasCumprod: number[];     // 1/√ᾱ_t
  readonly sqrtRecipm1AlphasCumprod: number[];   // √(1/ᾱ_t - 1)
}

export function buildSchedule(N: number): Schedule {
  const betas = cosineBetas(N);
  const alphas = betas.map((b) => 1 - b);
  // cumulative product
  const alphasCumprod: number[] = [];
  let acc = 1;
  for (const a of alphas) {
    acc *= a;
    alphasCumprod.push(acc);
  }
  const alphasCumprodPrev = [1.0, ...alphasCumprod.slice(0, -1)];
  const sqrtRecipAlphasCumprod = alphasCumprod.map((a) => Math.sqrt(1 / a));
  const sqrtRecipm1AlphasCumprod = alphasCumprod.map((a) =>
    Math.sqrt(1 / a - 1),
  );
  return {
    betas,
    alphas,
    alphasCumprod,
    alphasCumprodPrev,
    sqrtRecipAlphasCumprod,
    sqrtRecipm1AlphasCumprod,
  };
}

/**
 * One DDIM step (eta=0, deterministic).
 *
 * Mirrors gaussian_diffusion.py:806-855. With eta=0 the sigma term vanishes
 * (sigma=0 → no noise injected), so we collapse the math to a single closed form.
 *
 * @param xT          x_t at current timestep (input)
 * @param predXstart  model's x_start prediction at current timestep
 * @param t           current timestep index (0..N-1)
 * @param schedule    precomputed schedule
 * @returns           x_{t-1}
 */
export function ddimStepEta0(
  xT: T4,
  predXstart: T4,
  t: number,
  schedule: Schedule,
): T4 {
  const N = schedule.alphasCumprod.length;
  if (t < 0 || t >= N) {
    throw new Error(`ddimStepEta0: t=${t} out of range [0, ${N})`);
  }

  // gaussian_diffusion.py:837 — eps = (sqrt_recip_alpha_bar * x_t - pred_xstart) / sqrt_recipm1_alpha_bar
  const a = schedule.sqrtRecipAlphasCumprod[t]!;
  const b = schedule.sqrtRecipm1AlphasCumprod[t]!;
  const eps = epsFromXstart(a, xT, predXstart, b);

  // gaussian_diffusion.py:840 — alpha_bar_prev[t]
  const alphaBarPrev = schedule.alphasCumprodPrev[t]!;

  // gaussian_diffusion.py:848-851 with eta=0 (sigma=0):
  //   mean_pred = pred_xstart * sqrt(alpha_bar_prev) + eps * sqrt(1 - alpha_bar_prev)
  // No noise added when eta=0 (line 855: sample = mean_pred + 0 * sigma * noise)
  const sqrtAbPrev = Math.sqrt(alphaBarPrev);
  const sqrtOmAbPrev = Math.sqrt(1 - alphaBarPrev);
  return axpby(sqrtAbPrev, predXstart, sqrtOmAbPrev, eps);
}

/**
 * Descending-t loop indices for the DDIM sampler.
 * Returns [N-1, N-2, ..., 1, 0]. Mirrors the iteration in p_sample_loop_progressive.
 */
export function ddimTimesteps(schedule: Schedule): number[] {
  const N = schedule.alphasCumprod.length;
  const out: number[] = [];
  for (let t = N - 1; t >= 0; t--) out.push(t);
  return out;
}
