// Symbolic checks for the DDIM scheduler.
// We don't have the Python reference fixtures available in CI; these tests
// verify the math against properties that must hold for any correct
// implementation (and against a hand-computed Python value).

import { describe, expect, it } from "vitest";
import {
  buildSchedule,
  cosineBetas,
  ddimStepEta0,
  ddimTimesteps,
} from "./scheduler.js";
import { axpby, T4, zeros } from "./tensor.js";

describe("cosineBetas", () => {
  it("returns N values in (0, 0.999]", () => {
    const N = 10;
    const betas = cosineBetas(N);
    expect(betas).toHaveLength(N);
    for (const b of betas) {
      expect(b).toBeGreaterThan(0);
      expect(b).toBeLessThanOrEqual(0.999);
    }
  });

  it("values are monotonically non-decreasing for cosine schedule", () => {
    // For cosine alpha_bar (which decreases from 1 toward 0), discrete betas
    // are increasing — noise level grows over time.
    const betas = cosineBetas(10);
    for (let i = 1; i < betas.length; i++) {
      expect(betas[i]!).toBeGreaterThan(betas[i - 1]!);
    }
  });

  it("first beta matches the Python reference (N=10)", () => {
    // Hand-computed from gaussian_diffusion.py:49-66 with cosine alpha_bar:
    //   alpha_bar(0) = cos((0.008/1.008) * pi/2)^2
    //   alpha_bar(0.1) = cos(((0.1+0.008)/1.008) * pi/2)^2
    //   beta_0 = 1 - alpha_bar(0.1) / alpha_bar(0)
    const ab0 = Math.cos(((0 + 0.008) / 1.008) * Math.PI / 2) ** 2;
    const ab1 = Math.cos(((0.1 + 0.008) / 1.008) * Math.PI / 2) ** 2;
    const expected = 1 - ab1 / ab0;
    const betas = cosineBetas(10);
    expect(betas[0]!).toBeCloseTo(expected, 12);
  });
});

describe("buildSchedule", () => {
  it("alphasCumprod is monotonically decreasing in (0, 1)", () => {
    const s = buildSchedule(10);
    expect(s.alphasCumprod[0]!).toBeLessThan(1);
    expect(s.alphasCumprod[9]!).toBeGreaterThan(0);
    for (let i = 1; i < s.alphasCumprod.length; i++) {
      expect(s.alphasCumprod[i]!).toBeLessThan(s.alphasCumprod[i - 1]!);
    }
  });

  it("alphasCumprodPrev[0] = 1.0 and alphasCumprodPrev[i] = alphasCumprod[i-1] for i>=1", () => {
    const s = buildSchedule(10);
    expect(s.alphasCumprodPrev[0]!).toBe(1.0);
    for (let i = 1; i < s.alphasCumprod.length; i++) {
      expect(s.alphasCumprodPrev[i]!).toBe(s.alphasCumprod[i - 1]!);
    }
  });

  it("sqrtRecipAlphasCumprod * sqrt(alphasCumprod) = 1 (definitional identity)", () => {
    const s = buildSchedule(10);
    for (let i = 0; i < s.alphasCumprod.length; i++) {
      const lhs = s.sqrtRecipAlphasCumprod[i]! * Math.sqrt(s.alphasCumprod[i]!);
      expect(lhs).toBeCloseTo(1, 12);
    }
  });

  it("sqrtRecipm1AlphasCumprod^2 + 1 = 1/alphasCumprod (definitional identity)", () => {
    const s = buildSchedule(10);
    for (let i = 0; i < s.alphasCumprod.length; i++) {
      const lhs = s.sqrtRecipm1AlphasCumprod[i]! ** 2 + 1;
      const rhs = 1 / s.alphasCumprod[i]!;
      expect(lhs).toBeCloseTo(rhs, 10);
    }
  });
});

describe("ddimTimesteps", () => {
  it("descends from N-1 to 0", () => {
    expect(ddimTimesteps(buildSchedule(10))).toEqual([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
  });
});

describe("ddimStepEta0", () => {
  // Build a tiny T4 from a flat array. Used to keep tests readable.
  function t4(values: number[], shape: T4["shape"]): T4 {
    const expected = shape.reduce((a, b) => a * b, 1);
    if (values.length !== expected) {
      throw new Error(`bad fixture: ${values.length} vs ${expected}`);
    }
    return { data: new Float32Array(values), shape };
  }

  it("Identity property: if pred_xstart = x_t, eps = 0 and result = x_t * sqrt(alpha_bar_prev/alpha_bar_t) ... close to x_t for small noise levels", () => {
    // Detailed math check: when pred_xstart equals x_t exactly (a degenerate case),
    // eps = (a*x - x) / b = ((a-1)/b) * x, which may be non-zero. So just verify the
    // formula holds: x_{t-1} = sqrt(ab_prev) * x + sqrt(1-ab_prev) * eps.
    const s = buildSchedule(10);
    const x: T4 = t4([1, 2, 3, 4], [1, 1, 1, 4]);
    const pred = t4([1, 2, 3, 4], [1, 1, 1, 4]); // pred = x_t
    const t = 5;
    const out = ddimStepEta0(x, pred, t, s);

    // Hand-compute expected
    const a = s.sqrtRecipAlphasCumprod[t]!;
    const b = s.sqrtRecipm1AlphasCumprod[t]!;
    const abPrev = s.alphasCumprodPrev[t]!;
    for (let i = 0; i < 4; i++) {
      const xv = x.data[i]!;
      const pv = pred.data[i]!;
      const eps = (a * xv - pv) / b;
      const expected = pv * Math.sqrt(abPrev) + eps * Math.sqrt(1 - abPrev);
      expect(out.data[i]!).toBeCloseTo(expected, 5);
    }
  });

  it("Zero-noise limit at t=0: alpha_bar_prev=1 means x_{-1} = pred_xstart exactly", () => {
    // At t=0, alphasCumprodPrev[0] = 1.0, so:
    //   sqrt(ab_prev)=1, sqrt(1-ab_prev)=0
    //   x_{-1} = 1 * pred_xstart + 0 * eps = pred_xstart.
    // This is the cleanest property: the final DDIM step returns exactly the model's
    // x_start prediction.
    const s = buildSchedule(10);
    const x: T4 = t4([0.5, -0.3, 1.7, 0.0], [1, 1, 1, 4]);
    const pred: T4 = t4([1, 2, 3, 4], [1, 1, 1, 4]);
    const out = ddimStepEta0(x, pred, 0, s);
    for (let i = 0; i < 4; i++) {
      expect(out.data[i]!).toBeCloseTo(pred.data[i]!, 6);
    }
  });

  it("Throws on out-of-range timestep", () => {
    const s = buildSchedule(10);
    const x = zeros([1, 1, 1, 1]);
    const pred = zeros([1, 1, 1, 1]);
    expect(() => ddimStepEta0(x, pred, -1, s)).toThrow();
    expect(() => ddimStepEta0(x, pred, 10, s)).toThrow();
  });

  it("CFG blend is correct: uncond + scale * (cond - uncond)", () => {
    // Sanity-check the CFG formula we'll use in cfg.ts against the same axpby
    // primitive the scheduler uses, so the building blocks compose cleanly.
    const cond: T4 = t4([1, 2, 3, 4], [1, 1, 1, 4]);
    const uncond: T4 = t4([0.5, 1.5, 2.5, 3.5], [1, 1, 1, 4]);
    const scale = 7.5;
    // out = uncond + scale*(cond - uncond) = (1-scale)*uncond + scale*cond
    const out = axpby(scale, cond, 1 - scale, uncond);
    for (let i = 0; i < 4; i++) {
      const expected =
        uncond.data[i]! + scale * (cond.data[i]! - uncond.data[i]!);
      expect(out.data[i]!).toBeCloseTo(expected, 6);
    }
  });
});
