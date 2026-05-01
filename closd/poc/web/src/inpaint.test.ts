// Tests for inpaint.ts using the same synthetic-trunk pattern as ar_loop.test.ts.

import { describe, expect, it } from "vitest";
import {
  autoregressiveSampleWithInpainting,
  buildFeatureMask,
  buildInBetweenMask,
} from "./inpaint.js";
import type { TrunkInputs, TrunkSession } from "./ort_setup.js";
import { DEFAULT_CONFIG } from "./scheduler.js";
import { copy, inpaintMix, T4, zeros } from "./tensor.js";

describe("inpaintMix", () => {
  function t4(values: number[], shape: readonly number[]): T4 {
    return { data: new Float32Array(values), shape };
  }

  it("element-wise mix when mask matches pred shape", () => {
    const pred = t4([1, 2, 3, 4], [1, 1, 1, 4]);
    const ref = t4([10, 20, 30, 40], [1, 1, 1, 4]);
    const mask = t4([1, 0, 1, 0], [1, 1, 1, 4]);
    const out = inpaintMix(pred, ref, mask);
    expect(Array.from(out.data)).toEqual([10, 2, 30, 4]);
  });

  it("threshold-based mix: any positive mask = preserve, zero or negative = inpaint", () => {
    const pred = t4([1, 2, 3, 4], [1, 1, 1, 4]);
    const ref = t4([10, 20, 30, 40], [1, 1, 1, 4]);
    const mask = t4([0.5, 0, 1.0, -1.0], [1, 1, 1, 4]);
    const out = inpaintMix(pred, ref, mask);
    expect(Array.from(out.data)).toEqual([10, 2, 30, 4]);
  });

  it("broadcast over feature dim: mask shape [1,F,1,1] applies same per (b,t)", () => {
    // 1 batch, 2 features, 1 joint, 3 time steps
    const pred = t4([1, 2, 3, 4, 5, 6], [1, 2, 1, 3]);
    const ref = t4([10, 20, 30, 40, 50, 60], [1, 2, 1, 3]);
    // Mask preserves feature 0, inpaints feature 1
    const mask = t4([1, 0], [1, 2, 1, 1]);
    const out = inpaintMix(pred, ref, mask);
    // Feature 0 (first 3): preserve 10,20,30
    // Feature 1 (last 3): inpaint 4,5,6
    expect(Array.from(out.data)).toEqual([10, 20, 30, 4, 5, 6]);
  });

  it("throws on incompatible mask shape", () => {
    const pred = zeros([1, 2, 1, 3]);
    const ref = zeros([1, 2, 1, 3]);
    const badMask = zeros([1, 1, 2, 1]); // doesn't match either case
    expect(() => inpaintMix(pred, ref, badMask)).toThrow();
  });
});

describe("buildInBetweenMask", () => {
  it("preserves [0..start) and [end..total), inpaints [start..end)", () => {
    const m = buildInBetweenMask(2, 5, 7, 1); // featureDim=1 to keep test small
    expect(m.shape).toEqual([1, 1, 1, 7]);
    expect(Array.from(m.data)).toEqual([1, 1, 0, 0, 0, 1, 1]);
  });
});

describe("buildFeatureMask", () => {
  it("returns [1, F, 1, 1] tensor matching the partition", () => {
    const m = buildFeatureMask([true, false, true], 3);
    expect(m.shape).toEqual([1, 3, 1, 1]);
    expect(Array.from(m.data)).toEqual([1, 0, 1]);
  });

  it("rejects length mismatch", () => {
    expect(() => buildFeatureMask([true, false], 3)).toThrow();
  });
});

describe("autoregressiveSampleWithInpainting", () => {
  const cfg = DEFAULT_CONFIG;

  function makeIdentityTrunk(): TrunkSession {
    // Returns input.x as-is. With inpainting, masked frames will end up
    // being the reference value at convergence (t=0), and unmasked frames
    // will track the noise.
    return {
      executionProvider: "mock",
      async run(inputs: TrunkInputs): Promise<T4> {
        return copy(inputs.x);
      },
      release() {},
    };
  }

  it("produces a [1, 263, 1, 196] tensor", async () => {
    const out = await autoregressiveSampleWithInpainting(
      makeIdentityTrunk(),
      {
        textEmbed: zeros([1, 1, 512]),
        initialPrefix: zeros([1, 263, 1, 20]),
        referenceMotion: zeros([1, 263, 1, 196]),
        inpaintingMask: zeros([1, 263, 1, 196]),
      },
      { config: cfg, seed: 42 },
    );
    expect(out.shape).toEqual([1, 263, 1, 196]);
  });

  it("preserves reference frames where mask=true (in-between)", async () => {
    // Build a reference of all 7s and mask preserve frames 0..10 + 30..196.
    // Inpaint window is [10, 30). Within that window, the model output
    // varies; outside it should be exactly 7.
    const reference = (() => {
      const t = zeros([1, 263, 1, 196]);
      t.data.fill(7);
      return t;
    })();
    const mask = buildInBetweenMask(10, 30, 196);

    // Use a trunk that returns x_t scaled by 0.5 — output won't be 7 unless
    // the inpainting is doing its job.
    const trunk: TrunkSession = {
      executionProvider: "mock",
      async run(inputs: TrunkInputs) {
        const out = copy(inputs.x);
        for (let i = 0; i < out.data.length; i++) out.data[i] = out.data[i]! * 0.5;
        return out;
      },
      release() {},
    };

    const out = await autoregressiveSampleWithInpainting(
      trunk,
      {
        textEmbed: zeros([1, 1, 512]),
        initialPrefix: zeros([1, 263, 1, 20]),
        referenceMotion: reference,
        inpaintingMask: mask,
      },
      { config: cfg, seed: 42 },
    );

    // Frames 0..10 and 30..196: should equal 7 (preserved).
    // Frames 10..30: should NOT all equal 7 (the inpainted region — model output
    // converges from noise, can't end up exactly at 7).
    // Index into the [1, 263, 1, 196] tensor: feature 0, time t = data[t].
    // (Stride-major: feature*196 + time.)
    for (let t = 0; t < 10; t++) {
      expect(out.data[t]!).toBeCloseTo(7, 1);
    }
    for (let t = 30; t < 196; t++) {
      expect(out.data[t]!).toBeCloseTo(7, 1);
    }
    // Spot-check that the inpainted region differs from 7 for at least one frame.
    let allSeven = true;
    for (let t = 10; t < 30; t++) {
      if (Math.abs(out.data[t]! - 7) > 0.1) {
        allSeven = false;
        break;
      }
    }
    expect(allSeven).toBe(false);
  });

  it("calls the trunk 100 times like the no-inpaint AR loop", async () => {
    let calls = 0;
    const session: TrunkSession = {
      executionProvider: "mock",
      async run() {
        calls++;
        return zeros([1, 263, 1, 40]);
      },
      release() {},
    };
    await autoregressiveSampleWithInpainting(
      session,
      {
        textEmbed: zeros([1, 1, 512]),
        initialPrefix: zeros([1, 263, 1, 20]),
        referenceMotion: zeros([1, 263, 1, 196]),
        inpaintingMask: zeros([1, 263, 1, 196]),
      },
      { config: cfg, seed: 42 },
    );
    expect(calls).toBe(5 * 10 * 2);
  });

  it("upper-body mask (broadcast over time) preserves selected features", async () => {
    // Make an upper-body mask: features 0..130 = upper (inpaint, mask=0),
    // features 130..263 = lower (preserve, mask=1).
    const partition: boolean[] = [];
    for (let i = 0; i < 263; i++) partition.push(i >= 130);
    const mask = buildFeatureMask(partition);

    // Reference all 5s
    const reference = (() => {
      const t = zeros([1, 263, 1, 196]);
      t.data.fill(5);
      return t;
    })();

    // Trunk returns x*0.7 — output won't be 5 in unmasked features.
    const trunk: TrunkSession = {
      executionProvider: "mock",
      async run(inputs: TrunkInputs) {
        const out = copy(inputs.x);
        for (let i = 0; i < out.data.length; i++) out.data[i] = out.data[i]! * 0.7;
        return out;
      },
      release() {},
    };

    const out = await autoregressiveSampleWithInpainting(
      trunk,
      {
        textEmbed: zeros([1, 1, 512]),
        initialPrefix: zeros([1, 263, 1, 20]),
        referenceMotion: reference,
        inpaintingMask: mask,
      },
      { config: cfg, seed: 42 },
    );

    // Feature 200 (lower body, preserve) should be ~5 across all time.
    const lowerFeatStride = 200 * 196;
    for (let t = 0; t < 196; t++) {
      expect(out.data[lowerFeatStride + t]!).toBeCloseTo(5, 1);
    }
    // Feature 50 (upper body, inpaint) should differ from 5 somewhere.
    const upperFeatStride = 50 * 196;
    let allFive = true;
    for (let t = 0; t < 196; t++) {
      if (Math.abs(out.data[upperFeatStride + t]! - 5) > 0.1) {
        allFive = false;
        break;
      }
    }
    expect(allFive).toBe(false);
  });
});
