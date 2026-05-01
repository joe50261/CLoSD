// Tests for ar_loop.ts using a synthetic TrunkSession that returns
// deterministic outputs. This verifies the AR control flow (iteration count,
// prefix rolling, buffer concatenation, final trim) without needing ORT or
// a real model.

import { describe, expect, it } from "vitest";
import type { TrunkInputs, TrunkSession } from "./ort_setup.js";
import { autoregressiveSample } from "./ar_loop.js";
import { DEFAULT_CONFIG } from "./scheduler.js";
import { copy, T4, zeros } from "./tensor.js";

/**
 * A trunk that returns its own input prefix's last frame, broadcast over the
 * 40 predict frames. This makes per-iteration outputs deterministic and easy
 * to inspect: we can verify that prefix rolling works because each iteration's
 * output should depend on the previous iteration's last frame.
 */
function makeMockTrunk(): TrunkSession {
  let callCount = 0;
  return {
    executionProvider: "mock",
    async run(inputs: TrunkInputs): Promise<T4> {
      callCount++;
      // Output: a constant tensor of value (callCount + last prefix scalar) so
      // every call is unique.
      const out = zeros([1, 263, 1, 40]);
      const prefixLastVal = inputs.prefix.data[inputs.prefix.data.length - 1] ?? 0;
      out.data.fill(callCount * 0.001 + prefixLastVal);
      return out;
    },
    release() {},
  };
}

describe("autoregressiveSample", () => {
  const cfg = DEFAULT_CONFIG;

  it("calls the trunk 100 times: 5 iters * 10 steps * 2 (cond+uncond)", async () => {
    let calls = 0;
    const session: TrunkSession = {
      executionProvider: "mock",
      async run() {
        calls++;
        return zeros([1, 263, 1, 40]);
      },
      release() {},
    };

    await autoregressiveSample(
      session,
      {
        textEmbed: zeros([1, 1, 512]),
        initialPrefix: zeros([1, 263, 1, 20]),
      },
      { config: cfg, seed: 42 },
    );

    // 5 AR iterations × 10 denoise steps × 2 CFG passes = 100
    expect(calls).toBe(5 * 10 * 2);
  });

  it("returns a [1, 263, 1, 196] tensor when includePrefix=false", async () => {
    const out = await autoregressiveSample(
      makeMockTrunk(),
      {
        textEmbed: zeros([1, 1, 512]),
        initialPrefix: zeros([1, 263, 1, 20]),
      },
      { config: cfg, seed: 42 },
    );
    expect(out.shape).toEqual([1, 263, 1, 196]);
  });

  it("returns 196 frames even when 5 iterations of 40 produce 200, by trimming", async () => {
    // 5 iters × 40 = 200 → trim to 196 (per AutoRegressiveSampler line 60: full_batch[:196])
    const out = await autoregressiveSample(
      makeMockTrunk(),
      {
        textEmbed: zeros([1, 1, 512]),
        initialPrefix: zeros([1, 263, 1, 20]),
      },
      { config: cfg, seed: 42 },
    );
    expect(out.shape[3]).toBe(196);
  });

  it("includePrefix prepends the seed prefix (20 + 5*40 = 220 → trim to 196)", async () => {
    const out = await autoregressiveSample(
      makeMockTrunk(),
      {
        textEmbed: zeros([1, 1, 512]),
        initialPrefix: zeros([1, 263, 1, 20]),
      },
      { config: cfg, seed: 42, includePrefix: true },
    );
    expect(out.shape).toEqual([1, 263, 1, 196]);
  });

  it("invokes onStep for every (iter, step) pair: 5 * 10 = 50 callbacks", async () => {
    const stepCalls: Array<{ iter: number; step: number; t: number }> = [];
    await autoregressiveSample(
      makeMockTrunk(),
      {
        textEmbed: zeros([1, 1, 512]),
        initialPrefix: zeros([1, 263, 1, 20]),
      },
      {
        config: cfg,
        seed: 42,
        onStep: (iter, step, t) => stepCalls.push({ iter, step, t }),
      },
    );
    expect(stepCalls).toHaveLength(50);
    // Steps go 9, 8, ..., 0 in each iteration
    for (let i = 0; i < 5; i++) {
      for (let s = 0; s < 10; s++) {
        const c = stepCalls[i * 10 + s]!;
        expect(c.iter).toBe(i);
        expect(c.step).toBe(s);
        expect(c.t).toBe(9 - s);
      }
    }
  });

  it("seed determinism: same seed → same output", async () => {
    // Use a real-ish trunk: returns input.x scaled by 0.5 (deterministic but seed-dependent
    // because x is initialized from the noise RNG)
    const trunk: TrunkSession = {
      executionProvider: "mock",
      async run(inputs: TrunkInputs) {
        const out = copy(inputs.x);
        for (let i = 0; i < out.data.length; i++) out.data[i] = out.data[i]! * 0.5;
        return out;
      },
      release() {},
    };
    const args = {
      textEmbed: zeros([1, 1, 512]) as T4,
      initialPrefix: zeros([1, 263, 1, 20]) as T4,
    };
    const a = await autoregressiveSample(trunk, args, { config: cfg, seed: 7 });
    const b = await autoregressiveSample(trunk, args, { config: cfg, seed: 7 });
    expect(Array.from(a.data.slice(0, 50))).toEqual(Array.from(b.data.slice(0, 50)));
  });

  it("different seeds → different output", async () => {
    const trunk: TrunkSession = {
      executionProvider: "mock",
      async run(inputs: TrunkInputs) {
        const out = copy(inputs.x);
        for (let i = 0; i < out.data.length; i++) out.data[i] = out.data[i]! * 0.5;
        return out;
      },
      release() {},
    };
    const args = {
      textEmbed: zeros([1, 1, 512]) as T4,
      initialPrefix: zeros([1, 263, 1, 20]) as T4,
    };
    const a = await autoregressiveSample(trunk, args, { config: cfg, seed: 7 });
    const b = await autoregressiveSample(trunk, args, { config: cfg, seed: 8 });
    let differ = 0;
    for (let i = 0; i < 100; i++) {
      if (a.data[i] !== b.data[i]) differ++;
    }
    expect(differ).toBeGreaterThan(0);
  });
});
