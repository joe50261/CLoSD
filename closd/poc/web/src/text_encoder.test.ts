// Smoke test for the CLIP text encoder.
//
// transformers.js needs to download the Xenova/clip-vit-base-patch32 weights
// (~80MB) on first run and requires a WebGPU or WASM-capable runtime — so
// this test is gated behind RUN_NETWORK_TESTS=1. The Day 6 parity harness in
// the browser is the actual correctness check; here we just verify the
// API contract (shape, dtype) holds.

import { describe, expect, it } from "vitest";
import { loadTextEncoder } from "./text_encoder.js";

const SHOULD_RUN = process.env.RUN_NETWORK_TESTS === "1";

describe.skipIf(!SHOULD_RUN)("text_encoder (network-gated)", () => {
  it("returns a [1, 1, 512] tensor for a single prompt", async () => {
    const enc = await loadTextEncoder({ device: "wasm" });
    try {
      const out = await enc.encode("a person walks forward");
      expect(out.shape).toEqual([1, 1, 512]);
      expect(out.data).toBeInstanceOf(Float32Array);
      // The DiP "a person walks forward" prompt produces a non-trivially-norm vector.
      let norm = 0;
      for (let i = 0; i < out.data.length; i++) norm += out.data[i]! ** 2;
      expect(Math.sqrt(norm)).toBeGreaterThan(0.1);
    } finally {
      enc.release();
    }
  });
});
