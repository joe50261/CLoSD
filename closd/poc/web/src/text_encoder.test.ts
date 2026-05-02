// Smoke tests for the browser text encoders.
//
// transformers.js downloads ~60–80 MB of weights on first run and requires a
// WebGPU or WASM-capable runtime — so these tests are gated behind
// RUN_NETWORK_TESTS=1. The Day 6 parity harness in the browser is the actual
// correctness check; here we just verify the API contract (shape, dtype) holds.

import { describe, expect, it } from "vitest";
import { loadBertTextEncoder, loadTextEncoder } from "./text_encoder.js";

const SHOULD_RUN = process.env.RUN_NETWORK_TESTS === "1";

describe.skipIf(!SHOULD_RUN)("text_encoder (network-gated)", () => {
  it("CLIP returns a [1, 1, 512] tensor for a single prompt", async () => {
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

  it("DistilBERT returns [T_text, 1, 768] hidden + [1, T_text] padding mask", async () => {
    const enc = await loadBertTextEncoder({ device: "wasm" });
    try {
      const out = await enc.encode("a person walks forward");
      // T_text is at minimum 6 for "a person walks forward" with [CLS]/[SEP]
      // wrappers. We don't pin the exact value (tokenizer versions can vary),
      // only the contract.
      expect(out.embed.shape.length).toBe(3);
      expect(out.embed.shape[1]).toBe(1);
      expect(out.embed.shape[2]).toBe(768);
      expect(out.embed.shape[0]).toBeGreaterThanOrEqual(4);
      expect(out.paddingMask.shape).toEqual([1, out.embed.shape[0]]);
      // Single-prompt no-padding run: every mask value should be 0 (real token).
      for (let i = 0; i < out.paddingMask.data.length; i++) {
        expect(out.paddingMask.data[i]).toBe(0);
      }
      // Hidden states shouldn't be all-zero.
      let norm = 0;
      for (let i = 0; i < out.embed.data.length; i++)
        norm += out.embed.data[i]! ** 2;
      expect(Math.sqrt(norm)).toBeGreaterThan(0.1);
    } finally {
      enc.release();
    }
  });
});
