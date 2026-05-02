// ONNX Runtime Web setup: load the DiP trunk, configure the WebGPU EP,
// and expose a typed `run()` that takes T4 tensors and returns a T4 output.
//
// We hide the ORT API behind a thin façade so the rest of the PoC code
// doesn't depend on ort types — makes parity tests trivial to mock.
//
// Critical inputs/outputs match the export script's contract
// (see closd/poc/export_onnx.py docstring and SAMPLER_NOTES.md §2 + §3):
//   x          : float32 [1, 263, 1, 40]
//   timesteps  : int64   [1]
//   text_embed : float32 [T_text, 1, 768]   (DistilBERT last_hidden_state, seq-first)
//   text_mask  : bool    [1, T_text]        (True = padding token)
//   mask       : bool    [1, 1, 1, 40]
//   prefix     : float32 [1, 263, 1, 20]
//   text_uncond_mask : float32 [1]
//
//   pred_xstart : float32 [1, 263, 1, 40]
//
// T_text is a dynamic axis — the BERT tokenizer pads to the longest prompt
// in the batch. For B=1 it's just the prompt's tokenized length.

import * as ort from "onnxruntime-web";
import type { T4 } from "./tensor.js";

export interface TrunkInputs {
  x: T4;
  timestep: number;
  /** [T_text, 1, 768] DistilBERT last_hidden_state, seq-first. */
  textEmbed: T4;
  /** [1, T_text] bool, True = padding token. */
  textMask: T4;
  mask: T4;
  prefix: T4;
  textUncondMask: number; // 0 or 1
}

export interface TrunkSession {
  /** Last-attempted execution provider, e.g. "webgpu" or "wasm". */
  readonly executionProvider: string;
  run(inputs: TrunkInputs): Promise<T4>;
  release(): void;
}

export interface LoadOptions {
  /** Path to the .onnx file (FP32 or FP16). */
  modelUrl: string;
  /** Force a specific EP. Default: ["webgpu"] with no fallback so a fallback
   *  to WASM trips the success criterion in §"Loads" of the plan. */
  executionProviders?: ort.InferenceSession.ExecutionProviderConfig[];
  /** Set ORT log level for debugging. */
  logSeverity?: 0 | 1 | 2 | 3 | 4;
}

export async function loadTrunk(opts: LoadOptions): Promise<TrunkSession> {
  // Default to WebGPU only — we want hard failure if WebGPU is unavailable,
  // not a silent fallback to WASM. Plan §"Loads" makes this a go/no-go gate.
  const eps = opts.executionProviders ?? ["webgpu"];

  const session = await ort.InferenceSession.create(opts.modelUrl, {
    executionProviders: eps,
    graphOptimizationLevel: "all",
    logSeverityLevel: opts.logSeverity ?? 2,
  });

  // ORT exposes the actual EP used via the (sadly internal) options. We
  // record what we asked for; if it didn't take, the next .run() will
  // either error out or be slow enough to flag in perf.
  const executionProvider =
    typeof eps[0] === "string" ? eps[0] : (eps[0]?.name ?? "unknown");

  const run = async (inputs: TrunkInputs): Promise<T4> => {
    // ORT requires int64 as BigInt64Array — see SAMPLER_NOTES.md gotcha #5.
    const timesteps = new ort.Tensor(
      "int64",
      new BigInt64Array([BigInt(inputs.timestep)]),
      [1],
    );
    // bool tensors take Uint8Array — see SAMPLER_NOTES.md gotcha #6.
    const maskU8 = boolTensorBytes(inputs.mask);
    const textMaskU8 = boolTensorBytes(inputs.textMask);
    const feeds: Record<string, ort.Tensor> = {
      x: new ort.Tensor("float32", inputs.x.data, [...inputs.x.shape]),
      timesteps,
      text_embed: new ort.Tensor("float32", inputs.textEmbed.data, [
        ...inputs.textEmbed.shape,
      ]),
      text_mask: new ort.Tensor("bool", textMaskU8, [...inputs.textMask.shape]),
      mask: new ort.Tensor("bool", maskU8, [...inputs.mask.shape]),
      prefix: new ort.Tensor("float32", inputs.prefix.data, [
        ...inputs.prefix.shape,
      ]),
      text_uncond_mask: new ort.Tensor(
        "float32",
        new Float32Array([inputs.textUncondMask]),
        [1],
      ),
    };
    const result = await session.run(feeds);
    const out = result["pred_xstart"];
    if (!out) throw new Error("trunk did not return pred_xstart");
    if (out.type !== "float32") {
      throw new Error(`pred_xstart dtype ${out.type}, expected float32`);
    }
    const shape = out.dims as readonly number[];
    if (shape.length !== 4) {
      throw new Error(`pred_xstart rank ${shape.length}, expected 4`);
    }
    return {
      data: new Float32Array(out.data as Float32Array),
      shape: [shape[0]!, shape[1]!, shape[2]!, shape[3]!],
    };
  };

  return {
    executionProvider,
    run,
    release: () => session.release(),
  };
}

function boolTensorBytes(t: T4): Uint8Array {
  const out = new Uint8Array(t.data.length);
  for (let i = 0; i < t.data.length; i++) {
    out[i] = t.data[i]! > 0 ? 1 : 0;
  }
  return out;
}
