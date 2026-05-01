// CLIP text encoder via @huggingface/transformers (transformers.js).
//
// Why this isn't in the ONNX trunk: CLIP is a separate model with its own
// tokenizer, weights, and projection. Running it in the browser via the
// transformers.js pipeline saves us re-implementing tokenization and pulling
// CLIP weights into our ONNX bundle.
//
// Python reference (closd/diffusion_planner/model/mdm.py:166-181):
//   tokens = clip.tokenize(text, context_length=22, truncate=True)  # [bs, 22]
//   tokens = pad_with_zeros_to(tokens, 77)                          # [bs, 77]
//   text_embed = clip_model.encode_text(tokens).float().unsqueeze(0) # [1, bs, 512]
//
// transformers.js path:
//   tokenizer(text, { padding: 'max_length', max_length: 77, truncation: true })
//   model(tokens) -> .text_embeds  # [bs, 512]   (or .pooler_output, varies by model)
//   reshape -> [1, bs, 512]
//
// The HF model id Xenova/clip-vit-base-patch32 ships pre-converted ONNX weights
// for transformers.js with WebGPU support.

import { AutoTokenizer, CLIPTextModelWithProjection } from "@huggingface/transformers";
import type { T4 } from "./tensor.js";

const MODEL_ID = "Xenova/clip-vit-base-patch32";
// CLIP's standard context length. The DiP-side context_length=22 only affects
// truncation; the actual CLIP forward is always 77 tokens.
const TOKEN_MAX_LEN = 77;
// What DiP truncates to before zero-padding to 77 (mdm.py:172).
const DIP_TRUNCATE_LEN = 22;

export interface TextEncoder {
  /** Encode a single text prompt to a [1, 1, 512] float32 tensor. */
  encode(text: string): Promise<T4>;
  /** Free GPU resources held by the model. */
  release(): void;
}

export interface TextEncoderOptions {
  /** WebGPU is preferred; the pipeline falls back to WASM if unavailable.
   *  Set to "wasm" to force CPU for parity testing. */
  device?: "webgpu" | "wasm";
  /** Override model id, e.g. for FP16 variants. */
  modelId?: string;
  /** Truncate input to DiP's effective length (22 tokens) before zero-padding to 77.
   *  Defaults to true to match the Python reference exactly. */
  matchDipTruncation?: boolean;
}

export async function loadTextEncoder(
  opts: TextEncoderOptions = {},
): Promise<TextEncoder> {
  const modelId = opts.modelId ?? MODEL_ID;
  const device = opts.device ?? "webgpu";
  const truncateToDip = opts.matchDipTruncation ?? true;

  const tokenizer = await AutoTokenizer.from_pretrained(modelId);
  const model = await CLIPTextModelWithProjection.from_pretrained(modelId, {
    device,
    dtype: "fp32",
  });

  const encode = async (text: string): Promise<T4> => {
    // Step 1: tokenize. We always pad to 77 (CLIP's native max_length).
    // Truncate to 22 first if matching DiP semantics — this matters because
    // DiP-22 vs CLIP-77 truncation produces a different EOS token position
    // for prompts longer than 20 words, which changes the pooled embedding.
    const truncMax = truncateToDip ? DIP_TRUNCATE_LEN : TOKEN_MAX_LEN;
    const truncated = tokenizer(text, {
      padding: false,
      truncation: true,
      max_length: truncMax,
      return_tensors: "pt",
    });

    // Step 2: zero-pad to 77 (the actual CLIP forward length).
    const padded = padTokensTo(truncated, TOKEN_MAX_LEN, tokenizer);

    // Step 3: forward through CLIP text encoder.
    const out = await model(padded);
    // CLIPTextModelWithProjection returns text_embeds [bs, 512] (post-projection).
    // mdm.py uses clip_model.encode_text() which is exactly this projection path.
    const embedTensor = (out as any).text_embeds;
    if (!embedTensor) {
      throw new Error("CLIP model output missing text_embeds field");
    }
    const data = new Float32Array(embedTensor.data as Float32Array);
    const dims = embedTensor.dims as number[];
    if (dims.length !== 2 || dims[1] !== 512) {
      throw new Error(`unexpected text_embeds shape ${dims.join("x")}`);
    }
    // Reshape [bs, 512] -> [1, bs, 512] to match Python's unsqueeze(0).
    return { data, shape: [1, dims[0]!, 512] };
  };

  return { encode, release: () => model.dispose?.() };
}

/**
 * Pad input_ids and attention_mask to a fixed length with zeros, mirroring
 * mdm.py:175-177. Returns a new tokenized object usable as model input.
 */
function padTokensTo(
  inputs: any,
  targetLen: number,
  _tokenizer: any,
): any {
  // transformers.js token tensors are typed objects with .data and .dims.
  // We reconstruct them with zero-padding on the right.
  const inputIds = inputs.input_ids;
  const curLen = inputIds.dims[inputIds.dims.length - 1];
  if (curLen === targetLen) return inputs;
  if (curLen > targetLen) {
    throw new Error(`token length ${curLen} > target ${targetLen}; should have truncated`);
  }
  const bs = inputIds.dims[0];
  const padded = padTensor(inputIds, bs, targetLen, 0n);
  const maskTensor = inputs.attention_mask;
  const paddedMask =
    maskTensor !== undefined
      ? padTensor(maskTensor, bs, targetLen, 0n)
      : undefined;
  return paddedMask
    ? { input_ids: padded, attention_mask: paddedMask }
    : { input_ids: padded };
}

function padTensor(
  t: { data: BigInt64Array | Int32Array; dims: number[] },
  bs: number,
  targetLen: number,
  fill: bigint,
): { data: BigInt64Array | Int32Array; dims: number[] } {
  const isBigInt = t.data instanceof BigInt64Array;
  const newData = isBigInt
    ? new BigInt64Array(bs * targetLen)
    : new Int32Array(bs * targetLen);
  if (isBigInt) {
    (newData as BigInt64Array).fill(fill);
  } else {
    (newData as Int32Array).fill(Number(fill));
  }
  const curLen = t.dims[t.dims.length - 1]!;
  for (let b = 0; b < bs; b++) {
    for (let i = 0; i < curLen; i++) {
      const v = t.data[b * curLen + i]!;
      // TypedArrays of different element types share an index signature, but
      // TS narrows the union too aggressively here — cast to any to skirt it.
      (newData as any)[b * targetLen + i] = v;
    }
  }
  return { data: newData, dims: [bs, targetLen] };
}
