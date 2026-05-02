// Browser text encoders via @huggingface/transformers (transformers.js).
//
// Why this isn't in the ONNX trunk: the text encoder is a separate model with
// its own tokenizer, weights, and projection. Running it in the browser via
// transformers.js saves us re-implementing tokenization and pulling those
// weights into our ONNX bundle.
//
// Two flavors are supported, matching the two text-encoder branches in
// closd/diffusion_planner/model/mdm.py:
//
//   1. CLIP (mdm.py:166-181, text_encoder_type == "clip"):
//        tokens = clip.tokenize(text, context_length=22, truncate=True)  # [bs, 22]
//        tokens = pad_with_zeros_to(tokens, 77)                          # [bs, 77]
//        text_embed = clip_model.encode_text(tokens).unsqueeze(0)        # [1, bs, 512]
//      transformers.js id: Xenova/clip-vit-base-patch32
//
//   2. DistilBERT (mdm.py:183-190, text_encoder_type == "bert"):
//        out, mask = bert(text)                # last_hidden_state [bs, T_text, 768], attn_mask [bs, T_text]
//        out = out.permute(1, 0, 2)            # [T_text, bs, 768]
//        mask = ~mask                          # True = padding (PyTorch MHA convention)
//        return out, mask
//      transformers.js id: Xenova/distilbert-base-uncased
//
// Both shipped DiP checkpoints (no-target, multi-target) use DistilBERT, so the
// BERT path is the production code path. The CLIP path is kept around for
// backwards compatibility with older synth fixtures and any future CLIP-trained
// checkpoints.

import {
  AutoModel,
  AutoTokenizer,
  CLIPTextModelWithProjection,
} from "@huggingface/transformers";
import type { T4 } from "./tensor.js";

const MODEL_ID = "Xenova/clip-vit-base-patch32";
const BERT_MODEL_ID = "Xenova/distilbert-base-uncased";
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

// ---------------------------------------------------------------------------
// DistilBERT text encoder (production path for shipped DiP checkpoints).
// ---------------------------------------------------------------------------

export interface BertTextEncoderOutput {
  /** last_hidden_state permuted to seq-first: [T_text, 1, 768]. */
  embed: T4;
  /** Padding mask, True (1.0) = padding token. Shape [1, T_text]. Stored as
   *  Float32Array because the ONNX trunk expects it as a tensor we can compare
   *  against the .npy fixture (which fetchNpy decodes bool→float32). */
  paddingMask: T4;
}

export interface BertTextEncoder {
  /** Encode a single text prompt. T_text varies with the prompt length. */
  encode(text: string): Promise<BertTextEncoderOutput>;
  release(): void;
}

export interface BertTextEncoderOptions {
  device?: "webgpu" | "wasm";
  modelId?: string;
}

export async function loadBertTextEncoder(
  opts: BertTextEncoderOptions = {},
): Promise<BertTextEncoder> {
  const modelId = opts.modelId ?? BERT_MODEL_ID;
  const device = opts.device ?? "webgpu";

  const tokenizer = await AutoTokenizer.from_pretrained(modelId);
  const model = await AutoModel.from_pretrained(modelId, {
    device,
    dtype: "fp32",
  });

  const encode = async (text: string): Promise<BertTextEncoderOutput> => {
    // bs=1, no padding: BERT pads to longest in batch (mdm.py:28 generate_fixtures
    // single-prompt run produces tokens with no padding). T_text = number of tokens
    // including [CLS] and [SEP].
    const tokens = tokenizer(text, {
      padding: false,
      truncation: false,
      return_tensors: "pt",
    });

    const out = await model(tokens);
    const hidden = (out as any).last_hidden_state;
    if (!hidden) {
      throw new Error("BERT model output missing last_hidden_state field");
    }
    const dims = hidden.dims as number[];
    if (dims.length !== 3 || dims[0] !== 1) {
      throw new Error(`unexpected last_hidden_state shape ${dims.join("x")}`);
    }
    const tText = dims[1]!;
    const dTxt = dims[2]!;
    // last_hidden_state ships as Float32Array [bs=1, T_text, D]. The Python
    // reference permutes to [T_text, bs, D]; with bs=1 the underlying memory
    // layout is identical, so we just relabel the shape.
    const embedData = new Float32Array(hidden.data as Float32Array);

    // attention_mask is a BigInt64Array (or Int32Array depending on
    // transformers.js version) of shape [1, T_text]. Invert per mdm.py:189
    // so True = padding, then store as float32 to match the .npy fixture
    // layout fetched by parity.ts.
    const attnSrc = (tokens as any).attention_mask;
    if (!attnSrc) throw new Error("tokenizer output missing attention_mask");
    const attnLen = attnSrc.dims[1] ?? attnSrc.dims[attnSrc.dims.length - 1];
    if (attnLen !== tText) {
      throw new Error(
        `attention_mask length ${attnLen} != hidden T_text ${tText}`,
      );
    }
    const paddingMaskData = new Float32Array(tText);
    for (let i = 0; i < tText; i++) {
      // attention_mask: 1 = real token, 0 = padding. Inverted: 1 = padding.
      const v = Number(attnSrc.data[i]);
      paddingMaskData[i] = v === 0 ? 1 : 0;
    }

    return {
      embed: { data: embedData, shape: [tText, 1, dTxt] },
      paddingMask: { data: paddingMaskData, shape: [1, tText] },
    };
  };

  return { encode, release: () => model.dispose?.() };
}
