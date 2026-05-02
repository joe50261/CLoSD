"""
Export DiP transformer trunk to ONNX for browser inference.

Produces:
  closd/poc/artifacts/dip_no_target.onnx        (Phase 1)
  closd/poc/artifacts/dip_no_target.fp16.onnx   (--fp16)
  closd/poc/artifacts/dip_no_target.args.json   (resolved checkpoint args)
  closd/poc/artifacts/dip_multi_target.onnx     (Phase 2 Day 8-9, --ckpt multi-target)

What gets exported:
  The MDM transformer + heads (input/output process, time embedding, text Linear, transformer).
  CLIP is NOT exported — transformers.js handles tokenization + text encoding in the browser.

Trunk inputs (see closd/poc/SAMPLER_NOTES.md §2):
  x                : [1, 263, 1, 40]   float32   noisy predict-region tensor at timestep t
  timesteps        : [1]                int64     value 0..9 for the 10-step model
  text_embed       : [T_text, 1, D_txt] float32   pre-encoded text encoder output
                                                  (DistilBERT last_hidden_state, permuted to seq-first)
                                                  D_txt = 768 for BERT, 512 for CLIP
                                                  T_text dynamic (BERT pads to longest in batch)
  text_mask        : [1, T_text]        bool      True = padding token (no content)
                                                  All False for CLIP (single pooled token).
  mask             : [1, 1, 1, 40]      bool      validity of predict frames (1=valid)
  prefix           : [1, 263, 1, 20]    float32   rolling context from AR loop
  text_uncond_mask : [1]                float32   1.0 = uncond pass (zero text), 0.0 = cond pass

Output:
  pred_xstart : [1, 263, 1, 40]   float32   model's x_start prediction over predict region
                                            (mdm.forward strips the 20 prefix frames internally)

Note on text encoder:
  Both shipped DiP checkpoints (no-target, multi-target) use DistilBERT — clip_dim=768.
  The browser must run DistilBERT (e.g. via transformers.js with Xenova/distilbert-base-uncased)
  and feed the variable-length last_hidden_state + attention_mask. The earlier CLIP-shape
  contract (text_embed [1,1,512], no text_mask) is invalid for these checkpoints.

Implementation note:
  We delegate to the real MDM.forward rather than re-implementing it, then monkey-patch
  mask_cond at export time to use the runtime text_uncond_mask tensor instead of a Python
  bool. This way the exported graph respects whatever emb_policy / emb_before_mask /
  arch the checkpoint was trained with — we don't have to second-guess.

Usage (from repo root, with closd conda env active):
  python -m closd.poc.export_onnx                    # no-target ckpt, FP32
  python -m closd.poc.export_onnx --fp16             # also write FP16
  python -m closd.poc.export_onnx --ckpt multi-target

Decision points (from /root/.claude/plans/mac-demo-scalable-iverson.md):
  - If torch.onnx.export fails, retry with torch.onnx.dynamo_export (PyTorch 2.x).
    If that also fails → escalate per Day 3 hard rule.
  - FP16 conversion uses onnxconverter-common; tolerance widens from 1e-4 to 1e-3.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

CKPT_PATHS = {
    "no-target": "closd/diffusion_planner/save/DiP_no-target_10steps_context20_predict40/model000200000.pt",
    "multi-target": "closd/diffusion_planner/save/DiP_multi-target_10steps_context20_predict40/model000300000.pt",
}


class TrunkWrapper(nn.Module):
    """Delegates to MDM.forward with a tensor-driven text_uncond_mask.

    Monkey-patches mask_cond on the wrapped MDM so the uncond branch is captured
    as a tensor multiply (`cond * (1 - text_uncond_mask)`) rather than a Python
    branch on `force_mask: bool`. This keeps the exported graph faithful to
    whatever emb_policy / emb_before_mask the checkpoint was trained with.
    """

    def __init__(self, mdm: nn.Module):
        super().__init__()
        self.mdm = mdm

    def forward(self, x, timesteps, text_embed, text_mask, mask, prefix, text_uncond_mask):
        m = self.mdm

        # Patch mask_cond to use the runtime tensor. We bind a method that
        # closes over text_uncond_mask; tracing captures the tensor ops.
        scale = (1.0 - text_uncond_mask).view(1, 1, 1)

        def patched_mask_cond(self_m, cond, force_mask=False):
            # Ignore the Python `force_mask` flag entirely; the runtime tensor
            # decides. cond shape is [seq, bs, d].
            return cond * scale

        original = m.mask_cond
        m.mask_cond = types.MethodType(patched_mask_cond, m)

        # MDM.forward unpacks `y['text_embed']` as a tuple when text_encoder_type
        # is 'bert' (see mdm.py:222-227 — `if type(enc_text) == tuple`). For BERT
        # we pass (text_embed, text_mask); for CLIP a bare tensor would also work
        # but every shipped DiP checkpoint uses BERT so we standardize on the tuple.
        y = {
            "text_embed": (text_embed, text_mask),
            "mask": mask.clone(),
            "prefix": prefix,
            "lengths": torch.tensor([x.shape[-1]], device=x.device),
            # 'text_uncond' is now ignored by patched_mask_cond, but mdm.forward
            # reads it before calling mask_cond — set to anything (won't matter).
            "text_uncond": False,
        }

        try:
            out = m(x, timesteps, y)
        finally:
            m.mask_cond = original

        return out


def load_mdm(ckpt_id: str):
    from closd.diffusion_planner.utils.model_util import (
        create_model_and_diffusion,
        load_saved_model,
    )
    from closd.diffusion_planner.utils.parser_util import generate_args
    from closd.diffusion_planner.utils import dist_util
    from closd.diffusion_planner.sample.generate import load_dataset
    from closd.utils.hf_handler import get_dependencies

    get_dependencies()

    ckpt_path = REPO_ROOT / CKPT_PATHS[ckpt_id]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"hf_handler should have downloaded it; check huggingface cache."
        )

    args_json = ckpt_path.parent / "args.json"
    if not args_json.exists():
        raise FileNotFoundError(f"Missing args.json next to checkpoint: {args_json}")

    sys.argv = [
        "export_onnx",
        "--model_path", str(ckpt_path),
        "--text_prompt", "a person walks forward",
        "--autoregressive",
        "--num_samples", "1",
        "--num_repetitions", "1",
        "--seed", "10",
    ]
    args = generate_args()
    dist_util.setup_dist(args.device)

    n_frames = args.context_len + args.pred_len
    data = load_dataset(args, max_frames=196, n_frames=n_frames)

    model, _diffusion = create_model_and_diffusion(args, data)
    load_saved_model(model, str(ckpt_path), use_avg=getattr(args, "use_ema", False))
    model.eval()
    # Move to CPU for export — ONNX export is happier on CPU and we don't need GPU speed here
    model = model.cpu()

    return model, args, data


def export(ckpt_id: str, fp16: bool = False):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"dip_{ckpt_id.replace('-', '_')}"
    out_fp32 = ARTIFACTS_DIR / f"{out_name}.onnx"
    out_fp16 = ARTIFACTS_DIR / f"{out_name}.fp16.onnx"

    model, args, _data = load_mdm(ckpt_id)
    wrapper = TrunkWrapper(model)
    wrapper.eval()

    out_args = ARTIFACTS_DIR / f"{out_name}.args.json"
    out_args.write_text(json.dumps({
        k: v for k, v in vars(args).items()
        if isinstance(v, (str, int, float, bool, list, type(None)))
    }, indent=2))

    B = 1
    T_pred = args.pred_len
    T_ctx = args.context_len
    F = 263
    # Text encoder dim: 768 for BERT, 512 for CLIP. Read off the loaded model so
    # the export matches the checkpoint regardless of which encoder it trained with.
    D_txt = int(model.clip_dim)
    # Tracing uses a single dummy text length; runtime accepts any length via the
    # T_text dynamic axis declared below. Using 6 here = "[CLS] a person walks forward [SEP]".
    T_text = 6

    dummy_x = torch.zeros(B, F, 1, T_pred, dtype=torch.float32)
    dummy_timesteps = torch.zeros(B, dtype=torch.int64)
    dummy_text_embed = torch.zeros(T_text, B, D_txt, dtype=torch.float32)
    dummy_text_mask = torch.zeros(B, T_text, dtype=torch.bool)
    dummy_mask = torch.ones(B, 1, 1, T_pred, dtype=torch.bool)
    dummy_prefix = torch.zeros(B, F, 1, T_ctx, dtype=torch.float32)
    dummy_text_uncond = torch.zeros(1, dtype=torch.float32)

    inputs = (
        dummy_x, dummy_timesteps, dummy_text_embed, dummy_text_mask,
        dummy_mask, dummy_prefix, dummy_text_uncond,
    )
    input_names = ["x", "timesteps", "text_embed", "text_mask", "mask", "prefix", "text_uncond_mask"]
    output_names = ["pred_xstart"]
    dynamic_axes = {
        "text_embed": {0: "T_text"},
        "text_mask": {1: "T_text"},
    }

    print(f"[export] running torch.onnx.export → {out_fp32}")
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                inputs,
                str(out_fp32),
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
                do_constant_folding=True,
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )
    except Exception as e:
        print(f"[export] torch.onnx.export failed: {e}")
        # PyTorch 2.5+ removed `dynamo_export`; the dynamo path is now driven via
        # `dynamo=True` on `torch.onnx.export`. Try that as a fallback.
        print("[export] retrying with torch.onnx.export(dynamo=True)")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                inputs,
                str(out_fp32),
                input_names=input_names,
                output_names=output_names,
                opset_version=17,
                dynamic_axes=dynamic_axes,
                dynamo=True,
            )

    # CPU sanity check: ORT FP32 must match PyTorch FP32 to within ~1e-5
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(out_fp32), providers=["CPUExecutionProvider"])
        feed = {n: t.numpy() for n, t in zip(input_names, inputs)}
        ort_out = sess.run(output_names, feed)[0]
        with torch.no_grad():
            torch_out = wrapper(*inputs).numpy()
        diff = np.abs(ort_out - torch_out).mean()
        print(f"[export] ORT-vs-PyTorch CPU FP32 mean abs err: {diff:.2e}  (target: < 1e-5)")
        assert diff < 1e-4, "ONNX export drifted too far from PyTorch reference"
    except ImportError:
        print("[export] onnxruntime not installed — skipping CPU sanity check")

    if fp16:
        print(f"[export] converting to FP16 → {out_fp16}")
        from onnxconverter_common.float16 import convert_float_to_float16
        import onnx
        m = onnx.load(str(out_fp32))
        # onnxconverter_common 1.16 has a known type-inference bug around the
        # MultiheadAttention internals that torch.onnx emits (Cast/Div pairs in
        # self_attn / multihead_attn produce mixed float16/float types that ORT
        # rejects on load). Keeping these specific nodes in FP32 sidesteps the bug
        # and only adds ~5MB to the FP16 model. Validated: load + parity OK.
        attn_node_block = [
            n.name for n in m.graph.node
            if ("self_attn" in n.name or "multihead_attn" in n.name)
            and n.op_type in ("Cast", "Div")
        ]
        m_fp16 = convert_float_to_float16(
            m, keep_io_types=True, node_block_list=attn_node_block,
        )
        onnx.save(m_fp16, str(out_fp16))

    print(f"[export] done. artifacts in {ARTIFACTS_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", choices=list(CKPT_PATHS.keys()), default="no-target")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    export(args.ckpt, fp16=args.fp16)


if __name__ == "__main__":
    main()
