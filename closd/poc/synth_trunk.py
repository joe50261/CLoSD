"""
Synthetic DiP trunk for end-to-end PoC infrastructure validation.

WHY THIS EXISTS
  The real DiP checkpoint and CLIP weights live on HuggingFace + OpenAI's
  CDN, both of which are blocked by the sandbox where the deploy automation
  runs. Without those weights we can't run the real export.

  This module builds a SHAPE-COMPATIBLE, RANDOMLY-WEIGHTED transformer with
  the same six-input ONNX contract as export_onnx.py's TrunkWrapper. The
  resulting ONNX is structurally a real DiP model — it concatenates prefix,
  has a transformer encoder, strips the prefix from the output, predicts
  x_start over the 40-frame predict region. Just the weights are random.

WHAT IT VALIDATES (when paired with synth_fixtures.py)
  - The ONNX export pipeline runs end-to-end.
  - ORT-Web on WebGPU loads the file and produces output.
  - The TS scheduler / CFG / AR loop are numerically correct (parity MAE
    against fixtures from the same module should be ~0).
  - The Pages deploy serves a runnable parity harness without 404s.

WHAT IT DOES NOT VALIDATE
  - Whether the real DiP checkpoint exports cleanly (op support, shape
    inference). That risk only fires when the user runs export_onnx.py
    on a machine that can fetch the checkpoint.
  - Whether DiP's actual learned weights produce coherent motion in
    the browser.

Architecture (matches MDM no-target trunk's public interface):
  Inputs  matching closd/poc/SAMPLER_NOTES.md §2:
    x                : float32 [1, 263, 1, 40]
    timesteps        : int64   [1]
    text_embed       : float32 [1, 1, 512]   (raw CLIP-style projection)
    mask             : bool    [1, 1, 1, 40]
    prefix           : float32 [1, 263, 1, 20]
    text_uncond_mask : float32 [1]           (0 = cond, 1 = uncond)
  Output:
    pred_xstart      : float32 [1, 263, 1, 40]

Internal sizing is deliberately small (latent_dim=64, layers=2) so the
ONNX file stays under 1 MB and we can commit it to git without LFS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

CONTEXT_LEN = 20
PRED_LEN = 40
FEATURE_DIM = 263
CLIP_DIM = 512
N_DIFFUSION_STEPS = 10
LATENT_DIM = 64
N_LAYERS = 2
N_HEADS = 2
FF_SIZE = 128
SEED = 10


class DipSynthTrunk(nn.Module):
    """Mirrors mdm.MDM.forward semantics for the no-target ckpt:
    - prefix concat
    - text_embed as cached CLIP output, zeroed when text_uncond_mask=1
    - timestep embedding added to text embedding
    - transformer encoder
    - prefix-strip on output
    """

    def __init__(self) -> None:
        super().__init__()
        self.context_len = CONTEXT_LEN
        self.pred_len = PRED_LEN

        self.input_proj = nn.Linear(FEATURE_DIM, LATENT_DIM)
        self.timestep_emb = nn.Embedding(N_DIFFUSION_STEPS, LATENT_DIM)
        self.text_proj = nn.Linear(CLIP_DIM, LATENT_DIM)
        self.pos_emb = nn.Parameter(
            torch.randn(CONTEXT_LEN + PRED_LEN + 1, 1, LATENT_DIM) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=LATENT_DIM,
            nhead=N_HEADS,
            dim_feedforward=FF_SIZE,
            dropout=0.0,
            activation="gelu",
            batch_first=False,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.output_proj = nn.Linear(LATENT_DIM, FEATURE_DIM)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text_embed: torch.Tensor,
        mask: torch.Tensor,  # noqa: ARG002 — preserved for shape contract
        prefix: torch.Tensor,
        text_uncond_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x:      [1, 263, 1, 40]
        # prefix: [1, 263, 1, 20]
        full = torch.cat([prefix, x], dim=-1)  # [1, 263, 1, 60]

        # → [60, 1, 263] for transformer input
        full = full.squeeze(2).permute(2, 0, 1)
        full_proj = self.input_proj(full)  # [60, 1, 64]

        # Timestep embedding [1, 64] → [1, 1, 64]
        time_emb = self.timestep_emb(timesteps).unsqueeze(0)
        # Text projection through Linear, then mask-cond zeroing (matches
        # mdm.py:233 / mdm.py:159-163).
        text_proj = self.text_proj(text_embed)  # [1, 1, 64]
        text_proj = text_proj * (1.0 - text_uncond_mask).view(1, 1, 1)
        emb = time_emb + text_proj  # [1, 1, 64]

        seq = torch.cat([emb, full_proj], dim=0)  # [61, 1, 64]
        seq = seq + self.pos_emb[: seq.shape[0]]
        out = self.encoder(seq)  # [61, 1, 64]
        out = out[1:]  # drop emb token → [60, 1, 64]
        out = out[self.context_len :]  # strip prefix → [40, 1, 64]
        out = self.output_proj(out)  # [40, 1, 263]

        # → [1, 263, 1, 40]
        out = out.permute(1, 2, 0).unsqueeze(2)
        return out


def build_model() -> DipSynthTrunk:
    torch.manual_seed(SEED)
    m = DipSynthTrunk()
    m.eval()
    return m


def export_onnx(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model()

    inputs = (
        torch.zeros(1, FEATURE_DIM, 1, PRED_LEN, dtype=torch.float32),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, 1, CLIP_DIM, dtype=torch.float32),
        torch.ones(1, 1, 1, PRED_LEN, dtype=torch.bool),
        torch.zeros(1, FEATURE_DIM, 1, CONTEXT_LEN, dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32),
    )
    input_names = [
        "x",
        "timesteps",
        "text_embed",
        "mask",
        "prefix",
        "text_uncond_mask",
    ]
    output_names = ["pred_xstart"]

    print(f"[synth] exporting {out_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )

    # Sanity-check parity between PyTorch eager and ORT CPU.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    feed = {n: t.numpy() for n, t in zip(input_names, inputs)}
    ort_out = sess.run(output_names, feed)[0]
    with torch.no_grad():
        torch_out = model(*inputs).numpy()
    err = np.abs(ort_out - torch_out).mean()
    print(f"[synth] ORT-vs-PyTorch CPU FP32 MAE = {err:.2e}  (target: < 1e-5)")
    assert err < 1e-4, f"ONNX exported drifted: {err}"

    # Also write an FP16 variant for browser-side WebGPU efficiency.
    fp16_path = out_path.with_suffix(".fp16.onnx" if out_path.suffix == ".onnx" else out_path.suffix)
    if out_path.suffix == ".onnx":
        fp16_path = Path(str(out_path).replace(".onnx", ".fp16.onnx"))
    from onnxconverter_common.float16 import convert_float_to_float16
    import onnx as onnx_mod

    m_fp16 = convert_float_to_float16(onnx_mod.load(str(out_path)), keep_io_types=True)
    onnx_mod.save(m_fp16, str(fp16_path))
    print(f"[synth] FP16 → {fp16_path}")


def generate_fixtures(out_dir: Path) -> None:
    """Produce parity fixtures by running the same DipSynthTrunk model that
    we just exported. The browser will run the FP16 ONNX version of this
    model, so the per-step intermediates from the FP32 PyTorch reference
    are what TS will check against (within FP16 tolerance).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model()

    # Fixed-seed text embedding to mimic CLIP output. This stands in for the
    # transformers.js CLIP run in the browser — for this synth deploy the
    # browser-side text encoder isn't actually invoked because parity.ts
    # would compare against this saved tensor. (See note in synth_README.md.)
    g = torch.Generator().manual_seed(SEED + 1)
    text_embed = torch.randn(1, 1, CLIP_DIM, generator=g)
    np.save(out_dir / "text_embed.npy", text_embed.numpy())
    (out_dir / "prompt.txt").write_text("a person walks forward")  # for the harness UI

    # Initial AR prefix from a fresh RNG.
    g_prefix = torch.Generator().manual_seed(SEED + 2)
    prefix0 = torch.randn(1, FEATURE_DIM, 1, CONTEXT_LEN, generator=g_prefix)
    np.save(out_dir / "prefix.npy", prefix0.numpy())

    mask = torch.ones(1, 1, 1, PRED_LEN, dtype=torch.bool)
    np.save(out_dir / "mask.npy", mask.numpy())

    # Build the same beta/alpha schedule the TS scheduler uses (cosine, 10 steps).
    # Mirrors gaussian_diffusion.py:49-66 + buildSchedule in scheduler.ts.
    def alpha_bar(s: float) -> float:
        x = (s + 0.008) / 1.008
        return float(np.cos(x * np.pi / 2) ** 2)

    betas = []
    for i in range(N_DIFFUSION_STEPS):
        t1 = i / N_DIFFUSION_STEPS
        t2 = (i + 1) / N_DIFFUSION_STEPS
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
    betas_np = np.array(betas, dtype=np.float64)
    alphas = 1.0 - betas_np
    alphas_cumprod = np.cumprod(alphas)
    alphas_cumprod_prev = np.concatenate([[1.0], alphas_cumprod[:-1]])
    sqrt_recip_acp = np.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_acp = np.sqrt(1.0 / alphas_cumprod - 1.0)

    GUIDANCE_SCALE = 7.5

    # Two-iteration AR loop with per-step intermediates saved.
    cur_prefix = prefix0.clone()
    motions = []

    for it in range(2):
        iter_dir = out_dir / f"iter{it}"
        iter_dir.mkdir(exist_ok=True)
        np.save(iter_dir / "prefix.npy", cur_prefix.numpy())

        g_x = torch.Generator().manual_seed(SEED + 100 + it)
        x = torch.randn(1, FEATURE_DIM, 1, PRED_LEN, generator=g_x)
        np.save(out_dir / f"x_T_iter{it}.npy", x.numpy())

        for step_idx, t_int in enumerate(reversed(range(N_DIFFUSION_STEPS))):
            np.save(iter_dir / f"x_t_step{step_idx:02d}.npy", x.numpy())

            t_tensor = torch.tensor([t_int], dtype=torch.int64)

            with torch.no_grad():
                cond_out = model(
                    x, t_tensor, text_embed, mask, cur_prefix,
                    torch.zeros(1, dtype=torch.float32),  # cond
                )
                uncond_out = model(
                    x, t_tensor, text_embed, mask, cur_prefix,
                    torch.ones(1, dtype=torch.float32),   # uncond
                )

            np.save(iter_dir / f"cond_step{step_idx:02d}.npy", cond_out.numpy())
            np.save(iter_dir / f"uncond_step{step_idx:02d}.npy", uncond_out.numpy())

            pred_xstart = uncond_out + GUIDANCE_SCALE * (cond_out - uncond_out)
            np.save(iter_dir / f"pred_xstart_step{step_idx:02d}.npy", pred_xstart.numpy())

            # DDIM step (eta=0)
            a = float(sqrt_recip_acp[t_int])
            b = float(sqrt_recipm1_acp[t_int])
            eps = (a * x - pred_xstart) / b
            ab_prev = float(alphas_cumprod_prev[t_int])
            x = pred_xstart * np.sqrt(ab_prev) + eps * np.sqrt(1.0 - ab_prev)

        np.save(out_dir / f"motion_iter{it}.npy", x.numpy())
        motions.append(x.clone())
        cur_prefix = x[..., -CONTEXT_LEN:].clone()

    # Mean / std stats for the harness UI (set to zeros/ones since we have
    # no real HumanML3D dataset; the parity harness doesn't need them for
    # this synth deploy).
    np.save(out_dir / "mean.npy", np.zeros(FEATURE_DIM, dtype=np.float32))
    np.save(out_dir / "std.npy", np.ones(FEATURE_DIM, dtype=np.float32))

    print(f"[synth] phase1_core fixtures written to {out_dir}")
    print(f"[synth]  iter0 final motion norm = {motions[0].norm().item():.3f}")
    print(f"[synth]  iter1 final motion norm = {motions[1].norm().item():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["export", "fixtures", "all"], default="all")
    args = parser.parse_args()

    onnx_path = ARTIFACTS_DIR / "dip_no_target.onnx"
    fixtures_path = FIXTURES_DIR / "phase1_core"

    if args.mode in ("export", "all"):
        export_onnx(onnx_path)
        # Persist a synthetic args.json so the deployed args inspection works.
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS_DIR / "dip_no_target.args.json").write_text(
            json.dumps({
                "synthetic": True,
                "context_len": CONTEXT_LEN,
                "pred_len": PRED_LEN,
                "diffusion_steps": N_DIFFUSION_STEPS,
                "latent_dim": LATENT_DIM,
                "layers": N_LAYERS,
                "feature_dim": FEATURE_DIM,
                "note": "Random-weights synth model — see closd/poc/synth_trunk.py docstring.",
            }, indent=2)
        )

    if args.mode in ("fixtures", "all"):
        generate_fixtures(fixtures_path)


if __name__ == "__main__":
    sys.exit(main() or 0)
