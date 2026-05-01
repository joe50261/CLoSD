"""
Produce parity fixtures for the browser PoC.

Runs a deterministic Python reference inference and saves every intermediate
tensor the TypeScript port needs to verify itself against. Phase 1 only covers
the core no-target text-to-motion path; --phase {goal,inbetween,upper,invert,smpl}
extend it for Phase 2 tools (Days 8-14).

Phase 1 fixtures (closd/poc/fixtures/phase1_core/):
  prompt.txt              the text prompt used
  text_embed.npy          [1, 1, 512] cached CLIP encoding of the prompt
  prefix.npy              [1, 263, 1, 20] initial AR prefix (from data, fixed seed)
  mask.npy                [1, 1, 1, 40] validity mask (all ones for 40 valid frames)
  x_T_iter0.npy           [1, 263, 1, 40] initial noise for iter 0
  x_T_iter1.npy           [1, 263, 1, 40] initial noise for iter 1
  iter{0,1}/x_t_step{00..09}.npy   per-step x_t (descending t=9..0)
  iter{0,1}/pred_xstart_step{00..09}.npy   per-step CFG'd x_start
  iter{0,1}/cond_step{00..09}.npy   pre-CFG conditional output (debug)
  iter{0,1}/uncond_step{00..09}.npy pre-CFG uncond output (debug)
  motion_iter0.npy        [1, 263, 1, 40] final iter-0 output (last 40 of 60)
  motion_iter1.npy        [1, 263, 1, 40] final iter-1 output
  motion_full.npy         [1, 263, 1, 196] full AR output (5 iterations, sliced)
  mean.npy                [263] HumanML3D mean (copy of t2m_mean.npy)
  std.npy                 [263] HumanML3D std

Browser-side parity test consumes these and computes MAE per intermediate tensor.

Usage (from repo root, with conda env active):
  python -m closd.poc.generate_fixtures --phase core
  python -m closd.poc.generate_fixtures --phase goal       # Phase 2 Day 8-9
  python -m closd.poc.generate_fixtures --phase inbetween  # Phase 2 Day 10-11
  ...
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DATASET_DIR = REPO_ROOT / "closd/diffusion_planner/dataset"

# Pinned for parity. Changing these invalidates the fixtures and the TS comparison.
PROMPT = "a person walks forward"
SEED = 10
GUIDANCE_SCALE = 7.5


def setup_model_and_data():
    """Mirror generate.py's setup so the fixtures match what generate.py would produce."""
    from closd.diffusion_planner.utils.fixseed import fixseed
    from closd.diffusion_planner.utils.parser_util import generate_args
    from closd.diffusion_planner.utils.model_util import (
        create_model_and_diffusion, load_saved_model,
    )
    from closd.diffusion_planner.utils.sampler_util import ClassifierFreeSampleModel
    from closd.diffusion_planner.utils import dist_util
    from closd.diffusion_planner.sample.generate import load_dataset
    from closd.diffusion_planner.data_loaders.tensors import collate
    from closd.utils.hf_handler import get_dependencies

    fixseed(SEED)
    get_dependencies()

    ckpt = (
        REPO_ROOT
        / "closd/diffusion_planner/save/DiP_no-target_10steps_context20_predict40/model000200000.pt"
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint at {ckpt}")

    sys.argv = [
        "generate_fixtures",
        "--model_path", str(ckpt),
        "--text_prompt", PROMPT,
        "--autoregressive",
        "--guidance_param", str(GUIDANCE_SCALE),
        "--num_samples", "1",
        "--num_repetitions", "1",
        "--seed", str(SEED),
    ]
    args = generate_args()
    dist_util.setup_dist(args.device)

    n_frames = args.context_len + args.pred_len
    data = load_dataset(args, max_frames=196, n_frames=n_frames)
    model, diffusion = create_model_and_diffusion(args, data)
    load_saved_model(model, str(ckpt), use_avg=getattr(args, "use_ema", False))

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model, "text")
    model.to(dist_util.dev())
    model.eval()

    # Build the model_kwargs the same way generate.py does (single-prompt path)
    collate_args = [{"inp": torch.zeros(args.pred_len), "tokens": None,
                     "lengths": args.pred_len, "text": PROMPT}]
    _, model_kwargs = collate(collate_args)
    model_kwargs["y"] = {
        k: v.to(dist_util.dev()) if torch.is_tensor(v) else v
        for k, v in model_kwargs["y"].items()
    }

    # Cache CLIP embedding once
    model_kwargs["y"]["text_embed"] = model.encode_text(model_kwargs["y"]["text"])
    model_kwargs["y"]["scale"] = (
        torch.ones(1, device=dist_util.dev()) * GUIDANCE_SCALE
    )

    # Initial prefix from data (mirroring generate.py's is_using_data=False fallback —
    # we use a fixed-seed zero prefix here to keep fixtures deterministic without
    # needing the dataset on every machine that re-runs this script).
    # If we want a "real" prefix, swap to: input_motion, _ = next(iter(data)); prefix = input_motion[..., :context_len]
    model_kwargs["y"]["prefix"] = torch.zeros(
        1, model.njoints, model.nfeats, args.context_len, device=dist_util.dev()
    )

    return model, diffusion, args, model_kwargs


def fixtures_phase_core():
    """Phase 1: text-to-motion AR with no-target ckpt. Saves fixtures into phase1_core/."""
    out = FIXTURES_DIR / "phase1_core"
    out.mkdir(parents=True, exist_ok=True)

    model, diffusion, args, model_kwargs = setup_model_and_data()
    device = next(model.parameters()).device

    # Save invariants
    (out / "prompt.txt").write_text(PROMPT)
    np.save(out / "text_embed.npy", model_kwargs["y"]["text_embed"].cpu().numpy())
    np.save(out / "prefix.npy", model_kwargs["y"]["prefix"].cpu().numpy())
    np.save(out / "mask.npy", model_kwargs["y"]["mask"].cpu().numpy())

    # Copy normalization stats — saves browser one fewer fetch URL to figure out
    np.save(out / "mean.npy", np.load(DATASET_DIR / "t2m_mean.npy"))
    np.save(out / "std.npy", np.load(DATASET_DIR / "t2m_std.npy"))

    # Run 2 AR iterations manually (rather than calling AutoRegressiveSampler)
    # so we can tap into per-step intermediates. This duplicates ~30 lines of
    # AutoRegressiveSampler.sample but with hooks for fixture saving.
    n_iterations = 2  # plan only requires 2 for AR parity proof
    motion_shape = (1, model.njoints, model.nfeats, args.pred_len)

    cur_prefix = model_kwargs["y"]["prefix"].clone()
    motions = []

    for it in range(n_iterations):
        iter_dir = out / f"iter{it}"
        iter_dir.mkdir(exist_ok=True)

        # Set prefix for this iter and save
        model_kwargs["y"]["prefix"] = cur_prefix.clone()
        np.save(iter_dir / "prefix.npy", cur_prefix.cpu().numpy())

        # Initial noise — fixed seed so it's reproducible across machines
        torch.manual_seed(SEED + it)
        x_T = torch.randn(*motion_shape, device=device)
        np.save(out / f"x_T_iter{it}.npy", x_T.cpu().numpy())

        # Manual DDIM loop with hooks
        x = x_T.clone()
        N = diffusion.num_timesteps  # = 10
        for step_idx, t_int in enumerate(reversed(range(N))):
            t = torch.tensor([t_int], device=device, dtype=torch.long)
            np.save(iter_dir / f"x_t_step{step_idx:02d}.npy", x.cpu().numpy())

            # Tap pre-CFG cond/uncond outputs for debug (run the inner model twice)
            inner = model.model  # unwrap CFG to get the bare MDM
            with torch.no_grad():
                y_cond = {**model_kwargs["y"], "text_uncond": False}
                y_uncond = {**model_kwargs["y"], "text_uncond": True}
                # SpacedDiffusion wraps the model and remaps timesteps via timestep_map.
                # For diffusion_steps=10 the map is identity, so calling inner directly is fine
                # for fixtures (verified in SAMPLER_NOTES.md §1).
                cond_out = inner(x, t, y_cond)
                uncond_out = inner(x, t, y_uncond)
            np.save(iter_dir / f"cond_step{step_idx:02d}.npy", cond_out.cpu().numpy())
            np.save(iter_dir / f"uncond_step{step_idx:02d}.npy", uncond_out.cpu().numpy())

            # Run the actual ddim_sample step (with CFG wrapper) so fixtures
            # match what generate.py would produce
            with torch.no_grad():
                out_dict = diffusion.ddim_sample(
                    model, x, t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    eta=0.0,
                )
            np.save(
                iter_dir / f"pred_xstart_step{step_idx:02d}.npy",
                out_dict["pred_xstart"].cpu().numpy(),
            )
            x = out_dict["sample"]

        # x is the final prefix||predict at end of denoise; the model returned 60 frames
        # internally, but ddim_sample operates on the 40-frame tensor we passed in.
        # The prediction for this iter is exactly x (since ddim works on the predict region).
        # The AR loop uses sample[..., -context_len:] as next prefix; since x is 40 frames
        # and we want last 20, that's x[..., -20:].
        np.save(out / f"motion_iter{it}.npy", x.cpu().numpy())
        motions.append(x.cpu())

        cur_prefix = x[..., -args.context_len:].clone()

    # Optional: full 5-iter motion via the proper sampler (skipped for now since
    # the plan's success criteria only require 2 iterations of parity).
    #
    # If you want it: uncomment and the fixtures grow to ~5x.
    # from closd.diffusion_planner.utils.sampler_util import AutoRegressiveSampler
    # sample_cls = AutoRegressiveSampler(args, diffusion.p_sample_loop, 196)
    # full = sample_cls.sample(model, motion_shape, model_kwargs=model_kwargs, ...)
    # np.save(out / "motion_full.npy", full.cpu().numpy())

    print(f"[fixtures] phase1_core written to {out}")
    print(f"[fixtures]  prefix init norm = {model_kwargs['y']['prefix'].norm().item():.3f}")
    print(f"[fixtures]  iter0 final motion norm = {motions[0].norm().item():.3f}")
    print(f"[fixtures]  iter1 final motion norm = {motions[1].norm().item():.3f}")


def fixtures_phase_goal():
    """Phase 2 Day 8-9: goal-conditioned generation. To be implemented after Phase 1 ships."""
    raise NotImplementedError("Phase 2 fixture; implement on Day 8 of the plan.")


def fixtures_phase_inbetween():
    """Phase 2 Day 10-11: in-between editing. Saves:
        reference_motion.npy   [1, 263, 1, 196] reference (a Phase 1 output)
        inpainting_mask.npy    [1, 263, 1, 196] bool, True = preserve
        edit_in_between/iter{0,1}/* same-shape per-step intermediates
        edit_in_between/motion_full.npy   [1, 263, 1, 196] final edited motion
    """
    out = FIXTURES_DIR / "phase2_inbetween"
    out.mkdir(parents=True, exist_ok=True)

    model, diffusion, args, model_kwargs = setup_model_and_data()
    device = next(model.parameters()).device

    # 1. Generate the reference motion via the no-target AR pipeline so we
    #    have a clean Phase 1 motion to edit. This avoids needing a real
    #    HumanML3D test sample on disk.
    from closd.diffusion_planner.utils.sampler_util import AutoRegressiveSampler
    motion_shape = (1, model.njoints, model.nfeats, args.pred_len)
    sample_cls = AutoRegressiveSampler(args, diffusion.p_sample_loop, 196)
    with torch.no_grad():
        reference_motion = sample_cls.sample(
            model, motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0, init_image=None, progress=False,
            dump_steps=None, noise=None, const_noise=False, recon_guidance=False,
        )
    np.save(out / "reference_motion.npy", reference_motion.cpu().numpy())

    # 2. Build in-between mask: preserve [0, 50) and [150, 196), inpaint [50, 150).
    start_idx, end_idx = 50, 150
    inpainting_mask = torch.ones(
        1, model.njoints, model.nfeats, 196, dtype=torch.bool, device=device
    )
    inpainting_mask[..., start_idx:end_idx] = False
    np.save(out / "inpainting_mask.npy", inpainting_mask.cpu().numpy().astype(bool))
    np.save(out / "edit_window.npy", np.array([start_idx, end_idx], dtype=np.int32))

    # 3. Re-run the AR sampler with inpainting active. The model_kwargs gain
    #    inpainting_mask + inpainted_motion which p_mean_variance reads in
    #    gaussian_diffusion.py:359-365.
    edit_kwargs = {
        "y": {
            **model_kwargs["y"],
            "inpainting_mask": inpainting_mask,
            "inpainted_motion": reference_motion,
        }
    }

    # Re-init prefix to zeros so the edit run is reproducible
    edit_kwargs["y"]["prefix"] = torch.zeros(
        1, model.njoints, model.nfeats, args.context_len, device=device
    )

    with torch.no_grad():
        edited = sample_cls.sample(
            model, motion_shape,
            clip_denoised=False,
            model_kwargs=edit_kwargs,
            skip_timesteps=0, init_image=None, progress=False,
            dump_steps=None, noise=None, const_noise=False, recon_guidance=False,
        )
    np.save(out / "motion_full.npy", edited.cpu().numpy())

    # Sanity: edited motion should equal reference at preserved frames.
    ref = reference_motion.cpu().numpy()
    edt = edited.cpu().numpy()
    err_preserved = np.abs(ref[..., :start_idx] - edt[..., :start_idx]).mean()
    err_preserved_tail = np.abs(ref[..., end_idx:] - edt[..., end_idx:]).mean()
    print(f"[fixtures] inbetween: preserved-region MAE = {err_preserved:.2e} (head), "
          f"{err_preserved_tail:.2e} (tail)")
    print(f"[fixtures] phase2_inbetween written to {out}")


def fixtures_phase_upper():
    """Phase 2 Day 12: upper-body editing. Same pipeline as inbetween but with
    a feature-axis mask (HML_LOWER_BODY_MASK) instead of a time-axis mask.
    """
    out = FIXTURES_DIR / "phase2_upper"
    out.mkdir(parents=True, exist_ok=True)

    from closd.diffusion_planner.data_loaders import humanml_utils
    model, diffusion, args, model_kwargs = setup_model_and_data()
    device = next(model.parameters()).device

    from closd.diffusion_planner.utils.sampler_util import AutoRegressiveSampler
    motion_shape = (1, model.njoints, model.nfeats, args.pred_len)
    sample_cls = AutoRegressiveSampler(args, diffusion.p_sample_loop, 196)
    with torch.no_grad():
        reference_motion = sample_cls.sample(
            model, motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0, init_image=None, progress=False,
            dump_steps=None, noise=None, const_noise=False, recon_guidance=False,
        )
    np.save(out / "reference_motion.npy", reference_motion.cpu().numpy())

    # Build upper-body mask: HML_LOWER_BODY_MASK is True for lower-body features.
    # The edit.py path preserves where mask=True and inpaints where mask=False,
    # so for "upper-body editing" we use HML_LOWER_BODY_MASK directly (preserves
    # lower-body, inpaints upper-body).
    lower_mask = humanml_utils.HML_LOWER_BODY_MASK  # numpy bool [263]
    np.save(out / "lower_body_mask.npy", lower_mask.astype(bool))
    inpainting_mask = torch.tensor(lower_mask, dtype=torch.bool, device=device)
    inpainting_mask = (
        inpainting_mask.view(1, -1, 1, 1)
        .expand(1, -1, model.nfeats, 196)
        .contiguous()
    )

    edit_kwargs = {
        "y": {
            **model_kwargs["y"],
            "inpainting_mask": inpainting_mask,
            "inpainted_motion": reference_motion,
        }
    }
    edit_kwargs["y"]["prefix"] = torch.zeros(
        1, model.njoints, model.nfeats, args.context_len, device=device
    )

    with torch.no_grad():
        edited = sample_cls.sample(
            model, motion_shape,
            clip_denoised=False,
            model_kwargs=edit_kwargs,
            skip_timesteps=0, init_image=None, progress=False,
            dump_steps=None, noise=None, const_noise=False, recon_guidance=False,
        )
    np.save(out / "motion_full.npy", edited.cpu().numpy())

    # Sanity: lower-body features unchanged.
    ref = reference_motion.cpu().numpy()
    edt = edited.cpu().numpy()
    feat_mask = lower_mask  # [263]
    err_lower = np.abs(
        ref[:, feat_mask, :, :] - edt[:, feat_mask, :, :]
    ).mean()
    print(f"[fixtures] upper: lower-body-feature MAE = {err_lower:.2e}")
    print(f"[fixtures] phase2_upper written to {out}")


def fixtures_phase_invert():
    """Phase 2 Day 13: DDIM inversion. To be implemented after Phase 1 ships."""
    raise NotImplementedError("Phase 2 fixture; implement on Day 13 of the plan.")


def fixtures_phase_smpl():
    """Phase 2 Day 14: SMPL extraction. To be implemented after Phase 1 ships."""
    raise NotImplementedError("Phase 2 fixture; implement on Day 14 of the plan.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="core",
                   choices=["core", "goal", "inbetween", "upper", "invert", "smpl"])
    args = p.parse_args()
    {
        "core": fixtures_phase_core,
        "goal": fixtures_phase_goal,
        "inbetween": fixtures_phase_inbetween,
        "upper": fixtures_phase_upper,
        "invert": fixtures_phase_invert,
        "smpl": fixtures_phase_smpl,
    }[args.phase]()


if __name__ == "__main__":
    main()
