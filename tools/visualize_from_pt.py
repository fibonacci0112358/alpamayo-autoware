#!/usr/bin/env python3
"""Visualize .pt dataset samples with GT and model predictions (base vs SFT).

Loads .pt files containing keys like `image_frames`, `camera_indices`,
`ego_history_xyz`, `ego_history_rot`, `ego_future_xyz` and produces PNG
visualizations (camera grid + BEV) overlaying GT, base-model and SFT-model
predicted trajectories. Optionally produces MP4 videos.

Example:
  python3 tools/visualize_from_pt.py --pt-dir /data/pt_samples --max-samples 10 \
      --base-checkpoint nvidia/Alpamayo-1.5-10B --sft-checkpoint ./merged_final \
      --output-dir outputs/pt_vis --video
"""
from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import hydra.utils as hyu

from alpamayo1_5 import helper
from alpamayo1_5 import viz_utils
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

FRONT_CAMERA_INDEX = 1
MODEL_DTYPE = torch.bfloat16


def align_tokenizer_and_embeddings(model: Alpamayo1_5, label: str) -> None:
    """Ensure tokenizer vocab size matches VLM embedding matrix size.

    When loading with ``ignore_mismatched_sizes=True``, embedding weights can be
    skipped, leaving the model with a smaller embedding table than tokenizer ids.
    That causes CUDA gather index-out-of-bounds during generation.
    """
    tokenizer = getattr(model, "tokenizer", None)
    vlm = getattr(model, "vlm", None)
    if tokenizer is None or vlm is None or not hasattr(vlm, "get_input_embeddings"):
        return

    tokenizer_vocab_size = len(tokenizer)
    embedding_layer = vlm.get_input_embeddings()
    model_vocab_size = int(embedding_layer.num_embeddings)

    if tokenizer_vocab_size == model_vocab_size:
        return

    print(
        f"Warning: {label} tokenizer/embedding mismatch "
        f"({tokenizer_vocab_size} vs {model_vocab_size}); resizing embeddings."
    )
    vlm.resize_token_embeddings(tokenizer_vocab_size)


def repair_trajectory_fusion_attributes(model: Alpamayo1_5, label: str) -> None:
    """Fill in trajectory-fusion attributes that may be missing after checkpoint loading."""
    config = getattr(model, "config", None)
    traj_tokenizer = getattr(model, "traj_tokenizer", None)
    if traj_tokenizer is None and config is not None and getattr(config, "traj_tokenizer_cfg", None) is not None:
        print(f"Warning: {label} checkpoint is missing traj_tokenizer; instantiating from config.")
        model.traj_tokenizer = hyu.instantiate(config.traj_tokenizer_cfg, load_weights=False)
        traj_tokenizer = model.traj_tokenizer

    if getattr(model, "hist_traj_tokenizer", None) is None and traj_tokenizer is not None:
        if config is not None and getattr(config, "hist_traj_tokenizer_cfg", None) is not None:
            print(f"Warning: {label} checkpoint is missing hist_traj_tokenizer; instantiating from config.")
            model.hist_traj_tokenizer = hyu.instantiate(config.hist_traj_tokenizer_cfg)
        else:
            print(f"Warning: {label} checkpoint is missing hist_traj_tokenizer; reusing traj_tokenizer.")
            model.hist_traj_tokenizer = traj_tokenizer

    traj_token_start_idx = getattr(config, "traj_token_start_idx", None) if config is not None else None
    if getattr(model, "hist_token_start_idx", None) is None and traj_token_start_idx is not None:
        model.hist_token_start_idx = traj_token_start_idx
    if getattr(model, "future_token_start_idx", None) is None and traj_token_start_idx is not None:
        model.future_token_start_idx = traj_token_start_idx


def crop_empty_borders(image: np.ndarray) -> np.ndarray:
    """Trim fully black borders from a camera grid image.

    The camera grid uses zero-filled cells for unused layout slots. When only a
    subset of cells are populated, the default image contains large black margins
    that make the plot look broken. This keeps actual image content and removes
    only empty borders.
    """
    mask = np.any(image > 0, axis=-1)
    if not np.any(mask):
        return image

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1

    pad = 8
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(image.shape[0], r1 + pad)
    c1 = min(image.shape[1], c1 + pad)
    return image[r0:r1, c0:c1]


def rotate_bev_90ccw(trajs: np.ndarray) -> np.ndarray:
    """Rotate BEV coordinates 90 degrees counter-clockwise.

    This maps (x, y) -> (-y, x) so the displayed +x direction points upward.
    """
    return np.stack((-trajs[..., 1], trajs[..., 0]), axis=-1)


def find_pt_files(pt: Path | None, pt_dir: Path | None) -> list[Path]:
    if pt is not None:
        return [pt]
    if pt_dir is not None:
        return sorted([p for p in pt_dir.rglob("*.pt") if p.is_file()])
    return []


def load_model(checkpoint: str, label: str, device: str = "cuda", base_checkpoint: str | None = None) -> Alpamayo1_5:
    """Load a model, falling back to mismatched-size tolerant loading.

    If the checkpoint path lacks a `config.json` (OSError during config load),
    and `base_checkpoint` is provided, attempt to load the base model's
    config and instantiate the model using that config so SFT-only (weights-
    only) checkpoints can be loaded.
    """
    print(f"Loading {label} model: {checkpoint}...")
    # Prefer to instantiate with the checkpoint's own config if available —
    # this ensures we use the exact architecture the checkpoint was saved with.
    cfg = None
    try:
        cfg = Alpamayo1_5.config_class.from_pretrained(checkpoint)
    except Exception:
        # keep cfg None and fall back below
        cfg = None

    if cfg is not None:
        # If the checkpoint's config references a local VLM path that doesn't
        # exist on this machine, prefer the base checkpoint's VLM path when
        # available. This prevents Qwen3VLConfig.from_pretrained from failing
        # when the SFT config points to a training-only artifact path.
        vlm_ref = getattr(cfg, "vlm_name_or_path", None)
        if isinstance(vlm_ref, str) and not os.path.exists(vlm_ref) and base_checkpoint is not None:
            try:
                base_cfg = Alpamayo1_5.config_class.from_pretrained(base_checkpoint)
                base_vlm = getattr(base_cfg, "vlm_name_or_path", None)
                if base_vlm is not None:
                    print(f"Note: replacing missing vlm_name_or_path '{vlm_ref}' with base checkpoint vlm '{base_vlm}'.")
                    cfg.vlm_name_or_path = base_vlm
            except Exception:
                # If base config can't be loaded, continue and let the normal
                # instantiation path surface the error.
                pass
        try:
            model = Alpamayo1_5.from_pretrained(checkpoint, config=cfg, dtype=MODEL_DTYPE)
        except RuntimeError as exc:
            if "size mismatch" not in str(exc):
                raise
            print(
                f"Warning: strict loading failed for {label} with checkpoint config due to a size mismatch; retrying with ignore_mismatched_sizes=True."
            )
            model = Alpamayo1_5.from_pretrained(
                checkpoint,
                config=cfg,
                dtype=MODEL_DTYPE,
                ignore_mismatched_sizes=True,
            )
        except Exception:
            # If instantiation with checkpoint config fails, fall back to original flow
            cfg = None

    if cfg is None:
        try:
            model = Alpamayo1_5.from_pretrained(checkpoint, dtype=MODEL_DTYPE)
        except RuntimeError as exc:
            if "size mismatch" not in str(exc):
                raise
            print(
                f"Warning: strict loading failed for {label} due to a size mismatch; retrying with ignore_mismatched_sizes=True."
            )
            model = Alpamayo1_5.from_pretrained(
                checkpoint,
                dtype=MODEL_DTYPE,
                ignore_mismatched_sizes=True,
            )
        except OSError as exc:
            # Missing config.json in the checkpoint dir — try to reuse the base
            # checkpoint's config if available so weight-only SFT folders can be
            # applied.
            if base_checkpoint is None:
                raise

            print(f"Warning: failed to load config for {label}; attempting to use base config from {base_checkpoint}.")
            base_cfg = Alpamayo1_5.config_class.from_pretrained(base_checkpoint)
            try:
                model = Alpamayo1_5.from_pretrained(
                    checkpoint,
                    config=base_cfg,
                    dtype=MODEL_DTYPE,
                    ignore_mismatched_sizes=True,
                )
            except Exception:
                # If instantiation still fails, re-raise the original error to surface it.
                raise

    align_tokenizer_and_embeddings(model, label)
    repair_trajectory_fusion_attributes(model, label)
    return model.to(device)


def unload_model(model: Alpamayo1_5 | None) -> None:
    """Release GPU memory held by a model."""
    if model is None:
        return
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_pt_sample(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # support wrapped samples under 'data' or similar
        if "data" in obj and isinstance(obj["data"], dict):
            data = obj["data"]
        else:
            data = obj
    else:
        raise ValueError(f"Unexpected .pt content: {type(obj)} in {path}")

    required = ["image_frames", "ego_history_xyz", "ego_history_rot"]
    for k in required:
        if k not in data:
            raise KeyError(f"Required key '{k}' not found in {path}")
    return data


def normalize_sample_shapes(data: dict) -> dict:
    """Normalize sample tensors to the batched shapes expected by Alpamayo1_5."""
    normalized = dict(data)

    image_frames = normalized["image_frames"]
    if image_frames.ndim == 4:
        # Single-camera temporal stack -> add camera axis.
        normalized["image_frames"] = image_frames.unsqueeze(0)
    elif image_frames.ndim != 5:
        raise ValueError(f"Unexpected image_frames shape: {tuple(image_frames.shape)}")

    for key in ("ego_history_xyz", "ego_history_rot", "ego_future_xyz", "ego_future_rot"):
        value = normalized.get(key)
        if value is None:
            continue
        if key.endswith("_rot") and value.ndim == 4:
            # .pt samples store rotations as [B, T, 3, 3]; add traj-group dim.
            normalized[key] = value.unsqueeze(1)
        elif key.endswith("_xyz") and value.ndim == 3:
            normalized[key] = value.unsqueeze(1)
        elif value.ndim not in (4, 5):
            raise ValueError(f"Unexpected {key} shape: {tuple(value.shape)}")

    return normalized


def resize_frames_for_inference(image_frames: torch.Tensor, max_long_side: int = 512) -> torch.Tensor:
    """Downsample frames for vision inference while preserving aspect ratio.

    The .pt samples in this repo can contain full-resolution camera frames.
    The Qwen3-VL vision stack expects much smaller inputs, so we cap the long
    side to keep the pixel count in a safe range.
    """
    if image_frames.ndim != 5:
        raise ValueError(f"Expected image_frames with shape [N, T, C, H, W], got {tuple(image_frames.shape)}")

    n_cameras, num_frames, channels, height, width = image_frames.shape
    long_side = max(height, width)
    if long_side <= max_long_side:
        return image_frames

    scale = max_long_side / float(long_side)
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))

    flat = image_frames.reshape(n_cameras * num_frames, channels, height, width).float()
    flat = F.interpolate(flat, size=(target_height, target_width), mode="bilinear", align_corners=False)
    flat = flat.round().clamp(0, 255).to(torch.uint8)
    return flat.reshape(n_cameras, num_frames, channels, target_height, target_width).contiguous()


def select_front_camera_for_inference(data: dict) -> dict:
    """Keep only the front camera for inference to reduce VLM load."""
    selected = dict(data)
    image_frames = selected["image_frames"]
    camera_indices = selected.get("camera_indices")

    if image_frames.ndim != 5:
        raise ValueError(f"Expected batched image_frames, got {tuple(image_frames.shape)}")

    if camera_indices is None:
        front_frames = image_frames[:1]
        front_indices = torch.tensor([FRONT_CAMERA_INDEX], dtype=torch.int64)
    else:
        matches = (camera_indices == FRONT_CAMERA_INDEX).nonzero(as_tuple=False).flatten()
        if matches.numel() > 0:
            front_frames = image_frames[matches[:1]]
            front_indices = camera_indices[matches[:1]]
        else:
            front_frames = image_frames[:1]
            front_indices = torch.tensor([FRONT_CAMERA_INDEX], dtype=torch.int64)

    selected["image_frames"] = front_frames.contiguous()
    selected["camera_indices"] = front_indices.contiguous()
    return selected


def run_model_inference(model: Alpamayo1_5, processor, data: dict, args: argparse.Namespace, device: str = "cuda"):
    data = normalize_sample_shapes(data)
    data = select_front_camera_for_inference(data)
    inference_frames = resize_frames_for_inference(data["image_frames"], max_long_side=320)
    frames = inference_frames.flatten(0, 1)
    camera_indices = data.get("camera_indices", None)

    messages = helper.create_message(
        frames,
        camera_indices=camera_indices,
        num_frames_per_camera=4,
        nav_text=args.nav_text,
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
        device=device,
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs["tokenized_data"] = helper.to_device(model_inputs["tokenized_data"], device)
    model_inputs["ego_history_xyz"] = helper.to_device(
        model_inputs["ego_history_xyz"], device, dtype=torch.bfloat16
    )
    model_inputs["ego_history_rot"] = helper.to_device(
        model_inputs["ego_history_rot"], device, dtype=torch.bfloat16
    )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            num_traj_sets=1,
            max_generation_length=args.max_generation_length,
            return_extra=True,
        )

    return pred_xyz.detach().cpu(), extra


def plot_and_save(
    output_dir: Path,
    sample_name: str,
    data: dict,
    base_pred,
    sft_pred,
    extra_base,
    extra_sft,
    args: argparse.Namespace,
    show_base: bool = True,
):
    # camera grid
    image_frames = data["image_frames"]
    cam_idx = data.get("camera_indices", None)
    cam_grid = crop_empty_borders(viz_utils.make_camera_grid(image_frames, camera_indices=cam_idx))

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    fig.suptitle(sample_name, fontsize=14)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    ax_cam = fig.add_subplot(gs[0, 0])
    ax_bev = fig.add_subplot(gs[0, 1])

    ax_cam.imshow(np.clip(cam_grid / 255.0, 0, 1))
    ax_cam.axis("off")

    sft_xy = rotate_bev_90ccw(viz_utils.get_trajectories_xy(sft_pred))

    # Convert predicted trajectories from model frame (relative displacements)
    # to world coordinates by adding the last history position so they can be
    # directly compared with ground-truth `ego_future_xyz` which is in the
    # same world frame.
    hist = data.get("ego_history_xyz", None)
    offset = None
    if hist is not None:
        try:
            if hist.dim() == 4:
                offset = hist[0, 0, -1, :2].cpu().numpy()
            else:
                offset = hist[0, -1, :2].cpu().numpy()
        except Exception:
            offset = None

    if show_base and base_pred is not None:
        base_xy = rotate_bev_90ccw(viz_utils.get_trajectories_xy(base_pred))
        if offset is not None:
            base_xy = base_xy + offset
        viz_utils.plot_condition(ax_bev, base_xy, color="tab:blue", label="Base")

    # Keep the SFT display mirrored after the 90-degree rotation.
    sft_xy = -sft_xy
    if offset is not None:
        sft_xy = sft_xy + offset
    viz_utils.plot_condition(ax_bev, sft_xy, color="tab:green", label="SFT")

    gt = data.get("ego_future_xyz", None)
    if gt is not None:
        if gt.dim() == 4:
            g = gt[0, 0, :, :2].numpy()
        else:
            g = gt[0, :, :2].numpy()
        g = rotate_bev_90ccw(g)
        ax_bev.plot(g[:, 0], g[:, 1], color="black", linewidth=2.5, label="GT")

    ax_bev.set_xlabel("-y (m)")
    ax_bev.set_ylabel("x (m)")
    ax_bev.set_aspect("equal")
    ax_bev.set_xlim(*viz_utils.BEV_X_LIM)
    ax_bev.set_ylim(*viz_utils.BEV_Y_LIM)
    viz_utils._enforce_readable_axes(ax_bev)
    ax_bev.legend(loc="upper left", fontsize=8)
    ax_bev.grid(True, alpha=0.3)

    # Add CoT text if available (shortened)
    cot_text = None
    if isinstance(extra_base, dict) and "cot" in extra_base:
        try:
            cot_text = extra_base["cot"][0, 0, 0]
        except Exception:
            cot_text = None
    if cot_text is not None:
        if isinstance(cot_text, bytes):
            cot_text = cot_text.decode("utf-8", errors="ignore")
        cot_text = str(cot_text).strip()
        if cot_text:
            # place small textbox
            fig.text(0.05, 0.965, "CoT: " + (cot_text[:300] + "…" if len(cot_text) > 300 else cot_text), fontsize=7)

    out_png = output_dir / f"{sample_name}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def parse_device_list(device_list: str | None, default_count: int = 1) -> list[str]:
    if device_list is None or not device_list.strip():
        return [f"cuda:{idx}" for idx in range(default_count)]
    return [device.strip() for device in device_list.split(",") if device.strip()]


def process_sample_batch(
    pt_files: list[Path],
    args: argparse.Namespace,
    base_device: str | None,
    sft_device: str,
) -> list[Path]:
    processed = 0
    pngs: list[Path] = []

    base_model = None
    base_processor = None
    if base_device is not None:
        print(f"Loading base model on {base_device}...")
        base_model = load_model(args.base_checkpoint, "base", device=base_device)
        base_model.eval()
        base_processor = helper.get_processor(base_model.tokenizer)

    print(f"Loading SFT model on {sft_device}...")
    sft_model = load_model(args.sft_checkpoint, "SFT", device=sft_device, base_checkpoint=args.base_checkpoint)
    sft_model.eval()
    vlm_name = getattr(getattr(sft_model, "config", None), "vlm_name_or_path", None)
    if vlm_name:
        try:
            from transformers import AutoProcessor

            sft_processor = AutoProcessor.from_pretrained(vlm_name)
            sft_processor.tokenizer = sft_model.tokenizer
        except Exception:
            sft_processor = helper.get_processor(sft_model.tokenizer)
    else:
        sft_processor = helper.get_processor(sft_model.tokenizer)

    for p in pt_files:
        try:
            data = normalize_sample_shapes(load_pt_sample(p))
        except Exception as exc:
            print(f"Skipping {p}: {exc}")
            continue

        sample_name = p.stem
        print(f"Processing {p} -> {sample_name}")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        base_pred = None
        extra_base = None
        if base_model is not None:
            base_pred, extra_base = run_model_inference(base_model, base_processor, data, args, device=base_device)

        sft_pred, extra_sft = run_model_inference(sft_model, sft_processor, data, args, device=sft_device)

        out_png = plot_and_save(
            args.output_dir,
            sample_name,
            data,
            base_pred,
            sft_pred,
            extra_base,
            extra_sft,
            args,
            show_base=base_model is not None,
        )
        pngs.append(out_png)

        if args.video:
            try:
                import imageio

                video_path = args.output_dir / f"{sample_name}.mp4"
                image = imageio.imread(str(out_png))
                writer = imageio.get_writer(str(video_path), fps=1)
                writer.append_data(image)
                writer.close()
                print(f"Saved video: {video_path}")
            except Exception:
                print("Video generation failed; ensure 'imageio' is installed.")

        processed += 1

    unload_model(base_model)
    unload_model(sft_model)
    print(f"Completed {processed} samples. Outputs in {args.output_dir}")
    return pngs


def chunk_paths(paths: list[Path], num_chunks: int) -> list[list[Path]]:
    if num_chunks <= 1:
        return [paths]
    chunks: list[list[Path]] = [[] for _ in range(num_chunks)]
    for index, path in enumerate(paths):
        chunks[index % num_chunks].append(path)
    return [chunk for chunk in chunks if chunk]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", type=Path, default=None, help="Single .pt sample file")
    parser.add_argument("--pt-dir", type=Path, default=None, help="Directory containing .pt samples")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--base-checkpoint", type=str, default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--sft-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pt_visualizations"))
    parser.add_argument("--video", action="store_true", help="Combine PNGs into MP4 per sample")
    parser.add_argument("--num-traj-samples", type=int, default=4)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--nav-text", type=str, default=None)
    parser.add_argument("--skip-base", action="store_true", default=True, help="Skip base model inference and run only SFT (faster)")
    parser.add_argument("--base-device", type=str, default="cuda:0", help="Device for base model (e.g. cuda:0)")
    parser.add_argument("--sft-devices", type=str, default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7", help="Comma-separated devices for SFT workers")

    args = parser.parse_args()

    # Qwen3-VL's 3D convolution path can trip cuDNN internal errors on some setups.
    # Disable cuDNN so PyTorch falls back to its native implementation for this script.
    torch.backends.cudnn.enabled = False

    pt_files = find_pt_files(args.pt, args.pt_dir)
    if not pt_files:
        print("No .pt files found. Provide --pt or --pt-dir.")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    pngs = []
    if not args.sft_checkpoint:
        print("No --sft-checkpoint provided; SFT inference is required in the new multi-GPU mode.")
        sys.exit(1)

    if not args.skip_base:
        print("Base output is disabled in the current mode. Re-run with --skip-base to match the requested SFT-only visualization.")
        sys.exit(1)

    sft_devices = parse_device_list(args.sft_devices, default_count=8)
    sft_devices = sft_devices[: min(len(sft_devices), len(pt_files))]
    print(f"Using SFT devices: {', '.join(sft_devices)}")

    chunks = chunk_paths(pt_files[: args.max_samples], len(sft_devices))
    worker_args = []
    for index, chunk in enumerate(chunks):
        worker_args.append((chunk, args, None, sft_devices[index]))

    if len(worker_args) == 1:
        pngs = process_sample_batch(*worker_args[0])
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(worker_args), mp_context=ctx) as executor:
            futures = [executor.submit(process_sample_batch, *worker_arg) for worker_arg in worker_args]
            for future in as_completed(futures):
                pngs.extend(future.result())

    processed = len(pngs)
    print(f"Completed {processed} samples. Outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
