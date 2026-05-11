# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Alpamayo inference from regular image files (jpg/png/webp).

This script is a dataset-free alternative to test_inference.py.
It builds the same model input structure from local image files and
uses zero ego-history as a placeholder trajectory context.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CAMERA_INDICES = (0, 1, 2, 6)
FRONT_CAMERA_INDEX = 1
FULLHD_WIDTH = 1920
FULLHD_HEIGHT = 1080


def _load_image_tensor(path: Path) -> torch.Tensor:
    """Load an image as uint8 CHW tensor."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        data = torch.from_numpy(np.array(rgb, dtype="uint8"))
    return data.permute(2, 0, 1).contiguous()


def _collect_images(image_dir: Path, recursive: bool) -> list[Path]:
    """Collect and sort image files from a directory."""
    if recursive:
        candidates = [p for p in image_dir.rglob("*") if p.is_file()]
    else:
        candidates = [p for p in image_dir.iterdir() if p.is_file()]

    paths = [p for p in candidates if p.suffix.lower() in SUPPORTED_EXTS]
    paths.sort()
    return paths


def _apply_fps_filter(image_paths: list[Path], input_fps: int, output_fps: int) -> list[Path]:
    """Filter image paths based on input and output fps.
    
    Args:
        image_paths: List of all image paths (assumed to be at input_fps).
        input_fps: Frame rate of input images (e.g., 30).
        output_fps: Desired output frame rate (e.g., 10).
    
    Returns:
        Filtered list of image paths at output_fps.
    """
    if output_fps > input_fps:
        raise ValueError(
            f"Output FPS ({output_fps}) cannot be greater than input FPS ({input_fps})"
        )
    
    if output_fps == input_fps:
        return image_paths
    
    skip_rate = input_fps // output_fps
    filtered_paths = image_paths[::skip_rate]
    print(f"FPS filtering: {input_fps}fps -> {output_fps}fps (skip_rate={skip_rate})")
    print(f"Images: {len(image_paths)} -> {len(filtered_paths)}")
    return filtered_paths


def _build_dummy_history(num_history_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create zero translation and identity rotation history."""
    ego_history_xyz = torch.zeros((1, 1, num_history_steps, 3), dtype=torch.float32)

    eye = torch.eye(3, dtype=torch.float32)
    ego_history_rot = eye.view(1, 1, 1, 3, 3).repeat(1, 1, num_history_steps, 1, 1)
    return ego_history_xyz, ego_history_rot


def _build_single_camera_frames(
    temporal_frames: torch.Tensor,
    num_frames_per_camera: int,
    active_camera_index: int = FRONT_CAMERA_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build frames for multiple cameras, with one camera having temporal data and others black."""
    if temporal_frames.shape[0] != num_frames_per_camera:
        raise ValueError(
            f"Expected {num_frames_per_camera} temporal frames, got {temporal_frames.shape[0]}"
        )
    
    if temporal_frames.ndim != 4 or temporal_frames.shape[1] != 3:
        raise ValueError(
            f"Expected frames shape (T, C, H, W) with C=3, got {tuple(temporal_frames.shape)}"
        )
    
    black_frames = torch.zeros_like(temporal_frames)
    
    camera_frames = []
    for camera_index in CAMERA_INDICES:
        if camera_index == active_camera_index:
            camera_frames.append(temporal_frames)
        else:
            camera_frames.append(black_frames)
    
    frames = torch.cat(camera_frames, dim=0).contiguous()
    camera_indices = torch.tensor(CAMERA_INDICES, dtype=torch.int64)
    return frames, camera_indices


def _run_inference(
    model: Alpamayo1_5,
    processor,
    frames: torch.Tensor,
    camera_indices: torch.Tensor,
    ego_history_xyz: torch.Tensor,
    ego_history_rot: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict | None]:
    """Run one inference pass for the provided frames."""
    messages = helper.create_message(
        frames,
        camera_indices=camera_indices,
        num_frames_per_camera=args.num_frames_per_camera,
        nav_text=args.nav_text,
    )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        if args.nav_text:
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav(
                data=model_inputs,
                top_p=args.top_p,
                temperature=args.temperature,
                num_traj_samples=args.num_traj_samples,
                num_traj_sets=1,
                max_generation_length=args.max_generation_length,
                return_extra=True,
            )
        else:
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=args.top_p,
                temperature=args.temperature,
                num_traj_samples=args.num_traj_samples,
                num_traj_sets=1,
                max_generation_length=args.max_generation_length,
                return_extra=True,
            )

    traj = pred_xyz.detach().cpu()[0, 0, 0]
    return traj, extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory containing input images (jpg/png/webp/bmp).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search image files recursively under --image-dir.",
    )
    parser.add_argument(
        "--num-frames-per-camera",
        type=int,
        default=4,
        help="Number of temporal frames per camera.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="nvidia/Alpamayo-1.5-10B",
        help="HF model name or local checkpoint path.",
    )
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/predicted_trajectory.csv"),
        help="Output CSV path for all predicted trajectories.",
    )
    parser.add_argument(
        "--nav-text",
        type=str,
        default=None,
        help="Optional navigation instruction text.",
    )
    parser.add_argument(
        "--num-history-steps",
        type=int,
        default=16,
        help="Length of dummy ego-history trajectory.",
    )
    parser.add_argument(
        "--input-fps",
        type=int,
        default=30,
        help="Input frame rate of images (e.g., 30 for 30fps).",
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=30,
        help="Desired output frame rate (e.g., 10 for 10fps). Must be <= input-fps.",
    )
    parser.add_argument(
        "--dummy-black",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use generated black images instead of loading input images (default: true).",
    )

    args = parser.parse_args()

    if args.num_frames_per_camera <= 0:
        raise ValueError("--num-frames-per-camera must be > 0")

    if args.input_fps <= 0 or args.output_fps <= 0:
        raise ValueError("--input-fps and --output-fps must be > 0")

    if args.output_fps > args.input_fps:
        raise ValueError(
            f"--output-fps ({args.output_fps}) cannot be greater than --input-fps ({args.input_fps})"
        )

    if not args.dummy_black:
        if args.image_dir is None:
            raise ValueError("--image-dir is required unless --dummy-black is set")
        if not args.image_dir.exists() or not args.image_dir.is_dir():
            raise ValueError(f"--image-dir is not a directory: {args.image_dir}")

        image_paths = _collect_images(args.image_dir, args.recursive)
        if not image_paths:
            raise ValueError(f"No supported images found in {args.image_dir}.")
        print(f"Found {len(image_paths)} images in {args.image_dir}.")
    else:
        if args.image_dir is not None:
            print("--dummy-black is enabled; --image-dir is ignored.")
        image_paths = [None]
        print(
            "Using generated black images for all cameras: "
            f"frames_per_camera={args.num_frames_per_camera}, size={FULLHD_WIDTH}x{FULLHD_HEIGHT}"
        )

    print(f"Loading model: {args.model_name}...")
    model = Alpamayo1_5.from_pretrained(args.model_name, dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    ego_history_xyz, ego_history_rot = _build_dummy_history(args.num_history_steps)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.dummy_black:
        # Single dummy inference
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "x", "y", "z", "cot"])
            
            frames = torch.zeros((4, 3, FULLHD_HEIGHT, FULLHD_WIDTH), dtype=torch.uint8)
            frames, camera_indices = _build_single_camera_frames(
                frames,
                args.num_frames_per_camera,
                FRONT_CAMERA_INDEX,
            )
            
            print("[1/1] Processing dummy-black...")
            traj, extra = _run_inference(
                model,
                processor,
                frames,
                camera_indices,
                ego_history_xyz,
                ego_history_rot,
                args,
            )
            
            cot = None
            if isinstance(extra, dict) and "cot" in extra:
                try:
                    cot = extra["cot"][0, 0, 0]
                except Exception:
                    cot = None
            
            if cot is not None:
                if isinstance(cot, bytes):
                    cot = cot.decode("utf-8", errors="ignore")
                cot_text = str(cot).strip()
            else:
                cot_text = ""
            
            for step_index, xyz in enumerate(traj):
                writer.writerow([step_index, float(xyz[0]), float(xyz[1]), float(xyz[2]), cot_text])
            
            print(f"Saved trajectory to {args.output_csv}")
            
            if cot_text:
                print("\nChain-of-Causation:\n", cot_text)
    else:
        # Load all images and process with sliding window
        all_image_paths = image_paths
        all_image_paths = _apply_fps_filter(all_image_paths, args.input_fps, args.output_fps)
        
        all_images = [_load_image_tensor(p) for p in all_image_paths]
        print(f"Loaded {len(all_images)} images.")
        
        if len(all_images) < args.num_frames_per_camera:
            raise ValueError(
                f"Need at least {args.num_frames_per_camera} images, but found {len(all_images)}"
            )
        
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["window_index", "step", "x", "y", "z", "cot"])
            
            num_windows = len(all_images) - args.num_frames_per_camera + 1
            
            for window_idx in range(num_windows):
                # Extract sliding window of frames
                window_frames = torch.stack(
                    all_images[window_idx : window_idx + args.num_frames_per_camera],
                    dim=0
                )
                window_frames, camera_indices = _build_single_camera_frames(
                    window_frames,
                    args.num_frames_per_camera,
                    FRONT_CAMERA_INDEX,
                )
                
                print(f"[{window_idx + 1}/{num_windows}] Processing frames {window_idx}-{window_idx + args.num_frames_per_camera - 1}...")
                traj, extra = _run_inference(
                    model,
                    processor,
                    window_frames,
                    camera_indices,
                    ego_history_xyz,
                    ego_history_rot,
                    args,
                )
                
                cot = None
                if isinstance(extra, dict) and "cot" in extra:
                    try:
                        cot = extra["cot"][0, 0, 0]
                    except Exception:
                        cot = None
                
                if cot is not None:
                    if isinstance(cot, bytes):
                        cot = cot.decode("utf-8", errors="ignore")
                    cot_text = str(cot).strip()
                else:
                    cot_text = ""
                
                for step_index, xyz in enumerate(traj):
                    writer.writerow(
                        [window_idx, step_index, float(xyz[0]), float(xyz[1]), float(xyz[2]), cot_text]
                    )
                
                if window_idx % max(1, num_windows // 10) == 0:
                    print(f"Saved window {window_idx} to {args.output_csv}")
                
                if cot_text and window_idx == 0:  # Print CoT only for first window
                    print("\nChain-of-Causation (first window):\n", cot_text)
            
            print(f"Completed all {num_windows} windows. Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
