#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize Alpamayo trajectory predictions from CSV output.

This script reads the CSV output from infer_from_images.py, loads corresponding
input images, and visualizes the predicted trajectories with CoT text.
Outputs both individual PNG images and a video.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_csv_results(csv_path: Path) -> dict[int, dict[str, Any]]:
    """Load CSV results and group by window_index."""
    results = defaultdict(lambda: {"steps": [], "cot": ""})
    
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            window_idx = int(row["window_index"])
            step = int(row["step"])
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
            cot = row.get("cot", "")
            
            results[window_idx]["steps"].append({"step": step, "x": x, "y": y, "z": z})
            results[window_idx]["cot"] = cot
    
    return results


def get_image_for_window(window_idx: int, image_dir: Path, num_frames: int) -> Path | None:
    """Get the latest image path for a given window."""
    latest_frame_idx = window_idx + num_frames - 1
    image_path = image_dir / f"frame_{latest_frame_idx:06d}.jpg"
    if image_path.exists():
        return image_path
    return None


def _rotate_clockwise(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate plot coordinates 90 degrees clockwise."""
    return ys, -xs


def _nice_number(value: float, round_up: bool) -> float:
    """Return a rounded 'nice' number such as 1, 2, 5, 10, 20, 50."""
    if value <= 0:
        return 1.0

    exponent = math.floor(math.log10(value))
    fraction = value / (10**exponent)

    if round_up:
        if fraction <= 1:
            nice_fraction = 1
        elif fraction <= 2:
            nice_fraction = 2
        elif fraction <= 5:
            nice_fraction = 5
        else:
            nice_fraction = 10
    else:
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 3:
            nice_fraction = 2
        elif fraction < 7:
            nice_fraction = 5
        else:
            nice_fraction = 10

    return nice_fraction * (10**exponent)


def compute_axis_limits(results: dict[int, dict[str, Any]], padding_ratio: float = 0.1) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute shared axis limits from all trajectories using rounded bounds."""
    xs: list[float] = []
    ys: list[float] = []

    for window_data in results.values():
        for step in window_data["steps"]:
            x_rot, y_rot = _rotate_clockwise(np.asarray([step["x"]]), np.asarray([step["y"]]))
            xs.append(float(x_rot[0]))
            ys.append(float(y_rot[0]))

    if not xs or not ys:
        return (-1.0, 1.0), (-1.0, 1.0)

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    x_pad = x_span * padding_ratio
    y_pad = y_span * padding_ratio

    x_step = _nice_number((x_span + 2 * x_pad) / 6.0, round_up=True)
    y_step = _nice_number((y_span + 2 * y_pad) / 6.0, round_up=True)

    x_min = math.floor((x_min - x_pad) / x_step) * x_step
    x_max = math.ceil((x_max + x_pad) / x_step) * x_step
    y_min = math.floor((y_min - y_pad) / y_step) * y_step
    y_max = math.ceil((y_max + y_pad) / y_step) * y_step

    return (x_min, x_max), (y_min, y_max)


def create_trajectory_plot(
    traj_steps: list[dict],
    axis_limits: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    """Create a matplotlib figure showing the trajectory."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    if traj_steps:
        xs = np.asarray([step["x"] for step in traj_steps], dtype=np.float32)
        ys = np.asarray([step["y"] for step in traj_steps], dtype=np.float32)
        plot_xs, plot_ys = _rotate_clockwise(xs, ys)

        ax.plot(plot_xs, plot_ys, "b-o", linewidth=2, markersize=4, label="Predicted trajectory")
        ax.scatter([plot_xs[0]], [plot_ys[0]], c="green", s=100, marker="o", label="Start", zorder=5)
        if len(plot_xs) > 1:
            ax.scatter([plot_xs[-1]], [plot_ys[-1]], c="red", s=100, marker="^", label="End", zorder=5)

        ax.set_xlabel("Rotated X (m)")
        ax.set_ylabel("Rotated Y (m)")
        ax.set_title("Predicted Trajectory (clockwise rotated)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*axis_limits[0])
        ax.set_ylim(*axis_limits[1])

    fig.tight_layout()
    fig.canvas.draw()

    # Convert to numpy array
    traj_img = np.asarray(fig.canvas.buffer_rgba())
    traj_img = cv2.cvtColor(traj_img, cv2.COLOR_RGBA2BGR)

    plt.close(fig)
    return traj_img

def add_text_to_image(img: np.ndarray, text: str, position: str = "top") -> np.ndarray:
    """Add text overlay to image."""
    img_copy = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    text_color = (255, 255, 255)  # White in BGR
    bg_color = (0, 0, 0)  # Black background
    margin = 15
    padding = 10
    
    # Split text into lines if too long
    max_width = img_copy.shape[1] - 2 * margin
    lines = []
    for line in text.split("\n"):
        while len(line) > 0:
            text_size = cv2.getTextSize(line[:100], font, font_scale, font_thickness)[0]
            if text_size[0] <= max_width:
                lines.append(line)
                break
            else:
                lines.append(line[:50])
                line = line[50:]
    
    if not lines:
        return img_copy
    
    # Get text dimensions and calculate box size
    text_height = cv2.getTextSize("Aq", font, font_scale, font_thickness)[0][1]
    line_height = text_height + 8
    total_height = len(lines) * line_height + 2 * padding
    
    if position == "top":
        y_start = margin + padding
    else:
        y_start = img_copy.shape[0] - total_height - margin
    
    # Clamp box to left half of image
    x_min = margin
    x_max = min(img_copy.shape[1] // 2, margin + max_width + 2 * padding)
    y_min = max(0, y_start - padding)
    y_max = min(img_copy.shape[0], y_start + total_height)
    
    # Draw semi-transparent background
    overlay = img_copy.copy()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), bg_color, -1)
    img_copy = cv2.addWeighted(overlay, 0.7, img_copy, 0.3, 0)
    
    # Draw text
    for i, line in enumerate(lines):
        y = y_start + padding + i * line_height + text_height
        cv2.putText(img_copy, line, (margin + padding, y), font, font_scale, text_color, font_thickness)
    
    return img_copy


def visualize_trajectory(
    csv_path: Path,
    image_dir: Path,
    output_dir: Path,
    num_frames: int = 4,
    create_video: bool = True,
    video_fps: int = 30,
    video_max_width: int = 1920,
) -> None:
    """Visualize trajectories from CSV and create output images/video."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = load_csv_results(csv_path)
    
    print(f"Loaded {len(results)} windows from {csv_path}")

    # Fixed axis limits: ±10 m on both axes
    axis_limits = ((-10.0, 10.0), (-10.0, 10.0))
    print(f"Using fixed trajectory axis limits: x={axis_limits[0]}, y={axis_limits[1]}")
    
    frame_paths = []
    
    for window_idx in sorted(results.keys()):
        window_data = results[window_idx]
        image_path = get_image_for_window(window_idx, image_dir, num_frames)
        
        if image_path is None:
            print(f"Warning: Image not found for window {window_idx}")
            continue
        
        # Load input image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not load image {image_path}")
            continue
        
        # Create trajectory plot
        traj_plot = create_trajectory_plot(window_data["steps"], axis_limits)
        
        # Resize images to same height for side-by-side layout
        target_height = max(img.shape[0], traj_plot.shape[0])
        img_resized = cv2.resize(img, (int(img.shape[1] * target_height / img.shape[0]), target_height))
        traj_resized = cv2.resize(traj_plot, (int(traj_plot.shape[1] * target_height / traj_plot.shape[0]), target_height))
        
        # Combine images side by side
        combined = np.hstack([img_resized, traj_resized])
        
        # Add CoT text at the top
        if window_data["cot"]:
            combined = add_text_to_image(combined, f"CoT: {window_data['cot']}", position="top")
        
        # Save frame
        output_path = output_dir / f"frame_{window_idx:06d}.png"
        cv2.imwrite(str(output_path), combined)
        frame_paths.append(str(output_path))
        
        if (window_idx + 1) % 100 == 0:
            print(f"Processed {window_idx + 1} / {len(results)} windows")
    
    print(f"Saved {len(frame_paths)} visualization frames to {output_dir}")
    
    # Create video
    if create_video and frame_paths:
        video_path = output_dir / "trajectory_visualization.mp4"
        create_video_from_frames(frame_paths, str(video_path), fps=video_fps, max_width=video_max_width)
        print(f"Saved video to {video_path}")


def create_video_from_frames(
    frame_paths: list[str],
    output_video_path: str,
    fps: int = 30,
    max_width: int = 1920,
) -> None:
    """Create video from a list of frame paths using ffmpeg."""

    if not frame_paths:
        print("No frames to create video")
        return

    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg is not installed or not found in PATH")
        return
    
    # Use ffmpeg for more reliable video encoding
    output_dir = Path(output_video_path).parent
    frame_pattern = str(output_dir / "frame_%06d.png")
    
    try:
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-c:v", "libx264",  # H.264 codec via ffmpeg
            "-vf", f"scale='min(iw,{max_width})':-2",
            "-pix_fmt", "yuv420p",  # Pixel format for compatibility
            "-profile:v", "main",
            "-movflags", "+faststart",
            "-preset", "fast",  # Fast encoding
            "-crf", "23",  # Quality (lower = better, 0-51)
            output_video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Video saved to {output_video_path}")
        else:
            print(f"Error encoding video: {result.stderr}")

    except FileNotFoundError:
        print("Error: ffmpeg is not installed or not found in PATH")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to the CSV file from infer_from_images.py",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("input_images"),
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("trajectory_visualizations"),
        help="Output directory for visualization frames and video.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Number of temporal frames per window (should match infer_from_images.py).",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip video creation.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video frame rate.",
    )
    parser.add_argument(
        "--video-max-width",
        type=int,
        default=1920,
        help="Scale output video down to this width if larger.",
    )
    
    args = parser.parse_args()
    
    if not args.csv.exists():
        raise ValueError(f"CSV file not found: {args.csv}")
    
    if not args.image_dir.exists():
        raise ValueError(f"Image directory not found: {args.image_dir}")
    
    visualize_trajectory(
        args.csv,
        args.image_dir,
        args.output_dir,
        num_frames=args.num_frames,
        create_video=not args.no_video,
        video_fps=args.fps,
        video_max_width=args.video_max_width,
    )


if __name__ == "__main__":
    main()
