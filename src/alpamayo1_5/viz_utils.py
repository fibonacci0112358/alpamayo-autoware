# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BEV trajectory visualization utilities for nav-conditioned analysis.

Provides functions for plotting multi-condition trajectory comparisons
with camera image grids and BEV trajectory distributions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def get_trajectories_xy(pred_xyz: torch.Tensor) -> np.ndarray:
    """Extract XY trajectories from prediction tensor.

    Args:
        pred_xyz: Shape ``[B, n_traj_group, K, T, 3]`` or ``[1, 1, K, T, 3]``.

    Returns:
        Array of shape ``[K, T, 2]``.
    """
    return pred_xyz[0, 0, :, :, :2].detach().to(dtype=torch.float32).cpu().numpy()


MIN_AXIS_RANGE_M = 2.0
MIN_Y_RANGE_RATIO = 0.3
BEV_X_LIM = (-5.0, 5.0)
BEV_Y_LIM = (0.0, 10.0)


def _truncate(text: str, maxlen: int = 40) -> str:
    return text if len(text) <= maxlen else text[: maxlen - 1] + "\u2026"


def _enforce_readable_axes(ax: plt.Axes) -> None:
    """Ensure both axes have a minimum range and the plot isn't too flat."""
    for getter, setter in [(ax.get_xlim, ax.set_xlim), (ax.get_ylim, ax.set_ylim)]:
        lo, hi = getter()
        if hi - lo < MIN_AXIS_RANGE_M:
            mid = (lo + hi) / 2
            setter(mid - MIN_AXIS_RANGE_M / 2, mid + MIN_AXIS_RANGE_M / 2)

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    x_range = xhi - xlo
    y_range = yhi - ylo
    if y_range < MIN_Y_RANGE_RATIO * x_range:
        ymid = (ylo + yhi) / 2
        half = MIN_Y_RANGE_RATIO * x_range / 2
        ax.set_ylim(ymid - half, ymid + half)


def set_fixed_bev_limits(ax: plt.Axes) -> None:
    """Apply the fixed BEV display window requested for all plots."""
    ax.set_xlim(*BEV_X_LIM)
    ax.set_ylim(*BEV_Y_LIM)


def _crop_black_border(image: np.ndarray) -> np.ndarray:
    """Trim fully black borders from a camera frame."""
    mask = np.any(image > 0, axis=-1)
    if not np.any(mask):
        return image

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return image[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def plot_condition(ax: plt.Axes, trajs: np.ndarray, color: str, label: str) -> None:
    """Draw one representative trajectory for a condition.

    Args:
        ax: Matplotlib axes.
        trajs: Trajectories, shape ``[K, T, 2]``.
        color: Color for this condition.
        label: Legend label.
    """
    median = np.median(trajs, axis=0)
    ax.plot(median[:, 0], median[:, 1], color=color, linewidth=2.5, label=label)


def plot_bev_comparison(
    pred_with_nav: torch.Tensor,
    pred_no_nav: torch.Tensor,
    pred_counterfactual: torch.Tensor,
    nav_text: str,
    nav_text_swapped: str,
    gt_future_xyz: torch.Tensor | None = None,
    camera_images: np.ndarray | None = None,
    title: str = "",
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Two-panel figure: camera grid (top) + BEV trajectory comparison (bottom).

    Args:
        pred_with_nav: Predictions conditioned on nav, ``[1, 1, K, T, 3]``.
        pred_no_nav: Predictions without nav, ``[1, 1, K, T, 3]``.
        pred_counterfactual: Predictions with swapped nav, ``[1, 1, K, T, 3]``.
        nav_text: Original navigation instruction.
        nav_text_swapped: Direction-swapped navigation instruction.
        gt_future_xyz: Ground-truth trajectory, ``[1, 1, T, 3]`` or ``[1, T, 3]``.
        camera_images: Optional camera image grid as numpy array for the top panel.
        title: Plot title for the BEV panel.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    has_cam = camera_images is not None
    nrows = 2 if has_cam else 1
    height_ratios = [1, 1.3] if has_cam else [1]
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if nrows == 1:
        axes = [axes]

    if has_cam:
        ax_cam = axes[0]
        ax_cam.imshow(np.clip(camera_images / 255.0, 0, 1))
        ax_cam.set_title("Camera view at t0", fontsize=9)
        ax_cam.axis("off")

    ax = axes[-1]

    nav_orig = _truncate(nav_text)
    nav_swap = _truncate(nav_text_swapped)

    # Rotate display so that +x points upward by applying a 90deg CCW
    # rotation to coordinates: (x, y) -> (-y, x).
    def rotate_90ccw(trajs: np.ndarray) -> np.ndarray:
        # trajs: [K, T, 2] -> rotated same shape
        return np.stack((-trajs[..., 1], trajs[..., 0]), axis=-1)

    conditions = [
        (rotate_90ccw(get_trajectories_xy(pred_with_nav)), "tab:blue", f'p(traj|nav="{nav_orig}")'),
        (rotate_90ccw(get_trajectories_xy(pred_no_nav)), "tab:red", "p(traj)"),
        (rotate_90ccw(get_trajectories_xy(pred_counterfactual)), "tab:green", f'p(traj|opposite nav="{nav_swap}")'),
    ]
    for trajs, color, label in conditions:
        plot_condition(ax, trajs, color, label)

    if gt_future_xyz is not None:
        gt = gt_future_xyz.cpu() if gt_future_xyz.is_cuda else gt_future_xyz
        if gt.dim() == 4:
            g = gt[0, 0, :, :2].numpy()
        else:
            g = gt[0, :, :2].numpy()
        # rotate GT the same way: (x, y) -> (-y, x)
        g = np.stack((-g[:, 1], g[:, 0]), axis=-1)
        ax.plot(g[:, 0], g[:, 1], color="black", linewidth=2.5, label="GT")

    # After rotation: horizontal plotted axis corresponds to -y, vertical to x
    ax.set_xlabel("-y (m)")
    ax.set_ylabel("x (m)")
    ax.set_aspect("equal")
    set_fixed_bev_limits(ax)
    _enforce_readable_axes(ax)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


CAMERA_GRID_LAYOUT = {
    # Row 0: [pad, front_tele, pad]
    # Row 1: [cross_left, front_wide, cross_right]
    0: (1, 0),  # cross_left -> row 1, col 0
    1: (1, 1),  # front_wide -> row 1, col 1
    2: (1, 2),  # cross_right -> row 1, col 2
    6: (0, 1),  # front_tele -> row 0, col 1
}


def make_camera_grid(
    image_frames: torch.Tensor,
    camera_indices: torch.Tensor | None = None,
    ncols: int = 3,
) -> np.ndarray:
    """Arrange multi-camera image frames into a 2x3 grid for display.

    When ``camera_indices`` is provided, cameras are placed in a
    semantically meaningful layout::

        Row 0: [      pad       ] [  front_tele  ] [      pad       ]
        Row 1: [  cross_left    ] [  front_wide  ] [  cross_right   ]

    Falls back to sequential row-major layout if ``camera_indices`` is not provided.

    Args:
        image_frames: Shape ``[N_cameras, num_frames, C, H, W]`` uint8 tensors.
        camera_indices: Optional camera index tensor from ``data["camera_indices"]``.
        ncols: Number of columns in the grid.

    Returns:
        Numpy array of the grid image, shape ``[grid_H, grid_W, 3]``, uint8.
    """
    last_frames = image_frames[:, -1]  # [N_cameras, C, H, W]
    frames = last_frames.permute(0, 2, 3, 1).numpy()  # [N, H, W, C]
    frames = [_crop_black_border(frame) for frame in frames]

    if camera_indices is not None:
        # Render only cameras that are actually present, ordered by the
        # semantic layout, so unused slots do not appear as black rectangles.
        rows: dict[int, list[tuple[int, np.ndarray]]] = {}
        cam_ids = camera_indices.tolist()
        for i, cam_id in enumerate(cam_ids):
            if cam_id in CAMERA_GRID_LAYOUT:
                r, c = CAMERA_GRID_LAYOUT[cam_id]
                frame = frames[i]
                if not np.any(frame > 0):
                    continue
                rows.setdefault(r, []).append((c, frame))

        if rows:
            rendered_rows = []
            for r in sorted(rows):
                ordered = [img for _, img in sorted(rows[r], key=lambda item: item[0])]
                rendered_rows.append(ordered[0] if len(ordered) == 1 else np.concatenate(ordered, axis=1))
            if len(rendered_rows) == 1:
                return rendered_rows[0]

            max_width = max(row.shape[1] for row in rendered_rows)
            total_height = sum(row.shape[0] for row in rendered_rows)
            canvas = np.full((total_height, max_width, 3), 255, dtype=rendered_rows[0].dtype)
            y = 0
            for row in rendered_rows:
                height, width = row.shape[:2]
                canvas[y : y + height, :width] = row
                y += height
            return canvas

    n = frames.shape[0]
    nrows = (n + ncols - 1) // ncols
    while len(frames) < nrows * ncols:
        frames = np.concatenate([frames, np.zeros_like(frames[:1])], axis=0)
    rows = []
    for r in range(nrows):
        row = np.concatenate(frames[r * ncols : (r + 1) * ncols], axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)
