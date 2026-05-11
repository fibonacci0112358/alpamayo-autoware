# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Mapping

import torch

from .sft_alpamayo_1_5 import TrainableAlpamayo1_5


class TrainableAlpamayo1_5_Stage2(TrainableAlpamayo1_5):
    """Stage2 wrapper skeleton: compute action/expert loss.

    This is a minimal skeleton: concrete loss computation and data handling
    must be implemented to match Alpamayo1.5 action/expert expectations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_action_loss(self, model_outputs: Mapping[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """Compute action/expert loss.

        Expects `labels` to have shape compatible with the action space
        (e.g., [B, T, C_action]) and `model_outputs` to contain one of the
        prediction keys: ``pred``, ``pred_action``, ``action_logits``, or
        ``noise_pred``. The default loss is mean-squared error (L2) between
        the prediction and the labels. If `model_outputs` contains a
        `mask` tensor it will be used to average only over valid positions.
        """
        device = next(self.parameters()).device

        # possible output keys from different training harnesses
        pred = None
        for key in ("pred", "pred_action", "action_logits", "noise_pred", "pred_noise"):
            if key in model_outputs:
                pred = model_outputs[key]
                break

        if pred is None:
            # no prediction available — return zero loss to keep pipeline runnable
            return torch.tensor(0.0, device=device)

        # Ensure labels and pred are float tensors on same device
        pred = pred.float().to(device)
        labels = labels.float().to(device)

        # Optional mask: 1 = valid, 0 = ignore
        mask = model_outputs.get("mask", None)
        if mask is not None:
            mask = mask.float().to(device)
            # Broadcast mask to match pred shape if necessary
            if mask.dim() < pred.dim():
                mask = mask.unsqueeze(-1)
            diff = (pred - labels) ** 2 * mask
            loss = diff.sum() / (mask.sum().clamp_min(1.0))
            return loss

        # Default MSE
        return torch.nn.functional.mse_loss(pred, labels)
