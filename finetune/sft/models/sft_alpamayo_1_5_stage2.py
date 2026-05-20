# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from typing import Any, Mapping
import logging

import einops
import torch

from alpamayo1_5.models.base_model import IGNORE_INDEX

from .sft_alpamayo_1_5 import TrainableAlpamayo1_5


logger = logging.getLogger(__name__)


class TrainableAlpamayo1_5_Stage2(TrainableAlpamayo1_5):
    """Stage2 wrapper for action/expert diffusion loss.

    This mirrors the Alpamayo R1 Stage2 flow: fuse trajectory history tokens,
    run the VLM to obtain KV cache, build diffusion training targets from the
    future trajectory, and compute the expert loss from the denoised action
    prediction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # For Stage2 we need to train the expert / action / diffusion stacks.
        # The Stage1 wrapper freezes these by default; re-enable gradients
        # for the modules that should be optimized during Stage2.
        for name in ("expert", "action_space", "diffusion", "action_in_proj", "action_out_proj"):
            module = getattr(self, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True

        # Ensure VLM remains frozen unless cotrain_vlm is requested, and
        # enable gradients for everything else to guarantee the backward
        # graph connects to trainable parameters during Stage2.
        try:
            if not getattr(self, "cotrain_vlm", False):
                for n, p in self.named_parameters():
                    if n.startswith("vlm."):
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
        except Exception:
            pass

        # Log updated trainable parameter count for visibility during startup.
        try:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info("Stage2 init: total_params=%d trainable_params=%d", total, trainable)
            print(f"Stage2 init: total_params={total} trainable_params={trainable}")
        except Exception:
            pass

    def _process_position_ids_qwen2_5_vl(
        self, vlm_outputs: Any, batch_size: int, num_expert_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """Process the position ids for the expert model.

        Qwen 2.5 VL has a special RoPE, so we need to process the position ids
        Args:
            vlm_outputs: The outputs of the VLM model.
            batch_size: The batch size.
            num_expert_tokens: The number of expert tokens.
            device: The device.
        Returns:
            The processed position ids.
        """
        position_ids = torch.arange(num_expert_tokens, device=device)
        position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch_size).clone()
        delta = vlm_outputs.rope_deltas + vlm_outputs.past_key_values.get_seq_length()
        position_ids += delta.to(position_ids.device)
        return position_ids

    def forward(self, **inputs: Any) -> dict:
        """Forward for Stage2 action/expert SFT.
        (Debug Version: try-except removed from VLM forward to expose traceback)
        """

        device = next(self.parameters()).device

        def zero_loss():
            trainable_param = next((p for p in self.parameters() if p.requires_grad), None)
            if trainable_param is not None:
                return (trainable_param.sum() * 0).float()
            return torch.zeros((), device=device, requires_grad=True)

        tokenized_data = dict(inputs)
        input_ids = tokenized_data.pop("input_ids", None)

        if input_ids is None:
            return {"loss": zero_loss()}

        batch_size = input_ids.shape[0]
        labels = tokenized_data.pop("labels", None)

        ego_history_xyz = tokenized_data.pop("ego_history_xyz", None)
        ego_history_rot = tokenized_data.pop("ego_history_rot", None)
        ego_future_xyz = tokenized_data.pop("ego_future_xyz", None)
        ego_future_rot = tokenized_data.pop("ego_future_rot", None)
        labels_mask = tokenized_data.pop("labels_mask", None)

        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }

        # 1. Fuse trajectory tokens into input_ids & Align Masks
        try:
            if (
                hasattr(self, "fuse_traj_tokens")
                and getattr(self, "hist_traj_tokenizer", None) is not None
                and getattr(self, "hist_token_start_idx", None) is not None
            ):
                original_len = input_ids.shape[1]
                input_ids = self.fuse_traj_tokens(input_ids, traj_data)
                new_len = input_ids.shape[1]

                if new_len > original_len:
                    diff_len = new_len - original_len
                    if "attention_mask" in tokenized_data:
                        extra_mask = torch.ones((batch_size, diff_len), 
                                                device=device, dtype=tokenized_data["attention_mask"].dtype)
                        tokenized_data["attention_mask"] = torch.cat([tokenized_data["attention_mask"], extra_mask], dim=1)
                    
                    if labels_mask is not None:
                        extra_labels_mask = torch.zeros((batch_size, diff_len), 
                                                        device=device, dtype=labels_mask.dtype)
                        labels_mask = torch.cat([labels_mask, extra_labels_mask], dim=1)

                if "position_ids" in tokenized_data:
                    del tokenized_data["position_ids"]

        except Exception as e:
            logger.warning("Stage2 forward: fuse_traj_tokens failed: %s", e)

        # 2. Prepare labels
        if labels is None:
            labels = input_ids.clone()
        elif labels_mask is not None:
            labels = torch.where(labels_mask.bool(), labels, torch.full_like(labels, IGNORE_INDEX))

        # 3. VLM forward pass
        raw_vlm_kwargs = dict(tokenized_data)
        allowed_vlm_keys = {
            "attention_mask",
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "position_ids",
            "cache_position",
        }
        vlm_kwargs = {k: v for k, v in raw_vlm_kwargs.items() if k in allowed_vlm_keys}
        vlm_labels = labels if self.cotrain_vlm else None

        vlm_outputs = None
        context = nullcontext() if self.cotrain_vlm else torch.no_grad()
        with context:
            vlm_outputs = self.vlm(
                input_ids=input_ids,
                labels=vlm_labels,
                use_cache=True,
                **vlm_kwargs,
            )

        if ego_future_xyz is None or ego_future_rot is None:
            return {"loss": zero_loss()}

        # 4. Process trajectory for future action/diffusion training
        try:
            ego_history_xyz_f32 = ego_history_xyz.float() if ego_history_xyz is not None and ego_history_xyz.dtype != torch.float32 else ego_history_xyz
            ego_history_rot_f32 = ego_history_rot.float() if ego_history_rot is not None and ego_history_rot.dtype != torch.float32 else ego_history_rot
            ego_future_xyz_f32 = ego_future_xyz.float() if ego_future_xyz is not None and ego_future_xyz.dtype != torch.float32 else ego_future_xyz
            ego_future_rot_f32 = ego_future_rot.float() if ego_future_rot is not None and ego_future_rot.dtype != torch.float32 else ego_future_rot
            
            action = self.action_space.traj_to_action(
                traj_history_xyz=ego_history_xyz_f32,
                traj_history_rot=ego_history_rot_f32,
                traj_future_xyz=ego_future_xyz_f32,
                traj_future_rot=ego_future_rot_f32,
            )
            action = action.reshape(-1, *self.action_space.get_action_space_dims())
            training_data = self.diffusion.construct_training_data(action)
        except Exception as e:
            logger.warning("Stage2 forward: traj_to_action/diffusion training_data failed: %s", e)
            return {"loss": zero_loss()}

        try:
            proj_dtype = next(self.action_in_proj.parameters()).dtype
            noisy_x = training_data["noisy_x"].to(dtype=proj_dtype)
            timesteps = training_data["timesteps"].to(dtype=proj_dtype)
            action_embeds = self.action_in_proj(noisy_x, timesteps)
            expert_embeds = action_embeds
        except Exception as e:
            logger.warning("Stage2 forward: action_in_proj failed: %s", e)
            return {"loss": zero_loss()}

        # 5. Get and process KV cache
        try:
            kv_cache = None
            if vlm_outputs is not None:
                kv_cache = getattr(vlm_outputs, "past_key_values", None)
        except Exception as e:
            logger.warning("Stage2 forward: accessing past_key_values failed: %s", e)
            kv_cache = None

        # 6. Attempt to crop KV cache
        try:
            if hasattr(self, "config") and hasattr(self.config, "traj_token_ids"):
                future_start_token_id = self.config.traj_token_ids.get("future_start")
                if future_start_token_id is not None:
                    future_positions = (input_ids == future_start_token_id).nonzero(as_tuple=False)
                    if future_positions.numel() > 0:
                        last_traj_future_start_idx = future_positions[-1, 1] + 1
                        if hasattr(kv_cache, "crop"):
                            kv_cache.crop(last_traj_future_start_idx)
        except Exception as e:
            logger.warning("Stage2 forward: kv_cache crop failed: %s", e)

        if kv_cache is not None and self.stop_grad_from_vlm:
            try:
                for layer in getattr(kv_cache, "layers", []):
                    if hasattr(layer, "keys"):
                        layer.keys = layer.keys.detach()
                    if hasattr(layer, "values"):
                        layer.values = layer.values.detach()
            except Exception:
                pass

        # 7. Prepare position_ids
        try:
            if vlm_outputs is not None:
                position_ids = self._process_position_ids_qwen2_5_vl(
                    vlm_outputs, batch_size, expert_embeds.shape[1], expert_embeds.device
                )
            else:
                position_ids = torch.arange(expert_embeds.shape[1], device=expert_embeds.device)
                position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)
        except Exception:
            position_ids = torch.arange(expert_embeds.shape[1], device=expert_embeds.device)
            position_ids = position_ids.unsqueeze(0).repeat(batch_size, 1)

        # 8. Run expert forward
        pred = None
        if kv_cache is not None:
            try:
                forward_kwargs = {}
                if getattr(self.config, "expert_non_causal_attention", False):
                    forward_kwargs["is_causal"] = False
                expert_outputs = self.expert(
                    inputs_embeds=expert_embeds,
                    position_ids=position_ids,
                    past_key_values=kv_cache,
                    attention_mask=None,
                    use_cache=True,
                    **forward_kwargs,
                )
                diffusion_out = expert_outputs.last_hidden_state[:, -expert_embeds.shape[1] :]
                pred = self.action_out_proj(diffusion_out)
                pred = pred.view(-1, *self.action_space.get_action_space_dims())
            except Exception as e:
                logger.warning("Stage2 forward: expert forward failed: %s", e)
                pred = None
        
        if pred is None:
            try:
                pred = self.action_out_proj(expert_embeds)
                pred = pred.view(-1, *self.action_space.get_action_space_dims())
            except Exception as e:
                logger.warning("Stage2 forward: fallback action_out_proj failed: %s", e)
                return {"loss": zero_loss()}

        # 9. Compute diffusion loss
        try:
            future_traj_loss = self.diffusion.compute_loss_from_pred(training_data=training_data, pred=pred)

            future_traj_loss = torch.nan_to_num(future_traj_loss, nan=0.0, posinf=1e9, neginf=-1e9)
            loss = future_traj_loss
            if self.cotrain_vlm and vlm_outputs is not None and hasattr(vlm_outputs, "loss"):
                try:
                    loss = loss + vlm_outputs.loss
                except Exception as e:
                    pass
        except Exception as e:
            logger.warning("Stage2 forward: diffusion loss failed: %s", e)
            return {"loss": zero_loss()}

        result = {"loss": loss, "action_loss": future_traj_loss}
        if self.cotrain_vlm and vlm_outputs is not None and hasattr(vlm_outputs, "loss"):
            result["vlm_loss"] = vlm_outputs.loss
        return result