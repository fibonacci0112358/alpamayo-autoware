# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import torch

from alpamayo1_5.models.base_model import _resolve_torch_dtype, _ensure_qwen3vl_rope_scaling
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from transformers import Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def _get_param_count(module: torch.nn.Module) -> dict[str, int]:
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def _freeze_module(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = False


class TrainableAlpamayo1_5(Alpamayo1_5):
    """Minimal Trainable wrapper for Alpamayo1.5 suitable as Stage1 starter.

    Notes:
    - This wrapper intentionally keeps dependencies minimal so it can live inside
      this repository as a starting point. Training harness (Trainer, data
      pipeline, LoRA utilities) can be added or wired from an external training
      repo by copying necessary pieces into `finetune/sft/`.
    """

    def __init__(
        self,
        config: Any,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
        cotrain_vlm: bool = False,
        stop_grad_from_vlm: bool = True,
        stage1_vlm_checkpoint_path: str | None = None,
        use_lora: bool = False,
        lora_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        loaded_alpamayo: Alpamayo1_5 | None = None
        if pretrained_modules is None or "vlm" not in pretrained_modules:
            if config is None or getattr(config, "vlm_name_or_path", None) is None:
                raise ValueError("config.vlm_name_or_path is required to load Alpamayo1.5 weights")

            try:
                # Prefer loading the native Alpamayo checkpoint first so `vlm.*`
                # prefixed keys are restored correctly.
                loaded_alpamayo = Alpamayo1_5.from_pretrained(
                    config.vlm_name_or_path,
                    torch_dtype=_resolve_torch_dtype(config.model_dtype),
                    attn_implementation=config.attn_implementation,
                    ignore_mismatched_sizes=True,
                )
            except Exception as e:
                # The current Alpamayo HF implementation may reject FA2.
                # Retry without explicit attn_implementation before falling back.
                if "does not support Flash Attention 2.0" in str(e):
                    logger.warning(
                        "Retrying Alpamayo load without flash_attention_2 for vlm_name_or_path=%s",
                        config.vlm_name_or_path,
                    )
                    loaded_alpamayo = Alpamayo1_5.from_pretrained(
                        config.vlm_name_or_path,
                        torch_dtype=_resolve_torch_dtype(config.model_dtype),
                        attn_implementation=None,
                        ignore_mismatched_sizes=True,
                    )
                else:
                    raise

            try:
                if loaded_alpamayo is None:
                    raise RuntimeError("Alpamayo checkpoint load returned None")

                vlm = loaded_alpamayo.vlm
                _ensure_qwen3vl_rope_scaling(vlm.config)
                original_vocab_size = vlm.config.text_config.vocab_size

                target_vocab_size = len(getattr(loaded_alpamayo, "tokenizer", []) or [])
                if target_vocab_size > 0 and target_vocab_size != vlm.config.vocab_size:
                    logger.info(
                        "Resizing Alpamayo VLM embeddings from %s to tokenizer length %s",
                        vlm.config.vocab_size,
                        target_vocab_size,
                    )
                    vlm.resize_token_embeddings(target_vocab_size, mean_resizing=False)
                    vlm.config.text_config.vocab_size = target_vocab_size
                    vlm.config.vocab_size = target_vocab_size

                # Keep vocab aligned with checkpoint to avoid embedding reinit.
                if getattr(config, "vocab_size", None) != vlm.config.vocab_size:
                    logger.warning(
                        "Overriding config.vocab_size=%s with checkpoint vocab_size=%s",
                        getattr(config, "vocab_size", None),
                        vlm.config.vocab_size,
                    )
                    config.vocab_size = vlm.config.vocab_size

                pretrained_modules = {"vlm": vlm}
            except Exception as e:
                logger.warning(
                    "Falling back to direct Qwen3VL load for vlm_name_or_path=%s due to: %s",
                    config.vlm_name_or_path,
                    e,
                )
                vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                    config.vlm_name_or_path,
                    dtype=_resolve_torch_dtype(config.model_dtype),
                    attn_implementation=config.attn_implementation,
                )
                # Ensure rope_scaling is set for Qwen3VL compatibility
                _ensure_qwen3vl_rope_scaling(vlm.config)

                original_vocab_size = vlm.config.text_config.vocab_size
                vlm.resize_token_embeddings(config.vocab_size, mean_resizing=False)
                vlm.config.text_config.vocab_size = config.vocab_size
                vlm.config.vocab_size = config.vocab_size
                pretrained_modules = {"vlm": vlm}

        super().__init__(config, pretrained_modules, original_vocab_size)

        if loaded_alpamayo is not None:
            # Keep tokenizer/token IDs consistent with the checkpoint that loaded `vlm`.
            self.tokenizer = loaded_alpamayo.tokenizer
            self.special_token_ids = loaded_alpamayo.special_token_ids
            if getattr(loaded_alpamayo, "traj_tokenizer", None) is not None:
                self.traj_tokenizer = loaded_alpamayo.traj_tokenizer
            if getattr(loaded_alpamayo, "hist_traj_tokenizer", None) is not None:
                self.hist_traj_tokenizer = loaded_alpamayo.hist_traj_tokenizer
            if getattr(loaded_alpamayo, "future_token_start_idx", None) is not None:
                self.future_token_start_idx = loaded_alpamayo.future_token_start_idx
            if getattr(loaded_alpamayo, "hist_token_start_idx", None) is not None:
                self.hist_token_start_idx = loaded_alpamayo.hist_token_start_idx

        self.cotrain_vlm = cotrain_vlm
        self.stop_grad_from_vlm = stop_grad_from_vlm

        # Stage1 SFT only routes gradients through the VLM path. Keep the
        # expert/action stack frozen so trainable parameter counts and backward
        # behavior reflect the actual optimization target.
        _freeze_module(getattr(self, "expert", None))
        _freeze_module(getattr(self, "action_space", None))
        _freeze_module(getattr(self, "diffusion", None))
        _freeze_module(getattr(self, "action_in_proj", None))
        _freeze_module(getattr(self, "action_out_proj", None))

        if use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model

                lora_kwargs = dict(lora_config or {})
                lora_kwargs.pop("use_lora", None)
                lora_kwargs.pop("lora_config", None)
                lora_kwargs.pop("_target_", None)
                if "target_modules" not in lora_kwargs:
                    lora_kwargs["target_modules"] = [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ]
                if "task_type" not in lora_kwargs:
                    lora_kwargs["task_type"] = TaskType.CAUSAL_LM
                peft_config = LoraConfig(**lora_kwargs)
                self.vlm = get_peft_model(self.vlm, peft_config)
                logger.info("Enabled LoRA on Alpamayo1.5 VLM")
            except Exception as e:
                logger.warning(f"Failed to enable LoRA, continuing without it: {e}")

        # If the user provides a stage1 VLM checkpoint (state_dict), try to load
        # it into the internal `vlm` module. We use a permissive load (non-strict)
        # because checkpoint keys may vary; the caller should validate after load.
        if stage1_vlm_checkpoint_path is not None:
            try:
                sd = torch.load(stage1_vlm_checkpoint_path, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                self.vlm.load_state_dict(sd, strict=False)
                logger.info("Loaded Stage1 VLM checkpoint into wrapper (non-strict)")
            except Exception as e:
                logger.warning(f"Failed to load stage1_vlm_checkpoint: {e}")

        # By default freeze the VLM unless cotrain_vlm is requested. When LoRA
        # is enabled we keep the PEFT adapters trainable and only freeze the
        # VLM backbone in the non-LoRA path.
        if not self.cotrain_vlm and not use_lora:
            for p in self.vlm.parameters():
                p.requires_grad = False

        # Log parameter counts to help debugging
        logger.info("Model parameter count:")
        param_count = _get_param_count(self)
        for key, value in param_count.items():
            logger.info(f"{key}: {value:,}")

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass for Stage1 SFT.

        The collator feeds standard multimodal inputs plus optional trajectory
        history tensors. When trajectory history is provided, the prompt tokens
        are fused in the same way as inference-time inputs.
        """

        traj_data = None
        if ego_history_xyz is not None and ego_history_rot is not None and input_ids is not None:
            traj_data = {
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
            }
            # Stage1 training can run without trajectory token fusion when the
            # tokenizer mixin is not configured; in that case we keep the raw
            # multimodal text/image path and ignore the trajectory side inputs.
            if getattr(self, "hist_traj_tokenizer", None) is not None and getattr(self, "hist_token_start_idx", None) is not None:
                input_ids = self.fuse_traj_tokens(input_ids, traj_data)
                if labels is not None and labels.shape == input_ids.shape:
                    labels = input_ids.clone().masked_fill(labels == -100, -100)

        if kwargs.get("use_cache") is None:
            kwargs["use_cache"] = False

        return self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, *args, **kwargs) -> dict[str, Any]:
        # Delegate to underlying VLM if possible
        if hasattr(self.vlm, "prepare_inputs_for_generation"):
            return self.vlm.prepare_inputs_for_generation(*args, **kwargs)

        return super().prepare_inputs_for_generation(*args, **kwargs)
