# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from alpamayo1_5.models.base_model import _resolve_torch_dtype, _ensure_qwen3vl_rope_scaling
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.config import Alpamayo1_5Config
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, AutoConfig
from safetensors.torch import load_file as load_safetensors_file
import json

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# Ignore index for loss computation (matching Transformers default)
IGNORE_INDEX = -100


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


def _resolve_vlm_model_path(vlm_name_or_path: str) -> str:
    """Resolve a loadable HF checkpoint path for the VLM.

    Some Stage-1 output roots in this repo only contain tokenizer artifacts and
    ZeRO optimizer shards. When that happens, fall back to the canonical base
    Alpamayo VLM checkpoint so `from_pretrained` does not fail immediately.
    """
    candidate = Path(vlm_name_or_path)
    if not candidate.exists():
        return vlm_name_or_path

    weight_files = [
        candidate / "model.safetensors",
        candidate / "pytorch_model.bin",
        candidate / "model.safetensors.index.json",
        candidate / "pytorch_model.bin.index.json",
    ]
    if any(path.exists() for path in weight_files):
        return vlm_name_or_path

    # Some checkpoints are stored in nested step directories. Search one level
    # deeper for a standard HF weight file before giving up.
    for nested_dir in sorted(candidate.glob("**/")):
        if not nested_dir.is_dir():
            continue
        if any((nested_dir / name).exists() for name in ("model.safetensors", "pytorch_model.bin")):
            return str(nested_dir)

    fallback = os.getenv("ALPAMAYO_BASE_VLM_NAME_OR_PATH", "nvidia/Alpamayo-1.5-10B")
    logger.warning(
        "VLM checkpoint path %s does not contain loadable HF weights; falling back to %s",
        vlm_name_or_path,
        fallback,
    )
    return fallback


def _load_vlm_from_local_path(vlm_path: str, dtype: torch.dtype, attn_impl: str | None) -> Alpamayo1_5:
    """Load Alpamayo VLM directly from local path without using HF Hub validation."""
    vlm_path = Path(vlm_path).resolve()  # Resolve to absolute path
    
    logger.info(f"Looking for config.json in: {vlm_path}")
    logger.info(f"Directory exists: {vlm_path.exists()}")
    if vlm_path.exists():
        logger.info(f"Directory contents: {list(vlm_path.glob('*'))[:10]}")  # Log first 10 items
    
    # Try to find config.json in current directory or nested directories
    config_json = vlm_path / "config.json"
    if not config_json.exists():
        logger.info(f"config.json not in root, searching nested directories...")
        # Search for config.json in nested directories (e.g., checkpoint-*)
        found_configs = list(vlm_path.glob("**/config.json"))
        logger.info(f"Found {len(found_configs)} config.json files: {found_configs[:5]}")
        
        if found_configs:
            config_json = found_configs[0]
            logger.info(f"Using first found config from: {config_json}")
        else:
            raise FileNotFoundError(f"config.json not found in {vlm_path} or any nested directory. Directory exists: {vlm_path.exists()}")
    
    config_dir = config_json.parent
    logger.info(f"Using config from: {config_json}")
    
    # Load config
    with open(config_json) as f:
        config_dict = json.load(f)
    
    config = Alpamayo1_5Config(**config_dict)
    
    # Load VLM weights manually to avoid HF Hub repo_id validation
    logger.info(f"Loading VLM weights from {config_dir}")
    
    # Try to find safetensors or pytorch weight files in this directory
    weight_file = None
    if (config_dir / "model.safetensors").exists():
        weight_file = config_dir / "model.safetensors"
        logger.info(f"Loading from safetensors: {weight_file}")
        weights = load_safetensors_file(str(weight_file))
    elif (config_dir / "model.safetensors.index.json").exists():
        # Sharded safetensors
        index_file = config_dir / "model.safetensors.index.json"
        with open(index_file) as f:
            index = json.load(f)
        weights = {}
        for shard_file in set(index["weight_map"].values()):
            shard_path = config_dir / shard_file
            logger.info(f"Loading shard: {shard_file}")
            weights.update(load_safetensors_file(str(shard_path)))
    elif (config_dir / "pytorch_model.bin").exists():
        weight_file = config_dir / "pytorch_model.bin"
        logger.info(f"Loading from pytorch_model.bin: {weight_file}")
        weights = torch.load(str(weight_file), map_location="cpu")
    elif (config_dir / "pytorch_model.bin.index.json").exists():
        # Sharded pytorch weights
        index_file = config_dir / "pytorch_model.bin.index.json"
        with open(index_file) as f:
            index = json.load(f)
        weights = {}
        for shard_file in set(index["weight_map"].values()):
            shard_path = config_dir / shard_file
            logger.info(f"Loading shard: {shard_file}")
            weights.update(torch.load(str(shard_path), map_location="cpu"))
    else:
        raise FileNotFoundError(f"No weight files found in {config_dir}")
    
    # Instantiate base Alpamayo1_5 with a VLM config aligned to checkpoint tensor shapes.
    logger.info("Instantiating Alpamayo1_5 model with checkpoint-aligned VLM config")

    base_vlm_name_or_path = config_dict.get("vlm_name_or_path") or config.vlm_name_or_path
    qwen_cfg = Qwen3VLConfig.from_pretrained(
        base_vlm_name_or_path,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    # Derive critical architecture fields from local checkpoint tensors.
    embed_w = weights.get("vlm.model.language_model.embed_tokens.weight")
    gate_w = weights.get("vlm.model.language_model.layers.0.mlp.gate_proj.weight")
    kproj_w = weights.get("vlm.model.language_model.layers.0.self_attn.k_proj.weight")
    vis_merger_w = weights.get("vlm.model.visual.merger.linear_fc2.weight")

    if embed_w is not None:
        qwen_cfg.text_config.vocab_size = int(embed_w.shape[0])
        qwen_cfg.vocab_size = int(embed_w.shape[0])
        qwen_cfg.text_config.hidden_size = int(embed_w.shape[1])

    if gate_w is not None:
        qwen_cfg.text_config.intermediate_size = int(gate_w.shape[0])

    # Infer number of layers from checkpoint key names.
    layer_ids = []
    for key in weights.keys():
        m = re.match(r"^vlm\.model\.language_model\.layers\.(\d+)\.", key)
        if m:
            layer_ids.append(int(m.group(1)))
    if layer_ids:
        qwen_cfg.text_config.num_hidden_layers = max(layer_ids) + 1

    # Infer KV heads from k_proj rows and current head_dim.
    if kproj_w is not None and qwen_cfg.text_config.num_attention_heads > 0:
        hidden = int(qwen_cfg.text_config.hidden_size)
        n_heads = int(qwen_cfg.text_config.num_attention_heads)
        if hidden % n_heads == 0:
            head_dim = hidden // n_heads
            kv_proj_rows = int(kproj_w.shape[0])
            if head_dim > 0 and kv_proj_rows % head_dim == 0:
                qwen_cfg.text_config.num_key_value_heads = kv_proj_rows // head_dim

    # Vision merger output must match language hidden size for placeholder scatter.
    if vis_merger_w is not None and hasattr(qwen_cfg, "vision_config"):
        vision_out = int(vis_merger_w.shape[0])
        if hasattr(qwen_cfg.vision_config, "out_hidden_size"):
            qwen_cfg.vision_config.out_hidden_size = vision_out

    _ensure_qwen3vl_rope_scaling(qwen_cfg)
    vlm = Qwen3VLForConditionalGeneration(qwen_cfg)
    alpamayo = Alpamayo1_5(
        config,
        pretrained_modules={"vlm": vlm},
        original_vocab_size=qwen_cfg.text_config.vocab_size,
    )
    
    # Load only shape-compatible parameters to avoid RuntimeError on size mismatch.
    model_state = alpamayo.state_dict()
    compatible_weights: dict[str, torch.Tensor] = {}
    skipped_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    skipped_missing_in_model: list[str] = []

    for key, tensor in weights.items():
        if key not in model_state:
            skipped_missing_in_model.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(tensor.shape):
            skipped_mismatch.append((key, tuple(tensor.shape), tuple(model_state[key].shape)))
            continue
        compatible_weights[key] = tensor

    load_result = alpamayo.load_state_dict(compatible_weights, strict=False)

    logger.info(
        "Loaded %d/%d checkpoint tensors into local Alpamayo model",
        len(compatible_weights),
        len(weights),
    )
    if skipped_mismatch:
        logger.warning(
            "Skipped %d tensors due to shape mismatch; examples: %s",
            len(skipped_mismatch),
            skipped_mismatch[:5],
        )
    if skipped_missing_in_model:
        logger.warning(
            "Skipped %d tensors not present in current model; examples: %s",
            len(skipped_missing_in_model),
            skipped_missing_in_model[:5],
        )
    if getattr(load_result, "missing_keys", None):
        logger.warning("Model missing %d keys after load", len(load_result.missing_keys))
    if getattr(load_result, "unexpected_keys", None):
        logger.warning("Unexpected %d keys after load", len(load_result.unexpected_keys))

    logger.info("Successfully loaded compatible VLM weights")
    
    return alpamayo


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
                resolved_vlm_name_or_path = _resolve_vlm_model_path(config.vlm_name_or_path)
                is_local = Path(resolved_vlm_name_or_path).exists() or resolved_vlm_name_or_path.startswith("/")
                logger.info(f"Loading VLM from: {resolved_vlm_name_or_path} (local={is_local})")
                
                # Use direct local loader to avoid HF Hub validation entirely
                if is_local:
                    logger.info("Using direct local loader to bypass HF Hub validation")
                    loaded_alpamayo = _load_vlm_from_local_path(
                        resolved_vlm_name_or_path,
                        dtype=_resolve_torch_dtype(config.model_dtype),
                        attn_impl=config.attn_implementation,
                    )
                else:
                    # Fallback to HF Hub for non-local paths (e.g., HF Hub repo IDs)
                    logger.info(f"Using HF Hub from_pretrained for: {resolved_vlm_name_or_path}")
                    loaded_alpamayo = Alpamayo1_5.from_pretrained(
                        resolved_vlm_name_or_path,
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
                    resolved_vlm_name_or_path = _resolve_vlm_model_path(config.vlm_name_or_path)
                    is_local = Path(resolved_vlm_name_or_path).exists() or resolved_vlm_name_or_path.startswith("/")
                    
                    if is_local:
                        logger.info("Retrying local loader without FA2")
                        loaded_alpamayo = _load_vlm_from_local_path(
                            resolved_vlm_name_or_path,
                            dtype=_resolve_torch_dtype(config.model_dtype),
                            attn_impl=None,
                        )
                    else:
                        loaded_alpamayo = Alpamayo1_5.from_pretrained(
                            resolved_vlm_name_or_path,
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
                    _resolve_vlm_model_path(config.vlm_name_or_path),
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

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Enable gradient checkpointing for the model.

        Args:
            gradient_checkpointing_kwargs: Additional keyword arguments for gradient checkpointing.
        """
        enabled = False
        # Try enabling on the main VLM if available
        if hasattr(self, "vlm") and hasattr(self.vlm, "gradient_checkpointing_enable"):
            try:
                self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
                logger.info("Enabled gradient checkpointing on VLM")
                enabled = True
            except Exception as e:
                logger.warning(f"Failed to enable gradient checkpointing on VLM: {e}")

        # Also attempt to enable on other large submodules (expert, action_in_proj, action_out_proj)
        for name in ("expert", "action_in_proj", "action_out_proj", "diffusion"):
            module = getattr(self, name, None)
            if module is None:
                continue
            if hasattr(module, "gradient_checkpointing_enable"):
                try:
                    module.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
                    logger.info(f"Enabled gradient checkpointing on {name}")
                    enabled = True
                except Exception as e:
                    logger.warning(f"Failed to enable gradient checkpointing on {name}: {e}")

        if not enabled:
            # Fallback: try enabling on any child module that supports it
            for m in self.modules():
                if hasattr(m, "gradient_checkpointing_enable"):
                    try:
                        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
                        logger.info(f"Enabled gradient checkpointing on submodule {m.__class__.__name__}")
                        enabled = True
                    except Exception:
                        pass

        if not enabled:
            logger.warning(
                f"No submodules in {self.__class__.__name__} exposed gradient_checkpointing_enable()."
            )

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing for the model."""
        disabled = False
        if hasattr(self, "vlm") and hasattr(self.vlm, "gradient_checkpointing_disable"):
            try:
                self.vlm.gradient_checkpointing_disable()
                logger.info("Disabled gradient checkpointing on VLM")
                disabled = True
            except Exception as e:
                logger.warning(f"Failed to disable gradient checkpointing on VLM: {e}")

        for name in ("expert", "action_in_proj", "action_out_proj", "diffusion"):
            module = getattr(self, name, None)
            if module is None:
                continue
            if hasattr(module, "gradient_checkpointing_disable"):
                try:
                    module.gradient_checkpointing_disable()
                    logger.info(f"Disabled gradient checkpointing on {name}")
                    disabled = True
                except Exception as e:
                    logger.warning(f"Failed to disable gradient checkpointing on {name}: {e}")

        if not disabled:
            for m in self.modules():
                if hasattr(m, "gradient_checkpointing_disable"):
                    try:
                        m.gradient_checkpointing_disable()
                        logger.info(f"Disabled gradient checkpointing on submodule {m.__class__.__name__}")
                        disabled = True
                    except Exception:
                        pass

        if not disabled:
            logger.warning(
                f"No submodules in {self.__class__.__name__} exposed gradient_checkpointing_disable()."
            )

    def _compute_next_token_loss(
        self,
        outputs: Any,
        labels: torch.Tensor,
        labels_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute next-token prediction loss following the alpamayo_r1 approach.
        
        This method implements the loss computation from the original alpamayo:
        - Shifts labels left by 1 (predict next token)
        - Applies labels_mask to select loss-contributing positions
        - Computes cross-entropy only on selected positions
        
        Args:
            outputs: Model outputs containing logits of shape (B, L, V)
            labels: Token indices of shape (B, L)
            labels_mask: Boolean mask of shape (B, L), True where loss should be computed
            
        Returns:
            Scalar loss tensor
        """
        if labels_mask is None:
            labels_mask = torch.ones_like(labels, dtype=torch.bool)
        
        mask_sum = labels_mask[:, 1:].sum().item()
        if mask_sum == 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"[LOSS_DEBUG] labels_mask is all zeros! shape={labels_mask.shape} "
                f"labels_shape={labels.shape} mask_ratio={labels_mask.float().mean():.4f} "
                f"mask_sum_full={labels_mask.sum().item()} mask_sum_shifted={mask_sum}"
            )
            return torch.tensor(0.0, device=labels.device, dtype=outputs.logits.dtype)
        
        # Shift labels left by 1 (next token prediction)
        shift_labels = labels[..., 1:]
        shift_logits = outputs.logits[..., :-1, :].clone()
        
        # Apply mask: select only positions where loss should be computed
        shift_labels = shift_labels[labels_mask[:, 1:]].contiguous()
        shift_logits = shift_logits[labels_mask[:, 1:]].contiguous().float()
        
        # Ensure device alignment
        shift_labels = shift_labels.to(shift_logits.device)
        
        # Compute cross-entropy loss
        loss = torch.nan_to_num(
            F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=IGNORE_INDEX,
                reduction="mean"
            ),
            nan=0.0,
        )
        return loss

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Forward pass for Stage1 SFT with custom loss computation.

        The collator feeds standard multimodal inputs plus optional trajectory
        history tensors. When trajectory history is provided, the prompt tokens
        are fused in the same way as inference-time inputs.
        
        Args:
            input_ids: [B, L]
            attention_mask: [B, L], optional
            labels: [B, L], token indices for loss computation
            labels_mask: [B, L], boolean mask indicating which positions contribute to loss
            ego_history_xyz: [B, n_traj, T, 3], optional trajectory history
            ego_history_rot: [B, n_traj, T, 3], optional trajectory history
            **kwargs: Additional arguments passed to VLM
        """

        # Input-shape diagnostic logging removed to avoid noisy output during training

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

        # Forward through VLM: pass labels=None to prevent default loss computation
        # We will override the loss with our custom _compute_next_token_loss
        vlm_outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # Don't compute loss in VLM; we'll do it manually
            **kwargs,
        )

        # Compute custom loss using labels_mask if provided
        if labels is not None and labels_mask is not None:
            vlm_outputs.loss = self._compute_next_token_loss(vlm_outputs, labels, labels_mask)
        elif labels is not None:
            # Fallback: compute loss without mask (use default behavior)
            vlm_outputs.loss = self._compute_next_token_loss(vlm_outputs, labels, None)

        return vlm_outputs

    def prepare_inputs_for_generation(self, *args, **kwargs) -> dict[str, Any]:
        # Delegate to underlying VLM if possible
        if hasattr(self.vlm, "prepare_inputs_for_generation"):
            return self.vlm.prepare_inputs_for_generation(*args, **kwargs)

        return super().prepare_inputs_for_generation(*args, **kwargs)
