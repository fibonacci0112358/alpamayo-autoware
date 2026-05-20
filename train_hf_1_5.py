# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import sys
from pathlib import Path
from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import hydra
import hydra.utils as hyu
import time
import psutil
import os
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoProcessor, EarlyStoppingCallback, Trainer, TrainingArguments, TrainerCallback

from alpamayo1_5.config import Alpamayo1_5Config
from alpamayo1_5.helper import BASE_PROCESSOR_NAME
from alpamayo1_5.models.base_model import _ensure_qwen3vl_rope_scaling
from finetune.sft.data.pai_av_dataset import (
    PaiAvR1VlmSftDataset,
    PaiAvVlmSftCollator,
    PaiAvVlmSftDataset,
)
from finetune.sft.data.pt_manifest_dataset import PtManifestDataset, _read_manifest, PtManifestCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StepTimingCallback(TrainerCallback):
    """Monitor step time, memory, and GPU utilization progression."""
    
    def __init__(self):
        self.step_times = []
        self.step_memories = []
        self.step_start_time = None
        self.rank = int(os.getenv("RANK", "0"))
        self.is_rank0 = self.rank == 0
        self.process = psutil.Process(os.getpid())
        self.logging_interval = max(1, int(os.getenv("ALPAMAYO_DIAG_LOG_INTERVAL", "10")))
        self.last_logged_step = 0
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of a training step."""
        if not self.is_rank0:
            return
        self.step_start_time = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of a training step."""
        if not self.is_rank0:
            return
        if self.step_start_time is None:
            return
            
        step_time = time.time() - self.step_start_time
        rss_gb = self.process.memory_info().rss / 1e9
        vms_gb = self.process.memory_info().vms / 1e9
        
        self.step_times.append(step_time)
        self.step_memories.append(rss_gb)
        
        # Log on a configurable interval so diagnostics can run faster than Trainer logging.
        if state.global_step - self.last_logged_step >= self.logging_interval:
            avg_step_time = sum(self.step_times[-args.logging_steps:]) / min(len(self.step_times), args.logging_steps)
            
            # Detect linear time progression (memory leak indicator)
            if len(self.step_times) >= 2:
                time_diff = self.step_times[-1] - self.step_times[-2]
                mem_diff = self.step_memories[-1] - self.step_memories[-2]
            else:
                time_diff = 0
                mem_diff = 0
            
            logger.info(
                f"[TIMING] Step {state.global_step}: "
                f"step_time={step_time:.2f}s, "
                f"avg_recent={avg_step_time:.2f}s, "
                f"mem_rss={rss_gb:.2f}GB, "
                f"mem_vms={vms_gb:.2f}GB, "
                f"Δtime={time_diff:+.2f}s, "
                f"Δmem={mem_diff:+.3f}GB"
            )
            self.last_logged_step = state.global_step
            # Additional diagnostics: GPU memory and swap stats when available
            try:
                import gc
                import torch
                import psutil

                gpu_stats = None
                if torch.cuda.is_available():
                    # Log reserved and allocated memory per current device
                    dev = torch.cuda.current_device()
                    reserved = torch.cuda.memory_reserved(dev) / 1e9
                    allocated = torch.cuda.memory_allocated(dev) / 1e9
                    gpu_stats = f"gpu_reserved={reserved:.2f}GB gpu_allocated={allocated:.2f}GB"
                swap = psutil.swap_memory()
                logger.info(
                    "[DIAG] %s swap_total=%.2fGB swap_used=%.2fGB",
                    gpu_stats or "", swap.total / 1e9, swap.used / 1e9,
                )
                # Optional cleanup to test if memory fragmentation causes slowdown
                if os.getenv("ALPAMAYO_DIAG_POST_STEP_CLEANUP", "0") == "1":
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("[DIAG] Per-step gc.collect() and torch.cuda.empty_cache() executed")
            except Exception:
                pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Emit the raw scalar loss with higher precision than Trainer's default display."""
        if not self.is_rank0:
            return
        if not logs:
            return

        loss = logs.get("loss")
        if loss is None:
            return

        try:
            loss_value = float(loss)
        except Exception:
            return

        logger.info("[LOSS] step=%s loss=%.8f logs=%s", state.global_step, loss_value, {k: logs[k] for k in logs if k in {"loss", "learning_rate", "grad_norm", "epoch"}})


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SftTrainer(Trainer):
    """Custom Trainer for Stage1 SFT with proper labels_mask handling.
    
    This trainer passes labels_mask from the batch to the model's forward method,
    enabling the model to compute custom loss using only assistant token positions.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss while preserving labels_mask for custom loss calculation.
        
        The HF Trainer by default removes any non-standard keys from inputs before
        calling model.forward(). We extract labels_mask here and pass it explicitly.
        """
        # Extract labels_mask if present in the batch
        labels_mask = inputs.pop("labels_mask", None)
        
        # Diagnostic: log mask statistics periodically
        if labels_mask is not None and self.state.global_step % 50 == 0 and int(os.getenv("RANK", "0")) == 0:
            mask_ratio = labels_mask.float().mean().item()
            mask_sum = labels_mask[:, 1:].sum().item()
            logger.info(
                f"[MASK_DIAG] step={self.state.global_step} mask_shape={labels_mask.shape} "
                f"mask_ratio={mask_ratio:.4f} mask_sum_shifted={mask_sum} "
                f"labels_shape={inputs['labels'].shape if 'labels' in inputs else 'N/A'}"
            )
        
        # Call parent's forward pass
        outputs = model(**inputs)
        
        # If the model's forward already computed loss (using labels_mask),
        # that loss is in outputs.loss. Otherwise it will be None here.
        loss = outputs.loss if hasattr(outputs, "loss") else None
        
        # Fallback: if no loss was computed by the model, manually compute
        if loss is None:
            if "labels" in inputs:
                if hasattr(model, "_compute_next_token_loss") and labels_mask is not None:
                    loss = model._compute_next_token_loss(outputs, inputs["labels"], labels_mask)
                else:
                    # Minimal fallback to default cross-entropy
                    import torch.nn.functional as F
                    logits = outputs.logits
                    labels = inputs["labels"]
                    shift_labels = labels[..., 1:].contiguous()
                    shift_logits = logits[..., :-1, :].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.shape[-1]),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
        
        return (loss, outputs) if return_outputs else loss


def _get(cfg: DictConfig, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        value = cfg
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return default if value is None else value

    value = OmegaConf.select(cfg, key)
    return default if value is None else value


def _as_path_list(value: str | Path | Sequence[str | Path] | None) -> list[Path] | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value]


def _build_concat_dataset(datasets: list[Dataset]) -> Dataset:
    if not datasets:
        raise ValueError("At least one dataset is required")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def _resolve_local_dirs(
    local_dir_root: str | Path | None,
    local_dir_names: Sequence[str | Path] | None,
) -> list[Path] | None:
    if local_dir_names is None:
        return None

    names = list(local_dir_names)
    if not names:
        return None

    if local_dir_root is None:
        return [Path(item) for item in names]

    root = Path(local_dir_root)
    return [root / Path(item) for item in names]


def _resolve_manifest_paths_from_folders(
    local_dir_root: str | Path | None,
    local_dir_names: Sequence[str | Path] | None,
) -> list[Path] | None:
    local_dirs = _resolve_local_dirs(local_dir_root, local_dir_names)
    if not local_dirs:
        return None

    manifest_paths: list[Path] = []
    for local_dir in local_dirs:
        if local_dir.is_file() and local_dir.name == "manifest.json":
            manifest_paths.append(local_dir)
            continue

        candidates = sorted(local_dir.glob("**/manifest.json"))
        if not candidates:
            raise FileNotFoundError(f"No manifest.json found under {local_dir}")
        manifest_paths.append(candidates[0])

    return manifest_paths


def _build_dataset_from_cfg(
    cfg: DictConfig,
    manifest_path: str | Path | None,
    local_dir: str | Path | None,
    image_root: str | Path | None,
    default_num_frames_per_camera: int,
    include_camera_ids: bool,
    include_frame_nums: bool,
    use_nav_prompt: bool,
    chunk_ids,
    file_start,
    file_end,
):
    manifest_paths = _as_path_list(manifest_path)
    if manifest_paths:
        datasets: list[Dataset] = []
        use_pt: bool | None = None
        for resolved_manifest_path in manifest_paths:
            records = _read_manifest(resolved_manifest_path)
            resolved_use_pt = bool(
                records and isinstance(records[0].get("file"), str) and records[0]["file"].lower().endswith(".pt")
            )
            dataset_cls = PtManifestDataset if resolved_use_pt else PaiAvVlmSftDataset
            dataset = dataset_cls(
                manifest_path=resolved_manifest_path,
                image_root=image_root,
                default_num_frames_per_camera=default_num_frames_per_camera,
                include_camera_ids=include_camera_ids,
                include_frame_nums=include_frame_nums,
                use_nav_prompt=use_nav_prompt,
                chunk_ids=chunk_ids if chunk_ids is not None else _get(cfg, "data.chunk_ids"),
                file_start=file_start,
                file_end=file_end,
            )
            datasets.append(dataset)
            if use_pt is None:
                use_pt = resolved_use_pt
            elif use_pt != resolved_use_pt:
                raise ValueError("Mixed manifest types are not supported in one concatenated dataset")
        return _build_concat_dataset(datasets), bool(use_pt)

    local_dirs = _as_path_list(local_dir)
    if local_dirs:
        datasets: list[Dataset] = []
        use_pt: bool | None = None
        for resolved_local_dir in local_dirs:
            dataset = PaiAvR1VlmSftDataset(
                local_dir=resolved_local_dir,
                chunk_ids=chunk_ids if chunk_ids is not None else _get(cfg, "data.chunk_ids"),
                include_camera_ids=include_camera_ids,
                include_frame_nums=include_frame_nums,
                use_nav_prompt=use_nav_prompt,
                use_default_keyframe=bool(_get(cfg, "data.use_default_keyframe", False)),
                features_metadata=str(_get(cfg, "data.features_metadata", "features.csv")),
                clip_index_metadata=str(_get(cfg, "data.clip_index_metadata", "clip_index.parquet")),
                num_history_steps=int(_get(cfg, "data.num_history_steps", 16)),
                num_future_steps=int(_get(cfg, "data.num_future_steps", 64)),
                time_step=float(_get(cfg, "data.time_step", 0.1)),
                num_frames_per_camera=default_num_frames_per_camera,
                nav_text=_get(cfg, "data.nav_text"),
                completion=_get(cfg, "data.completion"),
                file_start=file_start,
                file_end=file_end,
            )
            datasets.append(dataset)
            if use_pt is None:
                use_pt = False
        return _build_concat_dataset(datasets), bool(use_pt)

    if local_dir:
        return PaiAvR1VlmSftDataset(
            local_dir=local_dir,
            chunk_ids=chunk_ids if chunk_ids is not None else _get(cfg, "data.chunk_ids"),
            include_camera_ids=include_camera_ids,
            include_frame_nums=include_frame_nums,
            use_nav_prompt=use_nav_prompt,
            use_default_keyframe=bool(_get(cfg, "data.use_default_keyframe", False)),
            features_metadata=str(_get(cfg, "data.features_metadata", "features.csv")),
            clip_index_metadata=str(_get(cfg, "data.clip_index_metadata", "clip_index.parquet")),
            num_history_steps=int(_get(cfg, "data.num_history_steps", 16)),
            num_future_steps=int(_get(cfg, "data.num_future_steps", 64)),
            time_step=float(_get(cfg, "data.time_step", 0.1)),
            num_frames_per_camera=default_num_frames_per_camera,
            nav_text=_get(cfg, "data.nav_text"),
            completion=_get(cfg, "data.completion"),
            file_start=file_start,
            file_end=file_end,
        ), False

    if manifest_path is None:
        raise ValueError("manifest_path is required when local_dir is not set")

    records = _read_manifest(Path(manifest_path)) if isinstance(manifest_path, str) else _read_manifest(manifest_path)
    use_pt = bool(records and isinstance(records[0].get("file"), str) and records[0]["file"].lower().endswith(".pt"))

    dataset_cls = PtManifestDataset if use_pt else PaiAvVlmSftDataset
    return dataset_cls(
        manifest_path=manifest_path,
        image_root=image_root,
        default_num_frames_per_camera=default_num_frames_per_camera,
        include_camera_ids=include_camera_ids,
        include_frame_nums=include_frame_nums,
        use_nav_prompt=use_nav_prompt,
        chunk_ids=chunk_ids if chunk_ids is not None else _get(cfg, "data.chunk_ids"),
        file_start=file_start,
        file_end=file_end,
    ), use_pt


@hydra.main(version_base=None, config_path="configs", config_name="stage1")
def train(cfg: DictConfig) -> None:
    """Stage1 entrypoint for Alpamayo1.5.

    When `cfg.data.manifest_path` is present, this script runs a minimal
    PaiAV-style supervised fine-tuning loop using Hugging Face Trainer.
    Otherwise it only instantiates the model and prints the config.
    """

    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Stage1 entrypoint started")

    lora_kwargs = {}
    if OmegaConf.is_config(cfg) and _get(cfg, "lora.use_lora", False):
        lora_cfg = OmegaConf.to_container(cfg.lora, resolve=True)
        lora_kwargs["use_lora"] = True
        lora_kwargs["lora_config"] = lora_cfg

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if isinstance(model_cfg, dict):
        model_cfg = dict(model_cfg)
        target = model_cfg.pop("_target_", None)
        if target is None:
            raise ValueError("cfg.model._target_ is required")

        model_config = model_cfg.get("config")
        if isinstance(model_config, dict):
            model_cfg["config"] = Alpamayo1_5Config(**model_config)
            # Ensure rope_scaling is set for Qwen3VL compatibility
            _ensure_qwen3vl_rope_scaling(model_cfg["config"])
            # If a local copy of the base model exists at repo root, prefer it
            try:
                local_base = REPO_ROOT / "Alpamayo-1.5-10B"
                if getattr(model_cfg["config"], "vlm_name_or_path", None) is None and local_base.exists():
                    model_cfg["config"].vlm_name_or_path = str(local_base)
            except Exception:
                pass
            # Provide a sensible default action_space_cfg when not provided
            if getattr(model_cfg["config"], "action_space_cfg", None) is None:
                model_cfg["config"].action_space_cfg = {
                    "_target_": "alpamayo1_5.action_space.unicycle_accel_curvature.UnicycleAccelCurvatureActionSpace"
                }
            # Provide defaults for diffusion and action projection modules
            if getattr(model_cfg["config"], "diffusion_cfg", None) is None:
                model_cfg["config"].diffusion_cfg = {
                    "_target_": "alpamayo1_5.diffusion.flow_matching.FlowMatching",
                }
            if getattr(model_cfg["config"], "action_in_proj_cfg", None) is None:
                model_cfg["config"].action_in_proj_cfg = {
                    "_target_": "alpamayo1_5.models.action_in_proj.PerWaypointActionInProjV2",
                }
            if getattr(model_cfg["config"], "action_out_proj_cfg", None) is None:
                model_cfg["config"].action_out_proj_cfg = {
                    "_target_": "torch.nn.Linear",
                }

        model_cls = hyu.get_class(target)
        model = model_cls(**model_cfg, **lora_kwargs)
    else:
        model = hyu.instantiate(cfg.model, _convert_="partial", **lora_kwargs)

    try:
        param_count = sum(p.numel() for p in model.parameters())
        logger.info("Instantiated model parameters: %d", param_count)
    except Exception:
        logger.debug("Model instantiated (could not count params)")

    # Enable gradient checkpointing to reduce memory usage and prevent OOM
    # This helps with long training runs where memory accumulates over iterations
    gradient_checkpointing_enabled = bool(_get(cfg, "model.gradient_checkpointing", True))
    if gradient_checkpointing_enabled:
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("Model does not support gradient checkpointing")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")

    manifest_path = _get(cfg, "data.manifest_path")
    local_dir = _get(cfg, "data.local_dir")
    train_local_dir_names = _get(cfg, "data.train_local_dir_names")
    valid_local_dir_names = _get(cfg, "data.valid_local_dir_names")
    local_dir_root = _get(cfg, "data.local_dir_root")
    train_chunk_ids = _get(cfg, "data.train_chunk_ids")
    valid_chunk_ids = _get(cfg, "data.valid_chunk_ids")
    train_file_start = _get(cfg, "data.train_file_start")
    train_file_end = _get(cfg, "data.train_file_end")
    valid_file_start = _get(cfg, "data.valid_file_start")
    valid_file_end = _get(cfg, "data.valid_file_end")
    train_manifest_paths = _resolve_manifest_paths_from_folders(local_dir_root, train_local_dir_names)
    valid_manifest_paths = _resolve_manifest_paths_from_folders(local_dir_root, valid_local_dir_names)

    if not manifest_path and not local_dir and not train_manifest_paths:
        logger.warning("Next: provide cfg.data.manifest_path, cfg.data.local_dir, or cfg.data.train_local_dir_names/local_dir_root to run Stage1 training.")
        return

    image_root = _get(cfg, "data.image_root")
    default_num_frames_per_camera = int(_get(cfg, "data.num_frames_per_camera", 4))
    include_camera_ids = bool(_get(cfg, "model.include_camera_ids", False))
    include_frame_nums = bool(_get(cfg, "model.include_frame_nums", False))
    use_nav_prompt = bool(_get(cfg, "data.use_nav_prompt", False))

    processor = AutoProcessor.from_pretrained(
        BASE_PROCESSOR_NAME,
        min_pixels=_get(cfg.model, "min_pixels"),
        max_pixels=_get(cfg.model, "max_pixels"),
    )
    processor.tokenizer = model.tokenizer

    dataset_manifest_source = train_manifest_paths if train_manifest_paths is not None else manifest_path
    dataset, use_pt = _build_dataset_from_cfg(
        cfg=cfg,
        manifest_path=dataset_manifest_source,
        local_dir=None if train_manifest_paths is not None else local_dir,
        image_root=image_root,
        default_num_frames_per_camera=default_num_frames_per_camera,
        include_camera_ids=include_camera_ids,
        include_frame_nums=include_frame_nums,
        use_nav_prompt=use_nav_prompt,
        chunk_ids=train_chunk_ids,
        file_start=train_file_start,
        file_end=train_file_end,
    )

    eval_dataset = None
    eval_manifest_source = valid_manifest_paths if valid_manifest_paths is not None else manifest_path
    if valid_chunk_ids is not None or valid_file_start is not None or valid_file_end is not None or valid_manifest_paths is not None:
        eval_dataset, _ = _build_dataset_from_cfg(
            cfg=cfg,
            manifest_path=eval_manifest_source,
            local_dir=None if valid_manifest_paths is not None else local_dir,
            image_root=image_root,
            default_num_frames_per_camera=default_num_frames_per_camera,
            include_camera_ids=include_camera_ids,
            include_frame_nums=include_frame_nums,
            use_nav_prompt=use_nav_prompt,
            chunk_ids=valid_chunk_ids,
            file_start=valid_file_start,
            file_end=valid_file_end,
        )

    collator = PtManifestCollator(processor=processor) if use_pt else PaiAvVlmSftCollator(processor=processor)

    training_cfg = cfg.get("training", {})
    try:
        use_lora_flag = bool(_get(cfg, "lora.use_lora", False))
    except Exception:
        use_lora_flag = False

    report_to = _get(training_cfg, "report_to", [])
    # Handle OmegaConf ListConfig and strings
    try:
        from omegaconf import ListConfig
    except Exception:
        ListConfig = None

    if report_to is None:
        report_to = []
    elif ListConfig is not None and isinstance(report_to, ListConfig):
        report_to = list(report_to)
    elif isinstance(report_to, str):
        report_to = [report_to]

    # Transformers Trainer expects `report_to=None` when no integrations
    if isinstance(report_to, (list, tuple)) and len(report_to) == 0:
        report_to = None

    eval_strategy = str(_get(training_cfg, "eval_strategy", "epoch"))
    save_strategy = str(_get(training_cfg, "save_strategy", "epoch"))
    eval_steps_val = _get(training_cfg, "eval_steps")
    eval_steps = int(eval_steps_val) if eval_steps_val is not None else None

    # Add timestamp to output_dir to preserve multiple runs.
    # When running under torchrun / distributed training, ensure a single
    # timestamp is used for all ranks to avoid multiple output directories
    # being created by each worker when they start at slightly different
    # wall-clock seconds.
    output_dir_base = str(_get(training_cfg, "output_dir", "outputs/stage1"))
    run_timestamp = None
    try:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            # Use a distributed broadcast so rank 0 picks the timestamp
            # and all other ranks receive the exact same integer seconds.
            import torch.distributed as dist

            if not dist.is_available():
                raise RuntimeError("torch.distributed not available")
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    init_method="env://",
                )

            if rank == 0:
                ts_int = int(time.time())
                ts_tensor = torch.tensor([ts_int], dtype=torch.long)
            else:
                ts_tensor = torch.tensor([0], dtype=torch.long)

            dist.broadcast(ts_tensor, src=0)
            run_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(int(ts_tensor.item())))
        else:
            run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    except Exception:
        # Fall back to local timestamp if distributed sync fails for any reason
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_dir_with_timestamp = f"{output_dir_base}_{run_timestamp}"
    logger.info(f"Output directory with timestamp: {output_dir_with_timestamp}")

    training_args = TrainingArguments(
        output_dir=output_dir_with_timestamp,
        per_device_train_batch_size=int(_get(training_cfg, "per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(_get(training_cfg, "gradient_accumulation_steps", 1)),
        learning_rate=float(_get(training_cfg, "learning_rate", 1e-5)),
        num_train_epochs=float(_get(training_cfg, "num_train_epochs", 1.0)),
        max_steps=int(_get(training_cfg, "max_steps", -1)),
        logging_steps=int(_get(training_cfg, "logging_steps", 10)),
        save_steps=int(_get(training_cfg, "save_steps", 100)) if not use_lora_flag else int(1e9),
        save_total_limit=int(_get(training_cfg, "save_total_limit", 2)),
        bf16=bool(_get(training_cfg, "bf16", True)),
        fp16=bool(_get(training_cfg, "fp16", False)),
        dataloader_num_workers=int(_get(training_cfg, "dataloader_num_workers", 4)),
        dataloader_pin_memory=bool(_get(training_cfg, "dataloader_pin_memory", True)),
        dataloader_persistent_workers=bool(_get(training_cfg, "dataloader_persistent_workers", True)),
        remove_unused_columns=False,
        report_to=report_to,
        optim=str(_get(training_cfg, "optim", "adamw_torch")),
        warmup_ratio=float(_get(training_cfg, "warmup_ratio", 0.03)),
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        deepspeed=_get(training_cfg, "deepspeed", None),
    )

    if use_lora_flag:
        # LoRA training only needs the adapter at the end; avoid writing full
        # model checkpoints during training to keep disk usage bounded.
        training_args.save_strategy = "no"

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    # Respect explicit config: do not auto-enable eval when eval_strategy is "no".
    if eval_dataset is not None and eval_strategy != "no":
        trainer_kwargs['eval_dataset'] = eval_dataset
    elif eval_dataset is not None and eval_strategy == "no":
        logger.warning("Evaluation dataset is available but eval_strategy=no; skipping evaluation (eval_dataset_size=%d)", len(eval_dataset))

    callbacks = [StepTimingCallback()]
    early_stopping_patience = _get(training_cfg, "early_stopping_patience", None)
    if early_stopping_patience is not None and int(early_stopping_patience) > 0:
        if eval_strategy == "no" or eval_dataset is None:
            logger.warning("Early stopping is configured but disabled because eval is unavailable (eval_strategy=no or no eval_dataset)")
        elif use_lora_flag:
            print("Early stopping is configured but disabled for LoRA mode because save_strategy=no")
        else:
            metric_for_best_model = str(_get(training_cfg, "metric_for_best_model", "eval_loss"))
            greater_is_better_cfg = _get(training_cfg, "greater_is_better", None)
            if greater_is_better_cfg is None:
                greater_is_better = not metric_for_best_model.endswith("loss")
            else:
                greater_is_better = bool(greater_is_better_cfg)

            training_args.load_best_model_at_end = bool(_get(training_cfg, "load_best_model_at_end", True))
            training_args.metric_for_best_model = metric_for_best_model
            training_args.greater_is_better = greater_is_better

            early_stopping_threshold = float(_get(training_cfg, "early_stopping_threshold", 0.0))
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=int(early_stopping_patience),
                    early_stopping_threshold=early_stopping_threshold,
                )
            )
            print(
                "Early stopping enabled: "
                f"patience={int(early_stopping_patience)}, "
                f"threshold={early_stopping_threshold}, "
                f"metric={metric_for_best_model}, "
                f"greater_is_better={greater_is_better}"
            )

    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = SftTrainer(**trainer_kwargs)

    print(f"Starting training on {len(dataset)} samples -> {training_args.output_dir}")
    # Optional gated torch.profiler for short-run distributed profiling
    if os.getenv("ALPAMAYO_DIAG_TORCHPROF", "0") == "1":
        try:
            log_dir = os.getenv("ALPAMAYO_DIAG_TORCHPROF_DIR", "./profiler_traces")
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            ):
                trainer.train()
        except Exception as e:
            logger.warning(f"Torch profiler unavailable or failed: {e}; running without profiler")
            trainer.train()
    else:
        trainer.train()
    if use_lora_flag:
        # Save model weights, config, and PEFT adapter for subsequent merging
        trainer.save_model(training_args.output_dir)
        model.tokenizer.save_pretrained(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
        try:
            from peft import PeftModel

            adapter_dir = os.path.join(str(training_args.output_dir), "lora_adapter")
            os.makedirs(adapter_dir, exist_ok=True)

            vlm = getattr(model, "vlm", None)
            if vlm is not None and isinstance(vlm, PeftModel):
                vlm.save_pretrained(adapter_dir, safe_serialization=True)
                print(f"Saved LoRA adapter to {adapter_dir}")
            elif isinstance(model, PeftModel):
                model.save_pretrained(adapter_dir, safe_serialization=True)
                print(f"Saved LoRA adapter to {adapter_dir}")
            else:
                print("Warning: lora.use_lora=true but no PeftModel was found; adapter not saved")
        except Exception as e:
            print(f"Exception during LoRA adapter save: {e}")
        print(f"Saved LoRA checkpoint, adapter, and tokenizer to {training_args.output_dir}")
    else:
        trainer.save_model(training_args.output_dir)
        model.tokenizer.save_pretrained(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
        print(f"Saved checkpoint and tokenizer to {training_args.output_dir}")


if __name__ == "__main__":
    train()
