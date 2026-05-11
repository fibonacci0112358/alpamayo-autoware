# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import hydra
import hydra.utils as hyu
from omegaconf import DictConfig, OmegaConf
from transformers import AutoProcessor, Trainer, TrainingArguments

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


def _build_dataset_from_cfg(
    cfg: DictConfig,
    manifest_path: str | Path | None,
    local_dir: str | Path | None,
    image_root: str | Path | None,
    default_num_frames_per_camera: int,
    chunk_ids,
    file_start,
    file_end,
):
    if local_dir:
        return PaiAvR1VlmSftDataset(
            local_dir=local_dir,
            chunk_ids=chunk_ids if chunk_ids is not None else _get(cfg, "data.chunk_ids"),
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

    print("Resolved config:\n", OmegaConf.to_yaml(cfg))
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
        print(f"Instantiated model parameters: {param_count:,}")
    except Exception:
        print("Model instantiated (could not count params)")

    manifest_path = _get(cfg, "data.manifest_path")
    local_dir = _get(cfg, "data.local_dir")
    train_chunk_ids = _get(cfg, "data.train_chunk_ids")
    valid_chunk_ids = _get(cfg, "data.valid_chunk_ids")
    train_file_start = _get(cfg, "data.train_file_start")
    train_file_end = _get(cfg, "data.train_file_end")
    valid_file_start = _get(cfg, "data.valid_file_start")
    valid_file_end = _get(cfg, "data.valid_file_end")
    if not manifest_path and not local_dir:
        print("\nNext: provide cfg.data.manifest_path or cfg.data.local_dir to run Stage1 training.")
        print("If you want, I can add a small-batch test harness to run forward/backward.")
        return

    image_root = _get(cfg, "data.image_root")
    default_num_frames_per_camera = int(_get(cfg, "data.num_frames_per_camera", 4))

    processor = AutoProcessor.from_pretrained(
        BASE_PROCESSOR_NAME,
        min_pixels=_get(cfg.model, "min_pixels"),
        max_pixels=_get(cfg.model, "max_pixels"),
    )
    processor.tokenizer = model.tokenizer

    dataset, use_pt = _build_dataset_from_cfg(
        cfg=cfg,
        manifest_path=manifest_path,
        local_dir=local_dir,
        image_root=image_root,
        default_num_frames_per_camera=default_num_frames_per_camera,
        chunk_ids=train_chunk_ids,
        file_start=train_file_start,
        file_end=train_file_end,
    )

    eval_dataset = None
    if valid_chunk_ids is not None or valid_file_start is not None or valid_file_end is not None:
        eval_dataset, _ = _build_dataset_from_cfg(
            cfg=cfg,
            manifest_path=manifest_path,
            local_dir=local_dir,
            image_root=image_root,
            default_num_frames_per_camera=default_num_frames_per_camera,
            chunk_ids=valid_chunk_ids,
            file_start=valid_file_start,
            file_end=valid_file_end,
        )

    collator = PtManifestCollator(processor=processor) if use_pt else PaiAvVlmSftCollator(processor=processor)

    training_cfg = cfg.get("training", {})
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

    eval_strategy = str(_get(training_cfg, "eval_strategy", "no"))
    eval_steps_val = _get(training_cfg, "eval_steps")
    eval_steps = int(eval_steps_val) if eval_steps_val is not None else None

    training_args = TrainingArguments(
        output_dir=str(_get(training_cfg, "output_dir", "outputs/stage1")),
        per_device_train_batch_size=int(_get(training_cfg, "per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(_get(training_cfg, "gradient_accumulation_steps", 1)),
        learning_rate=float(_get(training_cfg, "learning_rate", 1e-5)),
        num_train_epochs=float(_get(training_cfg, "num_train_epochs", 1.0)),
        max_steps=int(_get(training_cfg, "max_steps", -1)),
        logging_steps=int(_get(training_cfg, "logging_steps", 10)),
        save_steps=int(_get(training_cfg, "save_steps", 100)),
        save_total_limit=int(_get(training_cfg, "save_total_limit", 2)),
        bf16=bool(_get(training_cfg, "bf16", True)),
        fp16=bool(_get(training_cfg, "fp16", False)),
        dataloader_num_workers=int(_get(training_cfg, "dataloader_num_workers", 4)),
        remove_unused_columns=False,
        report_to=report_to,
        optim=str(_get(training_cfg, "optim", "adamw_torch")),
        warmup_ratio=float(_get(training_cfg, "warmup_ratio", 0.03)),
        evaluation_strategy=eval_strategy,
        eval_steps=eval_steps,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    # If eval_dataset exists, auto-enable "steps" evaluation strategy if not already set
    if eval_dataset is not None:
        trainer_kwargs['eval_dataset'] = eval_dataset
        # Regenerate training_args with eval_strategy="steps" if currently "no"
        if eval_strategy == "no":
            if eval_steps is None:
                eval_steps = max(1, len(dataset) // 10)  # Default: every 10% of training
            training_args.eval_strategy = "steps"
            training_args.eval_steps = eval_steps
            print(f"Auto-enabling evaluation: eval_strategy=steps, eval_steps={eval_steps}, eval_dataset_size={len(eval_dataset)}")

    trainer = Trainer(**trainer_kwargs)

    print(f"Starting training on {len(dataset)} samples -> {training_args.output_dir}")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    model.tokenizer.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    print(f"Saved checkpoint and tokenizer to {training_args.output_dir}")


if __name__ == "__main__":
    train()
