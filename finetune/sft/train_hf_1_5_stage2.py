#!/usr/bin/env python3
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
from omegaconf import DictConfig, OmegaConf
from transformers import AutoProcessor, Trainer, TrainingArguments
from torch.utils.data import ConcatDataset, Dataset

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
    """Build local directory paths from root and names (like Stage1)."""
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
    """Find manifest.json paths in the given folders (like Stage1)."""
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


def _resolve_stage1_checkpoint_path(path: str) -> str:
    """If `path` points to a directory without model files, try to
    locate the most-recent subdirectory that contains model artifacts
    (safetensors shards, pytorch_model.bin, or HF index files).
    This handles Stage1 runs that saved into timestamped subdirectories.
    """
    from pathlib import Path

    p = Path(path)
    if not p.exists() or not p.is_dir():
        return path

    # Common artifact names that mean this directory is a model repo
    marker_names = [
        "pytorch_model.bin",
        "model.safetensors",
        "model.safetensors.index.json",
        "tf_model.h5",
        "model.ckpt.index",
        "flax_model.msgpack",
    ]

    def contains_marker(d: Path) -> bool:
        try:
            for name in marker_names:
                if (d / name).exists():
                    return True
            # also accept sharded names like model-00001-of-00007.safetensors
            for f in d.iterdir():
                if f.is_file() and f.name.startswith("model-") and "safetensors" in f.name:
                    return True
        except Exception:
            return False
        return False

    # If base path already contains model artifacts, return it unchanged
    if contains_marker(p):
        return str(p)

    # Otherwise, search direct subdirectories and pick the newest valid one
    candidates = [d for d in p.iterdir() if d.is_dir()]
    valid = [d for d in candidates if contains_marker(d)]
    if valid:
        # choose most recently modified candidate
        chosen = max(valid, key=lambda d: d.stat().st_mtime)
        logger.info("Resolved Stage1 checkpoint from %s -> %s", path, str(chosen))
        return str(chosen)

    # no suitable subdir found; return original path
    return path


@hydra.main(version_base=None, config_path="configs", config_name="stage2")
def train(cfg: DictConfig) -> None:
    logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Stage2 entrypoint started")


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
            _ensure_qwen3vl_rope_scaling(model_cfg["config"])

            # If the config points at a Stage1 base output directory that
            # contains timestamped subfolders, resolve it to the actual
            # checkpoint folder so HF `from_pretrained` can find model files.
            try:
                vlm_path = getattr(model_cfg["config"], "vlm_name_or_path", None)
                if vlm_path:
                    resolved = _resolve_stage1_checkpoint_path(str(vlm_path))
                    if resolved != str(vlm_path):
                        logger.info("Auto-resolved Stage1 VLM path: %s -> %s", vlm_path, resolved)
                        model_cfg["config"].vlm_name_or_path = resolved
            except Exception:
                # non-fatal; continue with original path
                pass
        model_cls = hyu.get_class(target)
        model = model_cls(**model_cfg, **lora_kwargs)
    else:
        model = hyu.instantiate(cfg.model, _convert_="partial", **lora_kwargs)

    # Enable gradient checkpointing to reduce memory usage and prevent OOM
    # This helps with long training runs where memory accumulates over iterations
    gradient_checkpointing_enabled = bool(_get(cfg, "model.gradient_checkpointing", True))
    if gradient_checkpointing_enabled:
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.debug("Gradient checkpointing enabled")
            else:
                logger.warning("Model does not support gradient checkpointing")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")

    manifest_path = _get(cfg, "data.manifest_path")
    local_dir = _get(cfg, "data.local_dir")
    train_chunk_ids = _get(cfg, "data.train_chunk_ids")
    valid_chunk_ids = _get(cfg, "data.valid_chunk_ids")
    train_file_start = _get(cfg, "data.train_file_start")
    train_file_end = _get(cfg, "data.train_file_end")
    valid_file_start = _get(cfg, "data.valid_file_start")
    valid_file_end = _get(cfg, "data.valid_file_end")
    
    # If not provided directly, try to resolve from train_local_dir_names + local_dir_root (like Stage1)
    train_manifest_paths = None
    valid_manifest_paths = None
    if not manifest_path and not local_dir:
        local_dir_root = _get(cfg, "data.local_dir_root")
        train_local_dir_names = _get(cfg, "data.train_local_dir_names")
        valid_local_dir_names = _get(cfg, "data.valid_local_dir_names")
        if local_dir_root and train_local_dir_names:
            train_manifest_paths = _resolve_manifest_paths_from_folders(local_dir_root, train_local_dir_names)
        if local_dir_root and valid_local_dir_names:
            valid_manifest_paths = _resolve_manifest_paths_from_folders(local_dir_root, valid_local_dir_names)
    
    if not manifest_path and not local_dir and not train_manifest_paths:
        logger.warning("Next: provide cfg.data.manifest_path, cfg.data.local_dir, or cfg.data.train_local_dir_names/local_dir_root to run Stage2 training.")
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

    # Use all train_manifest_paths if available (like Stage1), otherwise fall back to single manifest_path
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
    if valid_chunk_ids is not None or valid_file_start is not None or valid_file_end is not None or valid_manifest_paths is not None:
        eval_manifest_source = valid_manifest_paths if valid_manifest_paths is not None else manifest_path
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
    report_to = _get(training_cfg, "report_to", [])
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

    if isinstance(report_to, (list, tuple)) and len(report_to) == 0:
        report_to = None

    eval_strategy = str(_get(training_cfg, "eval_strategy", "epoch"))
    save_strategy = str(_get(training_cfg, "save_strategy", "epoch"))
    eval_steps_val = _get(training_cfg, "eval_steps")
    eval_steps = int(eval_steps_val) if eval_steps_val is not None else None

    training_args = TrainingArguments(
        output_dir=str(_get(training_cfg, "output_dir", "outputs/stage2")),
        per_device_train_batch_size=int(_get(training_cfg, "per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(_get(training_cfg, "gradient_accumulation_steps", 1)),
        learning_rate=float(_get(training_cfg, "learning_rate", 1e-5)),
        num_train_epochs=float(_get(training_cfg, "num_train_epochs", 1.0)),
        max_steps=int(_get(training_cfg, "max_steps", -1)),
        logging_steps=int(_get(training_cfg, "logging_steps", 10)),
        save_steps=int(_get(training_cfg, "save_steps", 100)),
        save_strategy=save_strategy,
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
        deepspeed=_get(training_cfg, "deepspeed", None),
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    if eval_dataset is not None:
        trainer_kwargs['eval_dataset'] = eval_dataset

    trainer = Trainer(**trainer_kwargs)

    logger.info("Starting Stage2 training on %d samples -> %s", len(dataset), training_args.output_dir)
    trainer.train()
    trainer.save_model(training_args.output_dir)
    model.tokenizer.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    print(f"Saved checkpoint and tokenizer to {training_args.output_dir}")


if __name__ == "__main__":
    train()
