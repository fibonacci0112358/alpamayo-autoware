#!/usr/bin/env python3
"""Merge a Stage-1 LoRA adapter into a full Alpamayo1.5 checkpoint.

This script mirrors the behavior of the upstream `merge_stage1_lora.py` so
that the merged output is a full HF-style checkpoint directory that can be
consumed by Stage2 training and the TRT export pipeline.

Example:

  python finetune/sft/scripts/merge_lora_and_save.py \
    --base-model-dir /path/to/base_model \
    --adapter-dir /path/to/lora_adapter \
    --output-dir /path/to/merged_checkpoint
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model-dir", required=True, help="Base Alpamayo checkpoint dir")
    p.add_argument("--adapter-dir", required=True, help="Stage-1 LoRA adapter dir")
    p.add_argument("--output-dir", required=True, help="Merged checkpoint output dir")
    p.add_argument(
        "--max-shard-size",
        default="4GB",
        help="Maximum safetensors shard size when saving the merged checkpoint.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the output directory.",
    )
    return p.parse_args()


_WEIGHT_SUFFIXES = (".bin", ".safetensors")


def _load_config_json(model_dir: Path) -> dict | None:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _has_stage1_architecture(config_data: dict | None) -> bool:
    if not isinstance(config_data, dict):
        return False
    return bool(config_data.get("action_space_cfg"))


def _is_weight_file(name: str) -> bool:
    if name.endswith(_WEIGHT_SUFFIXES):
        return True
    return name.endswith(".index.json") and ("safetensors" in name or "pytorch_model" in name)


def _copy_non_weight_files(src_dir: Path, dst_dir: Path) -> list[str]:
    copied: list[str] = []
    for path in sorted(src_dir.iterdir()):
        if not path.is_file():
            continue
        if _is_weight_file(path.name):
            continue
        shutil.copy2(path, dst_dir / path.name)
        copied.append(path.name)
    return copied


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_model_dir)
    adapter_dir = Path(args.adapter_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the explicitly provided base model directory, but if it is a raw
    # Alpamayo VLM checkpoint without the Stage1 action-stack config, fall back
    # to the parent directory of the adapter. That parent is the Stage1 full
    # checkpoint produced by this repo's training entrypoint.
    effective_base_dir = base_dir
    base_cfg = _load_config_json(base_dir)
    if not _has_stage1_architecture(base_cfg):
        adapter_parent = adapter_dir.parent
        adapter_parent_cfg = _load_config_json(adapter_parent)
        if _has_stage1_architecture(adapter_parent_cfg):
            print(
                f"Base model dir {base_dir} does not look like a Stage1 checkpoint; "
                f"falling back to adapter parent {adapter_parent}."
            )
            effective_base_dir = adapter_parent

    if not (effective_base_dir / "config.json").exists():
        print(f"Missing config.json in base_model_dir: {effective_base_dir}")
        return 2
    if not (adapter_dir / "adapter_config.json").exists():
        print(f"Missing adapter_config.json in adapter_dir: {adapter_dir}")
        return 2

    try:
        from finetune.sft.models.sft_alpamayo_1_5 import TrainableAlpamayo1_5
    except Exception as e:
        print("Failed to import TrainableAlpamayo1_5:", e)
        return 1

    try:
        from peft import PeftModel
    except Exception as e:
        print("peft import failed:", e)
        return 1

    print(f"Loading base model from {effective_base_dir} ...")
    model = TrainableAlpamayo1_5.from_pretrained(
        str(effective_base_dir),
        dtype="auto",
        cotrain_vlm=False,
        stage1_vlm_checkpoint_path=None,
    )

    print(f"Loading LoRA adapter from {adapter_dir} ...")
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    print("Merging adapter into base model ...")
    try:
        merged = model.merge_and_unload()
    except Exception as e:
        print("merge_and_unload failed, attempting state_dict fallback:", e)
        try:
            sd = model.state_dict()
            base = TrainableAlpamayo1_5.from_pretrained(
                str(effective_base_dir),
                dtype="auto",
                cotrain_vlm=False,
                stage1_vlm_checkpoint_path=None,
            )
            base.load_state_dict(sd, strict=False)
            merged = base
        except Exception as e2:
            print("Fallback merge failed:", e2)
            return 1

    print(f"Saving merged checkpoint to {out_dir} ...")
    try:
        merged.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size=args.max_shard_size)
    except TypeError:
        # Older HF versions may not accept max_shard_size
        merged.save_pretrained(str(out_dir), safe_serialization=True)

    copied = _copy_non_weight_files(base_dir, out_dir)
    print(f"Copied {len(copied)} non-weight files from base checkpoint.")
    print("Done. Use this directory as model.stage1_vlm_checkpoint_path for stage 2:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
