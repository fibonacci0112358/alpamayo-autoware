"""Copy existing HuggingFace snapshot of nvidia/Alpamayo-1.5-10B into the repository root.

Usage:
    python finetune/sft/scripts/copy_base_model_to_repo.py

This script will search the HuggingFace cache under ~/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B/snapshots/*
and copy the most recent snapshot directory to <repo_root>/Alpamayo-1.5-10B if not already present.
"""
from pathlib import Path
import shutil
import os

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
MODEL_NAMESPACE = "models--nvidia--Alpamayo-1.5-10B"
TARGET_NAME = "Alpamayo-1.5-10B"

candidates = list((CACHE_DIR / MODEL_NAMESPACE / "snapshots").glob("*")) if (CACHE_DIR / MODEL_NAMESPACE / "snapshots").exists() else []
if not candidates:
    print("No local HuggingFace snapshots found for nvidia/Alpamayo-1.5-10B in cache.")
    print("If you have a snapshot elsewhere, copy it to the repository root manually as ./Alpamayo-1.5-10B")
    raise SystemExit(1)

# pick most recent by mtime
candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
src = candidates[0]

target = REPO_ROOT / TARGET_NAME
if target.exists():
    print(f"Target {target} already exists; skipping copy.")
    print(f"If you want to refresh it, remove {target} and rerun this script.")
    raise SystemExit(0)

print(f"Copying base model from {src} to {target} ...")
shutil.copytree(src, target)
print("Copy complete.")
print(f"You can now run training with model.config.vlm_name_or_path={target}")
