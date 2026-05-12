#!/usr/bin/env python3
"""Inspect a .pt manifest used by Stage1/Stage2 SFT.

This utility prints a compact summary of the manifest and validates that the
first sample contains the tensor fields required by Stage2 training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


REQUIRED_PT_KEYS = (
    "image_frames",
    "camera_indices",
    "ego_history_xyz",
    "ego_history_rot",
    "ego_future_xyz",
    "ego_future_rot",
)


def _read_manifest(manifest_path: Path) -> list[dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "samples", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported manifest structure: {manifest_path}")


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _describe_tensor(value: torch.Tensor) -> str:
    return f"shape={tuple(value.shape)} dtype={value.dtype}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional root used to resolve relative .pt paths in the manifest",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="How many samples to inspect in detail",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print("Manifest not found:", manifest_path)
        return 2

    records = _read_manifest(manifest_path)
    print(f"manifest: {manifest_path}")
    print(f"records: {len(records)}")

    if not records:
        print("Manifest is empty")
        return 3

    root = Path(args.image_root) if args.image_root is not None else manifest_path.parent
    inspected = min(args.max_samples, len(records))

    for index, record in enumerate(records[:inspected]):
        print(f"\nsample[{index}]")
        print("  manifest keys:", sorted(record.keys()))
        file_field = record.get("file")
        if file_field is None:
            print("  missing file field")
            return 4

        pt_path = _resolve_path(root, str(file_field))
        print("  pt:", pt_path)
        if not pt_path.exists():
            print("  missing pt file")
            return 5

        payload = torch.load(pt_path, map_location="cpu")
        keys = sorted(payload.keys())
        print("  pt keys:", keys)

        missing = [key for key in REQUIRED_PT_KEYS if key not in payload]
        if missing:
            print("  missing required keys:", missing)
            return 6

        for key in REQUIRED_PT_KEYS:
            value = payload[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {_describe_tensor(value)}")
            else:
                print(f"  {key}: {type(value).__name__}")

    print("\nStage2 manifest inspection succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())