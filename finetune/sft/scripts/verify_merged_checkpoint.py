#!/usr/bin/env python3
"""Verify that a merged checkpoint can be loaded by Alpamayo1_5 inference class.

Usage:
  python finetune/sft/scripts/verify_merged_checkpoint.py --merged-dir /path/to/merged_checkpoint

This script attempts to call `Alpamayo1_5.from_pretrained` with the given
directory. It is a light-weight check to detect obvious incompatibilities.
"""

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merged-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    merged = Path(args.merged_dir)
    if not merged.exists():
        print("Merged checkpoint path not found:", merged)
        return 2

    try:
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    except Exception as e:
        print("Failed to import Alpamayo1_5:", e)
        return 1

    try:
        print("Attempting to load merged checkpoint from:", merged)
        model = Alpamayo1_5.from_pretrained(str(merged))
        print("Model loaded. Parameter count (approx):", sum(p.numel() for p in model.parameters()))
    except Exception as e:
        print("Failed to load model from merged checkpoint:", e)
        return 3

    print("Merged checkpoint verification succeeded.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
