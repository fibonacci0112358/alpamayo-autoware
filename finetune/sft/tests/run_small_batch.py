#!/usr/bin/env python3
"""Run a minimal small-batch forward+backward test.

This script is intended to validate the training loop mechanics inside this
repository without requiring a full Alpamayo1.5 model or external training
repo. It runs a tiny toy model by default. Use `--use-vlm` to attempt to
instantiate `TrainableAlpamayo1_5` if you have a compatible config/checkpoint.
"""

from __future__ import annotations

import argparse
import sys

import torch


def run_toy_test(device: str = "cpu") -> None:
    print("Running toy small-batch forward/backward on device:", device)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn((2, 16), device=device)
    target = torch.randn((2, 16), device=device)

    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    print("Loss (before):", float(loss))
    loss.backward()
    opt.step()
    opt.zero_grad()

    out2 = model(x)
    loss2 = torch.nn.functional.mse_loss(out2, target)
    print("Loss (after one step):", float(loss2))


def try_instantiate_vlm(device: str = "cpu") -> None:
    try:
        from finetune.sft.models.sft_alpamayo_1_5 import TrainableAlpamayo1_5

        print("Found TrainableAlpamayo1_5 class; attempting to instantiate with minimal args")
        try:
            model = TrainableAlpamayo1_5(config=None)
            print("Instantiated TrainableAlpamayo1_5 (may be incomplete). Running param count...")
            try:
                cnt = sum(p.numel() for p in model.parameters())
                print(f"Parameter count: {cnt:,}")
            except Exception:
                print("Could not count parameters")
        except Exception as e:
            print("Instantiation failed:", e)
            print("If you want to run a full Stage1 test, provide a compatible config and checkpoint.")

    except Exception as e:
        print("Could not import TrainableAlpamayo1_5:", e)
        print("Ensure the file exists and imports in this repo are resolvable.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument(
        "--use-vlm",
        action="store_true",
        help="Try to instantiate TrainableAlpamayo1_5 (may fail without proper config)",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to cpu")
        device = "cpu"

    run_toy_test(device)

    if args.use_vlm:
        try_instantiate_vlm(device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
