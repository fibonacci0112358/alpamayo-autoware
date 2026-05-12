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


def try_instantiate_vlm(device: str = "cpu", vlm_name_or_path: str | None = None) -> None:
    try:
        from finetune.sft.models.sft_alpamayo_1_5 import TrainableAlpamayo1_5

        print("Found TrainableAlpamayo1_5 class; attempting to instantiate with minimal args")
        try:
            if vlm_name_or_path:
                from types import SimpleNamespace

                cfg = SimpleNamespace(vlm_name_or_path=vlm_name_or_path, model_dtype="auto", attn_implementation=None)
                model = TrainableAlpamayo1_5(config=cfg)
            else:
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
    parser.add_argument(
        "--stage2",
        action="store_true",
        help="Run a minimal Stage2 compute_action_loss smoke test",
    )
    parser.add_argument(
        "--vlm-name-or-path",
        default=None,
        help="Path or HF id for the Alpamayo1.5 VLM base checkpoint to instantiate (optional)",
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to cpu")
        device = "cpu"

    run_toy_test(device)

    if args.use_vlm:
        try_instantiate_vlm(device, args.vlm_name_or_path)

    if args.stage2:
        try:
            from finetune.sft.models.sft_alpamayo_1_5_stage2 import (
                TrainableAlpamayo1_5_Stage2,
            )

            print("Found TrainableAlpamayo1_5_Stage2; running compute_action_loss test")
            model = None
            try:
                # Try to instantiate with a minimal config; allow failures
                if args.vlm_name_or_path:
                    from types import SimpleNamespace

                    cfg = SimpleNamespace(vlm_name_or_path=args.vlm_name_or_path, model_dtype="auto", attn_implementation=None)
                    model = TrainableAlpamayo1_5_Stage2(config=cfg)
                else:
                    model = TrainableAlpamayo1_5_Stage2(config=None)
                print("Instantiated Stage2 wrapper (may be incomplete)")
            except Exception as e:
                print("Instantiation failed (continuing with static test):", e)

            # Create fake model outputs and labels to exercise compute_action_loss
            B, T, C = 2, 4, 3
            pred = torch.randn((B, T, C), device=device)
            labels = torch.randn((B, T, C), device=device)
            model_outputs = {"pred": pred}

            if model is not None:
                try:
                    loss = model.compute_action_loss(model_outputs, labels)
                    print("compute_action_loss (model):", float(loss))
                except Exception as e:
                    print("compute_action_loss (model) failed:", e)

            # Exercise compute_action_loss by creating a minimal dummy `self`
            try:
                from finetune.sft.models.sft_alpamayo_1_5_stage2 import (
                    TrainableAlpamayo1_5_Stage2 as _Dummy,
                )

                class _MinimalSelf:
                    def parameters(self):
                        # provide at least one parameter so .device can be resolved
                        p = torch.nn.Parameter(torch.zeros(1))
                        return iter([p])

                dummy = _MinimalSelf()
                loss = _Dummy.compute_action_loss(dummy, model_outputs, labels)
                print("compute_action_loss (static):", float(loss))
            except Exception as e:
                print("Static compute_action_loss test failed:", e)

        except Exception as e:
            print("Could not run Stage2 test:", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
