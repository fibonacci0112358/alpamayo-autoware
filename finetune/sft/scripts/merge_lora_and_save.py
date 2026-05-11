#!/usr/bin/env python3
"""Merge PEFT/LoRA adapters into base model and save merged checkpoint.

Usage example:
  python finetune/sft/scripts/merge_lora_and_save.py \
    --base-checkpoint /path/to/base_model \
    --lora-dir /path/to/lora_adapter \
    --output-dir /path/to/merged_checkpoint \
    --tokenizer-dir /path/to/tokenizer_or_processor

This script uses `peft`'s merge_and_unload when available. It is intended to
live inside this repository so users don't need to run code in the external
training repo.
"""

import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--lora-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tokenizer-dir", required=False)
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(args.base_checkpoint)
    lora = Path(args.lora_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from transformers import AutoModel
    except Exception as e:
        print("transformers import failed:", e)
        return 1

    try:
        from peft import PeftModel
    except Exception as e:
        print("peft import failed:", e)
        return 1

    print("Loading base model from", base)
    model = AutoModel.from_pretrained(str(base), trust_remote_code=True)

    print("Loading PEFT adapter from", lora)
    peft_model = PeftModel.from_pretrained(model, str(lora))

    if hasattr(peft_model, "merge_and_unload"):
        print("Merging LoRA into base model (merge_and_unload)")
        merged = peft_model.merge_and_unload()
    else:
        print("PEFT model has no merge_and_unload; attempting state_dict merge")
        # Fallback: try to get state_dict and load into base non-strict
        try:
            sd = peft_model.state_dict()
            model.load_state_dict(sd, strict=False)
            merged = model
        except Exception as e:
            print("Fallback merge failed:", e)
            return 1

    print("Saving merged model to", out)
    merged.save_pretrained(str(out))

    if args.tokenizer_dir:
        try:
            from transformers import AutoTokenizer
            print("Saving tokenizer from", args.tokenizer_dir)
            tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
            tok.save_pretrained(str(out))
        except Exception as e:
            print("Tokenizer copy failed:", e)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
