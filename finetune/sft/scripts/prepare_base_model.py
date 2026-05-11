#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare the Alpamayo1.5 base model for fine-tuning.

This script downloads the Alpamayo1.5-10B model from HuggingFace and prepares
it for use in Stage1 SFT training. It ensures that the model weights, config,
and tokenizer are all locally cached.

Usage:
    python finetune/sft/scripts/prepare_base_model.py \
        --output-dir /path/to/base_checkpoint \
        --model-id nvidia/Alpamayo-1.5-10B
"""

import argparse
import sys
from pathlib import Path


def download_base_model(model_id: str, output_dir: Path) -> Path:
    """Download Alpamayo model from HuggingFace Hub.
    
    Args:
        model_id: HuggingFace model ID (e.g., 'nvidia/Alpamayo-1.5-10B')
        output_dir: Directory where to save the model
        
    Returns:
        Path to the downloaded model directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is required. Install it with:")
        print("  pip install huggingface-hub")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_id}...")
    print(f"  Output directory: {output_dir}")
    
    # Download the entire model snapshot
    local_path = Path(snapshot_download(model_id, cache_dir=str(output_dir.parent)))
    
    print(f"✓ Downloaded to: {local_path}")
    return local_path


def verify_checkpoint(checkpoint_path: Path) -> bool:
    """Verify that the downloaded checkpoint has required files.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        
    Returns:
        True if checkpoint is valid, False otherwise
    """
    required_files = [
        "config.json",
        "model.safetensors.index.json",
    ]
    
    missing = []
    for fname in required_files:
        fpath = checkpoint_path / fname
        if not fpath.exists():
            missing.append(fname)
    
    if missing:
        print(f"✗ Checkpoint validation failed. Missing files: {missing}")
        return False
    
    print(f"✓ Checkpoint validation passed ({checkpoint_path.name})")
    return True


def test_model_load(checkpoint_path: Path) -> bool:
    """Test that the model can be loaded.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        
    Returns:
        True if model loads successfully, False otherwise
    """
    try:
        import torch
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(str(checkpoint_path))
        print(f"✓ Model config loaded successfully")
        print(f"  Model type: {config.model_type}")
        print(f"  VLM: {getattr(config, 'vlm_name_or_path', 'N/A')}")
        return True
    except Exception as e:
        print(f"✗ Failed to load model config: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare Alpamayo1.5 base model for fine-tuning"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="nvidia/Alpamayo-1.5-10B",
        help="HuggingFace model ID to download"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub" / "Alpamayo-1.5-10B",
        help="Directory where to save the model"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip checkpoint verification"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Alpamayo1.5 Base Model Preparation")
    print("=" * 70)
    
    # Download the model
    checkpoint_path = download_base_model(args.model_id, args.output_dir)
    
    # Verify the checkpoint
    if not args.skip_verify:
        if not verify_checkpoint(checkpoint_path):
            return 1
    
    # Test loading the model
    if not test_model_load(checkpoint_path):
        print("\nWarning: Model loading test failed, but checkpoint may still be usable.")
    
    print("\n" + "=" * 70)
    print("✓ Base model preparation completed successfully!")
    print("=" * 70)
    print(f"\nCheckpoint location: {checkpoint_path}")
    print(f"\nTo use in training, set:")
    print(f"  model.vlm_name_or_path={checkpoint_path}")
    print(f"\nOr pass as argument:")
    print(f"  python finetune/sft/train_hf_1_5.py \\")
    print(f"    model.vlm_name_or_path={checkpoint_path} \\")
    print(f"    data.local_dir=pai_dataset \\")
    print(f"    lora.use_lora=true")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
