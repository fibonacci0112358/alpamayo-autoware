Alpamayo-autoware finetune helpers
=================================

This folder contains minimal, repository-local helpers to run small-scale
fine-tuning experiments for Alpamayo1.5 without depending on an external
training repository. It is intentionally lightweight: for production training,
copy the full training harness you prefer into this folder or wire the
existing training repo accordingly.

Files of interest
- `models/sft_alpamayo_1_5.py`: Stage1 trainable wrapper (VLM freeze/co-train support)
- `models/sft_alpamayo_1_5_stage2.py`: Stage2 skeleton (action/expert loss placeholder)
- `data/pai_av_dataset.py`: PaiAV-style JSON/JSONL manifest dataset and collator for Stage1 SFT
- `tests/run_small_batch.py`: Toy small-batch forward/backward test (quick sanity check)
- `scripts/merge_lora_and_save.py`: Merge PEFT/LoRA adapter into base model and save merged checkpoint

Quick start
1. Run toy small-batch test (CPU):

```bash
source a1_5_venv/bin/activate
python3 finetune/sft/tests/run_small_batch.py --device cpu
```

2. If you have a compatible Alpamayo1.5 base checkpoint and LoRA adapter, merge them:

```bash
source a1_5_venv/bin/activate
python3 finetune/sft/scripts/merge_lora_and_save.py \
  --base-checkpoint /path/to/base_checkpoint \
  --lora-dir /path/to/lora_adapter \
  --output-dir /path/to/merged_output \
  --tokenizer-dir /path/to/tokenizer
```

4. Verify merged checkpoint can be loaded by inference class:

```bash
source a1_5_venv/bin/activate
python3 finetune/sft/scripts/verify_merged_checkpoint.py --merged-dir /path/to/merged_output
```

If the script succeeds it will print a parameter count and exit 0. If it fails,
inspect the printed error to determine whether the checkpoint format or tokenizer
is incompatible with `Alpamayo1_5.from_pretrained`.

3. To run a real Stage1 small-batch test, place a minimal Hydra config and dataset
   or instantiate `TrainableAlpamayo1_5` from Python and run one forward/backward.

PaiAV-style manifest support
- The Stage1 entrypoint now accepts `cfg.data.manifest_path` and can run a minimal
  supervised fine-tuning loop with the repository-local collator.
- Supported record fields: `images` / `image_paths` / `frames`, optional
  `camera_indices`, `nav_text`, `messages`, and `completion` / `response` /
  `assistant_text` / `target_text`.
- Each record is turned into the same driving-style chat prompt used by
  `src/alpamayo1_5/helper.py`, so the batch format stays aligned with inference.

Original `alpamayo_r1/data` style local dataset support
- The Stage1 entrypoint also accepts `cfg.data.local_dir` and follows the
  original local PAI dataset loading path (`PAIDataset` / `load_physical_aiavdataset`
  equivalent behavior) implemented in `finetune/sft/data/pai_av_dataset.py`.
- This mode reads local metadata files (`features.csv`, `clip_index.parquet`) and
  loads clip data directly from chunk artifacts.
- `physical_ai_av` must be available in the environment for this mode.

Hydra config style
- The Stage1 launcher now uses `finetune/sft/configs/stage1.yaml` as the default
  config, so you can override fields in the same nested style as Alpamayo1.
- Example:

```bash
source a1_5_venv/bin/activate
python3 finetune/sft/train_hf_1_5.py \
  data.manifest_path=/path/to/pai_av_stage1_manifest.jsonl \
  data.image_root=/path/to/images \
  training.output_dir=outputs/stage1_pai \
  lora.use_lora=true
```

```bash
source a1_5_venv/bin/activate
python3 finetune/sft/train_hf_1_5.py \
  data.local_dir=/path/to/pai_local_export \
  data.chunk_ids=0-9 \
  training.output_dir=outputs/stage1_pai_local \
  lora.use_lora=true
```

Next steps
- Integrate a minimal HF Trainer or custom training loop; add dataset collate functions.
- Implement accurate action/expert loss in `sft_alpamayo_1_5_stage2.py` based on Alpamayo1.5 internals.
