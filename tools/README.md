# Tools

This folder contains helper scripts used for evaluation and visualization.

`visualize_from_pt.py` - Load `.pt` dataset samples and generate visualizations
that overlay ground-truth trajectories with model predictions from a base
checkpoint and an optional SFT checkpoint.

Quick example:

```bash
python3 tools/visualize_from_pt.py \
  --pt-dir /mnt/nvme/alpamayo_data/001/rosbag2_2026_05_13-10_43_56 \
  --max-samples 5 \
  --base-checkpoint nvidia/Alpamayo-1.5-10B \
  --sft-checkpoint /mnt/nvme/alpamayo_outputs/stage2_h100_8gpu_full_folders_pipeline_20260521_170419_2017710 \
  --output-dir outputs/pt_visualizations --video
```

Dependencies: the repository's Python environment (PyTorch, Transformers, Matplotlib).
For MP4 output, install `imageio`.
