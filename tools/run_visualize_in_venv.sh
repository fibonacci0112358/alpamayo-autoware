#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run tools/visualize_from_pt.py inside the a1_5_venv virtualenv
# Usage: tools/run_visualize_in_venv.sh --pt-dir /path/to/pts --output-dir outputs/vis --max-samples 5

VENV_DIR="${VENV_DIR:-a1_5_venv}"
VENV_ACTIVATE="$VENV_DIR/bin/activate"

if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "Virtualenv activate script not found: $VENV_ACTIVATE" >&2
  exit 2
fi

# Default arguments are hardcoded here for convenience.
PT_FILE=""
PT_DIR="/mnt/nvme/alpamayo_data/001/rosbag2_2026_05_13-10_43_56"
MAX_SAMPLES="10"
BASE_CHECKPOINT="/mnt/nvme/alpamayo_outputs/stage1_vlm_full_folders_h100_8gpu/checkpoint-1011"
SFT_CHECKPOINT="/mnt/nvme/alpamayo_outputs/stage2_h100_8gpu_full_folders_pipeline_20260521_170419_2017710"
OUTPUT_DIR="outputs/pt_visualizations"
VIDEO="true"
NUM_TRAJ_SAMPLES="4"
TOP_P="0.98"
TEMPERATURE="0.6"
MAX_GENERATION_LENGTH="256"
NAV_TEXT=""

# source the virtualenv
# shellcheck disable=SC1090
. "$VENV_ACTIVATE"

# Ensure repository root is on PYTHONPATH so imports like `from alpamayo1_5` work
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# If no CLI args provided, build them from the environment variables above.
if [ "$#" -eq 0 ]; then
  ARGS=()
  if [ -n "$PT_FILE" ]; then
    ARGS+=(--pt "$PT_FILE")
  fi
  if [ -n "$PT_DIR" ]; then
    ARGS+=(--pt-dir "$PT_DIR")
  fi
  ARGS+=(--max-samples "$MAX_SAMPLES")
  ARGS+=(--base-checkpoint "$BASE_CHECKPOINT")
  if [ -n "$SFT_CHECKPOINT" ]; then
    ARGS+=(--sft-checkpoint "$SFT_CHECKPOINT")
  fi
  ARGS+=(--output-dir "$OUTPUT_DIR")
  if [ "$VIDEO" = "true" ] || [ "$VIDEO" = "1" ]; then
    ARGS+=(--video)
  fi
  ARGS+=(--num-traj-samples "$NUM_TRAJ_SAMPLES")
  ARGS+=(--top-p "$TOP_P")
  ARGS+=(--temperature "$TEMPERATURE")
  ARGS+=(--max-generation-length "$MAX_GENERATION_LENGTH")
  if [ -n "$NAV_TEXT" ]; then
    ARGS+=(--nav-text "$NAV_TEXT")
  fi

  python3 "$REPO_ROOT/tools/visualize_from_pt.py" "${ARGS[@]}"
else
  # forward any provided CLI arguments to the python script (takes precedence)
  python3 "$REPO_ROOT/tools/visualize_from_pt.py" "$@"
fi
