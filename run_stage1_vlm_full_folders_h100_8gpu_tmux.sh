#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SESSION_NAME="alpamayo-stage1-vlm-full-folders-h100-8gpu"
SESSION_NAME="${1:-$DEFAULT_SESSION_NAME}"
OUTPUT_DIR="${2:-$REPO_ROOT/outputs/stage1_vlm_deepspeed_zero3_8gpu_v1}"
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/${SESSION_NAME}.log"

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found in PATH."
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

INNER_COMMAND=$(cat <<EOF
set -uo pipefail
cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

timestamp_line() {
  awk '{ print $0, strftime("[%Y-%m-%dT%H:%M:%S%z]"); fflush(); }'
}

echo "[train] session started" | timestamp_line | tee -a "$LOG_FILE"

if ! source "$REPO_ROOT/a1_5_venv/bin/activate"; then
  echo "[train] failed to activate venv: $REPO_ROOT/a1_5_venv/bin/activate" | timestamp_line | tee -a "$LOG_FILE"
  exec bash
fi

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR"

stdbuf -oL -eL torchrun --nproc_per_node=8 finetune/sft/train_hf_1_5.py --config-name=stage1_vlm_full_folders_h100_8gpu hydra.run.dir="$OUTPUT_DIR" 2>&1 | timestamp_line | tee -a "$LOG_FILE"
exit_code=\${PIPESTATUS[0]}
echo "[train] exit_code=\$exit_code" | timestamp_line | tee -a "$LOG_FILE"
echo "[train] finished" | timestamp_line | tee -a "$LOG_FILE"
exec bash
EOF
)

tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" bash -lc "$INNER_COMMAND"

echo "Started tmux session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo "Attach with: tmux attach -t $SESSION_NAME"