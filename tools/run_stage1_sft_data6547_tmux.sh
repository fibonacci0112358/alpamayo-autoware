#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_SESSION_NAME="alpamayo-stage1-sft-data6547"
SESSION_NAME="${1:-$DEFAULT_SESSION_NAME}"
OUTPUT_BASE="${2:-/mnt/nvme/alpamayo_outputs/stage1_sft_data6547}"
LOG_DIR="$REPO_ROOT/logs"
# optional third arg: run timestamp to group multi-process outputs under same run
RUN_TS="${3:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_BASE}_$RUN_TS"
LOG_FILE="$LOG_DIR/${SESSION_NAME}_${RUN_TS}.log"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found in PATH."
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  # If the existing session only runs an interactive shell (leftover from debugging),
  # allow recreating it unless KEEP_SESSION is set.
  pane_cmds=$(tmux list-panes -t "$SESSION_NAME" -F '#{pane_current_command}' 2>/dev/null | tr '\n' ' ')
  if [ "${KEEP_SESSION:-0}" != "1" ] && echo "$pane_cmds" | egrep -q "^(bash|sh)"; then
    echo "Found existing session with interactive shell only; replacing: $SESSION_NAME"
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
  else
    echo "tmux session already exists: $SESSION_NAME"
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 0
  fi
fi

INNER_COMMAND=$(cat <<EOF
set -uo pipefail
cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

timestamp_line() {
  awk '{ print \$0, strftime("[%Y-%m-%dT%H:%M:%S%z]"); fflush(); }'
}

echo "[train] session started" | timestamp_line | tee -a "$LOG_FILE"

if ! source "$REPO_ROOT/a1_5_venv/bin/activate"; then
  echo "[train] failed to activate venv: $REPO_ROOT/a1_5_venv/bin/activate" | timestamp_line | tee -a "$LOG_FILE"
  exec bash
fi

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-14}"
export MKL_NUM_THREADS="\${MKL_NUM_THREADS:-14}"
export KMP_AFFINITY="granularity=fine,compact"
export OMP_WAIT_POLICY="PASSIVE"
mkdir -p "$OUTPUT_DIR"

if [[ -z "\${MASTER_PORT:-}" ]]; then
  MASTER_PORT="\$((20000 + (RANDOM % 20000)))"
fi
echo "[train] using master port: \$MASTER_PORT" | timestamp_line | tee -a "$LOG_FILE"

DS_CONFIG="finetune/sft/configs/deepspeed/zero2.json"
echo "[train] using deepspeed config: \$DS_CONFIG" | timestamp_line | tee -a "$LOG_FILE"

stdbuf -oL -eL torchrun --master_port="\$MASTER_PORT" --nproc_per_node=8 finetune/sft/train_hf_1_5.py --config-name=stage1_sft_data6547 hydra.run.dir="$OUTPUT_DIR" training.output_dir="$OUTPUT_DIR" training.deepspeed="\$DS_CONFIG" 2>&1 | timestamp_line | tee -a "$LOG_FILE"
exit_code=\${PIPESTATUS[0]}
echo "[train] exit_code=\$exit_code" | timestamp_line | tee -a "$LOG_FILE"
echo "[train] finished" | timestamp_line | tee -a "$LOG_FILE"
if [ "${KEEP_SESSION:-0}" = "1" ]; then
  exec bash
fi
EOF
)

tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" bash -lc "$INNER_COMMAND"

# create/update symlink from base to this run
ln -sfn "$OUTPUT_DIR" "$OUTPUT_BASE"

echo "Started tmux session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo "Output dir: $OUTPUT_DIR (symlink: $OUTPUT_BASE -> $OUTPUT_DIR)"
echo "Attach with: tmux attach -t $SESSION_NAME"