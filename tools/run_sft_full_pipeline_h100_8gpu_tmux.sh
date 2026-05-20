#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"

STAGE1_OUTPUT_DIR="$REPO_ROOT/outputs/stage1_vlm_full_folders_h100_8gpu_pipeline"
STAGE2_OUTPUT_DIR="$REPO_ROOT/outputs/stage2_h100_8gpu_full_folders_pipeline"
STAGE1_LOG_FILE="$LOG_DIR/alpamayo-stage1-pipeline-${RUN_TS}.log"
STAGE2_LOG_FILE="$LOG_DIR/alpamayo-stage2-pipeline-${RUN_TS}.log"
MAIN_LOG_FILE="$LOG_DIR/sft_full_pipeline_${RUN_TS}.log"
PIPELINE_SESSION_NAME="alpamayo-sft-full-pipeline-${RUN_TS}"
DEEPSPEED_CONFIG="finetune/sft/configs/deepspeed/zero2.json"
STAGE2_CONFIG_NAME="stage2_h100_8gpu_full_folders"

mkdir -p "$LOG_DIR" "$STAGE1_OUTPUT_DIR" "$STAGE2_OUTPUT_DIR"
STAGE1_RUN_DIR="${STAGE1_OUTPUT_DIR}_$RUN_TS"
STAGE2_RUN_DIR="${STAGE2_OUTPUT_DIR}_$RUN_TS"
mkdir -p "$STAGE1_RUN_DIR" "$STAGE2_RUN_DIR"

timestamp_line() {
  awk '{ print $0, strftime("[%Y-%m-%dT%H:%M:%S%z]"); fflush(); }'
}

log_main() {
  echo "$1" | tee -a "$MAIN_LOG_FILE"
}

run_logged_command() {
  local log_file="$1"
  shift
  "$@" 2>&1 | timestamp_line | tee -a "$log_file"
  return "${PIPESTATUS[0]}"
}

find_stage_model_dir() {
  local output_dir="$1"
  local checkpoint_dir
  checkpoint_dir="$(find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1)"
  if [[ -n "$checkpoint_dir" ]]; then
    echo "$checkpoint_dir"
    return 0
  fi

  local direct_model
  direct_model="$(find "$output_dir" -maxdepth 2 \( -name 'pytorch_model.bin' -o -name 'model.safetensors' \) | head -1)"
  if [[ -n "$direct_model" ]]; then
    dirname "$direct_model"
    return 0
  fi

  if [[ -f "$output_dir/pytorch_model.bin" || -f "$output_dir/model.safetensors" ]]; then
    echo "$output_dir"
    return 0
  fi

  return 1
}

if ! command -v tmux >/dev/null 2>&1; then
  log_main "[pipeline] ERROR: tmux is required but was not found in PATH"
  exit 1
fi

PIPELINE_COMMAND=$(cat <<EOF
set -euo pipefail
cd "$REPO_ROOT"

timestamp_line() {
  awk '{ print \$0, strftime("[%Y-%m-%dT%H:%M:%S%z]"); fflush(); }'
}

log_main() {
  echo "\$1" | tee -a "$MAIN_LOG_FILE"
}

run_logged_command() {
  local log_file="\$1"
  shift
  "\$@" 2>&1 | timestamp_line | tee -a "\$log_file"
  return "\${PIPESTATUS[0]}"
}

find_stage_model_dir() {
  local output_dir="\$1"
  local checkpoint_dir
  checkpoint_dir="\$(find "\$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1)"
  if [[ -n "\$checkpoint_dir" ]]; then
    echo "\$checkpoint_dir"
    return 0
  fi

  local direct_model
  direct_model="\$(find "\$output_dir" -maxdepth 2 \( -name 'pytorch_model.bin' -o -name 'model.safetensors' \) | head -1)"
  if [[ -n "\$direct_model" ]]; then
    dirname "\$direct_model"
    return 0
  fi

  if [[ -f "\$output_dir/pytorch_model.bin" || -f "\$output_dir/model.safetensors" ]]; then
    echo "\$output_dir"
    return 0
  fi

  return 1
}

log_main "[pipeline] Full SFT pipeline started at \$(date)"
log_main "[pipeline] Main log file: $MAIN_LOG_FILE"
log_main "[pipeline] Stage1 log file: $STAGE1_LOG_FILE"
log_main "[pipeline] Stage2 log file: $STAGE2_LOG_FILE"

if ! source "$REPO_ROOT/a1_5_venv/bin/activate"; then
  log_main "[pipeline] ERROR: failed to activate venv: $REPO_ROOT/a1_5_venv/bin/activate"
  exit 1
fi

if ! command -v torchrun >/dev/null 2>&1; then
  log_main "[pipeline] ERROR: torchrun was not found in PATH after venv activation"
  exit 1
fi

if ! command -v stdbuf >/dev/null 2>&1; then
  log_main "[pipeline] ERROR: stdbuf was not found in PATH"
  exit 1
fi

export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-14}"
export MKL_NUM_THREADS="\${MKL_NUM_THREADS:-14}"
export KMP_AFFINITY="granularity=fine,compact"
export OMP_WAIT_POLICY="PASSIVE"

MASTER_PORT="\${MASTER_PORT:-$((20000 + (RANDOM % 20000)))}"
log_main "[pipeline] Using master port: \$MASTER_PORT"

log_main "[pipeline] Stage1 training starting"
run_logged_command "$STAGE1_LOG_FILE" stdbuf -oL -eL torchrun --master_port="\$MASTER_PORT" --nproc_per_node=8 finetune/sft/train_hf_1_5.py --config-name=stage1_vlm_full_folders_h100_8gpu hydra.run.dir="$STAGE1_RUN_DIR" training.output_dir="$STAGE1_RUN_DIR" training.deepspeed="$DEEPSPEED_CONFIG"
stage1_exit_code=\$?
log_main "[pipeline] Stage1 exit_code=\$stage1_exit_code"

if [[ \$stage1_exit_code -ne 0 ]]; then
  log_main "[pipeline] ERROR: Stage1 failed; Stage2 will not start"
  exit "\$stage1_exit_code"
fi

STAGE1_MODEL_PATH="\$(find_stage_model_dir "$STAGE1_RUN_DIR")"
if [[ -z "\${STAGE1_MODEL_PATH:-}" ]]; then
  log_main "[pipeline] ERROR: could not find Stage1 model directory under $STAGE1_RUN_DIR"
  exit 1
fi

log_main "[pipeline] Stage1 model directory: \$STAGE1_MODEL_PATH"
log_main "[pipeline] Creating/updating symlink: $STAGE1_OUTPUT_DIR -> $STAGE1_RUN_DIR"
ln -sfn "$STAGE1_RUN_DIR" "$STAGE1_OUTPUT_DIR"
log_main "[pipeline] Stage2 training starting"
run_logged_command "$STAGE2_LOG_FILE" stdbuf -oL -eL torchrun --master_port="\$MASTER_PORT" --nproc_per_node=8 finetune/sft/train_hf_1_5_stage2.py --config-name="$STAGE2_CONFIG_NAME" hydra.run.dir="$STAGE2_RUN_DIR" training.output_dir="$STAGE2_RUN_DIR" model.config.vlm_name_or_path="\$STAGE1_MODEL_PATH"
stage2_exit_code=\$?
log_main "[pipeline] Stage2 exit_code=\$stage2_exit_code"

if [[ \$stage2_exit_code -ne 0 ]]; then
  log_main "[pipeline] ERROR: Stage2 failed"
  exit "\$stage2_exit_code"
fi

log_main "[pipeline] Creating/updating symlink: $STAGE2_OUTPUT_DIR -> $STAGE2_RUN_DIR"
ln -sfn "$STAGE2_RUN_DIR" "$STAGE2_OUTPUT_DIR"

log_main "[pipeline] Finished successfully at \$(date)"
EOF
)

if tmux has-session -t "$PIPELINE_SESSION_NAME" 2>/dev/null; then
  log_main "[pipeline] ERROR: tmux session already exists: $PIPELINE_SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$PIPELINE_SESSION_NAME" -c "$REPO_ROOT" bash -lc "$PIPELINE_COMMAND"

log_main "[pipeline] Detached tmux session created: $PIPELINE_SESSION_NAME"
log_main "[pipeline] Logs will continue under: $MAIN_LOG_FILE"
log_main "[pipeline] You can close this terminal safely."
