#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"

# Session and output base names
STAGE1_SESSION="alpamayo-stage1-sft-data6547"
STAGE2_SESSION="alpamayo-stage2-sft-data6547"
STAGE1_OUTPUT_BASE="/mnt/nvme/alpamayo_outputs/stage1_sft_data6547"
STAGE2_OUTPUT_BASE="/mnt/nvme/alpamayo_outputs/stage2_sft_data6547"

# Optional: allow overriding via env/args
RUN_TS="${1:-$(date +%Y%m%d_%H%M%S)}"

echo "[pipeline] Starting full SFT pipeline (data6547) at $RUN_TS"
mkdir -p "$LOG_DIR"

echo "[pipeline] Launching Stage1 (tmux session: $STAGE1_SESSION)"
"$REPO_ROOT/tools/run_stage1_sft_data6547_tmux.sh" "$STAGE1_SESSION" "$STAGE1_OUTPUT_BASE" "$RUN_TS"

echo "[pipeline] Waiting for Stage1 tmux session to finish..."
while tmux has-session -t "$STAGE1_SESSION" 2>/dev/null; do
  sleep 60
done

echo "[pipeline] Stage1 finished. Launching Stage2 (tmux session: $STAGE2_SESSION)"
"$REPO_ROOT/tools/run_stage2_sft_data6547_tmux.sh" "$STAGE2_SESSION" "$STAGE2_OUTPUT_BASE" "$RUN_TS"

echo "[pipeline] Stage2 started. Monitor tmux session: $STAGE2_SESSION"
echo "[pipeline] Done." 
