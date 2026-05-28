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
SKIP_STAGE1=0
if [ -L "$STAGE1_OUTPUT_BASE" ] || [ -d "$STAGE1_OUTPUT_BASE" ]; then
  STAGE1_TARGET="$(readlink -f "$STAGE1_OUTPUT_BASE" 2>/dev/null || echo "$STAGE1_OUTPUT_BASE")"
  if [ -e "$STAGE1_TARGET/model.safetensors.index.json" ] || ls "$STAGE1_TARGET"/checkpoint* >/dev/null 2>&1; then
    echo "[pipeline] Found existing Stage1 outputs at $STAGE1_TARGET; skipping Stage1 launch."
    SKIP_STAGE1=1
  fi
fi

if [ "$SKIP_STAGE1" -eq 0 ]; then
  "$REPO_ROOT/tools/run_stage1_sft_data6547_tmux.sh" "$STAGE1_SESSION" "$STAGE1_OUTPUT_BASE" "$RUN_TS"
else
  echo "[pipeline] Skipping Stage1; proceeding to wait/check outputs before Stage2."
fi

if [ "$SKIP_STAGE1" -eq 1 ]; then
  echo "[pipeline] Stage1 outputs detected earlier; skipping wait and proceeding to Stage2."
else
  echo "[pipeline] Waiting for Stage1 to finish and produce outputs..."
  # Wait until either the stage1 tmux session exits and outputs appear, or until outputs appear
  while true; do
    if tmux has-session -t "$STAGE1_SESSION" 2>/dev/null; then
      echo "[pipeline] Stage1 tmux session still exists; sleeping 60s..."
      sleep 60
      continue
    fi

    # session not present; check for output symlink/dir and expected artifacts
    if [ -L "$STAGE1_OUTPUT_BASE" ] || [ -d "$STAGE1_OUTPUT_BASE" ]; then
      TARGET="$(readlink -f "$STAGE1_OUTPUT_BASE" 2>/dev/null || echo "$STAGE1_OUTPUT_BASE")"
      if [ -e "$TARGET/model.safetensors.index.json" ] || ls "$TARGET"/checkpoint* >/dev/null 2>&1; then
        echo "[pipeline] Stage1 output found at $TARGET"
        break
      else
        echo "[pipeline] Stage1 session ended but outputs not ready in $TARGET; sleeping 30s..."
        sleep 30
        continue
      fi
    else
      echo "[pipeline] No Stage1 session and no output symlink yet; sleeping 30s..."
      sleep 30
    fi
  done
fi

echo "[pipeline] Stage1 finished. Launching Stage2 (tmux session: $STAGE2_SESSION)"
"$REPO_ROOT/tools/run_stage2_sft_data6547_tmux.sh" "$STAGE2_SESSION" "$STAGE2_OUTPUT_BASE" "$RUN_TS"

echo "[pipeline] Stage2 started. Monitor tmux session: $STAGE2_SESSION"
echo "[pipeline] Done." 
