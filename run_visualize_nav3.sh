#!/bin/bash

source /opt/ros/humble/setup.bash
source a1_5_venv/bin/activate

IMAGE_DIR="input_images"
CSV_DIR="outputs"

run_visualize_if_exists() {
  local csv_path="$1"
  local out_dir="$2"
  local label="$3"

  if [ ! -f "$csv_path" ]; then
    echo "[SKIP] ${label}: CSV not found -> ${csv_path}"
    return 0
  fi

  echo "[START] ${label}: ${csv_path} -> ${out_dir}"
  python3 visualize_trajectory.py \
    --csv "$csv_path" \
    --image-dir "$IMAGE_DIR" \
    --output-dir "$out_dir"

  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] ${label} (exit=${rc})"
    return $rc
  fi
  echo "[DONE] ${label}"
  return 0
}

run_visualize_if_exists "${CSV_DIR}/run_straight.csv" "trajectory_visualizations_straight" "straight"
run_visualize_if_exists "${CSV_DIR}/run_stop.csv" "trajectory_visualizations_stop" "stop"
run_visualize_if_exists "${CSV_DIR}/run_left_turn.csv" "trajectory_visualizations_left_turn" "left_turn"

echo "Completed nav-text visualizations."
