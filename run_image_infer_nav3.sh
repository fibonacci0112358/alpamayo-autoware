#!/bin/bash

source /opt/ros/humble/setup.bash
source a1_5_venv/bin/activate

IMAGE_DIR="input_images"
FRAMES_PER_CAMERA=4
OUTPUT_DIR="outputs"

mkdir -p "${OUTPUT_DIR}"

# 1) Straight
python3 src/alpamayo1_5/infer_from_images.py \
  --no-dummy-black \
  --image-dir "${IMAGE_DIR}" \
  --num-frames-per-camera "${FRAMES_PER_CAMERA}" \
  --nav-text "Continue straight." \
  --output-csv "${OUTPUT_DIR}/run_straight.csv"

# 2) Stop
python3 src/alpamayo1_5/infer_from_images.py \
  --no-dummy-black \
  --image-dir "${IMAGE_DIR}" \
  --num-frames-per-camera "${FRAMES_PER_CAMERA}" \
  --nav-text "Stop." \
  --output-csv "${OUTPUT_DIR}/run_stop.csv"

# 3) Left turn
python3 src/alpamayo1_5/infer_from_images.py \
  --no-dummy-black \
  --image-dir "${IMAGE_DIR}" \
  --num-frames-per-camera "${FRAMES_PER_CAMERA}" \
  --nav-text "Turn left." \
  --output-csv "${OUTPUT_DIR}/run_left_turn.csv"

echo "Completed all nav-text runs:"
echo "  - ${OUTPUT_DIR}/run_straight.csv"
echo "  - ${OUTPUT_DIR}/run_stop.csv"
echo "  - ${OUTPUT_DIR}/run_left_turn.csv"
