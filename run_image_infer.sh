#!/bin/bash

source /opt/ros/humble/setup.bash
source a1_5_venv/bin/activate


python3 src/alpamayo1_5/infer_from_images.py --no-dummy-black --image-dir input_images --num-frames-per-camera 4 --output-csv outputs/run1.csv