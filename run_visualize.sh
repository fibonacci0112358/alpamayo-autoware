#!/bin/bash

source /opt/ros/humble/setup.bash
source a1_5_venv/bin/activate

python3 visualize_trajectory.py --csv outputs/run1.csv --image-dir input_images --output-dir trajectory_visualizations