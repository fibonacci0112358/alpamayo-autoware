#!/bin/bash

source /opt/ros/humble/setup.bash
source a1_5_venv/bin/activate

python3 ./src/alpamayo_ros/alpamayo_ros/alpamayo_node.py --ros-args \
    -p camera_topics:="['/sensing/camera/camera3/image_raw/compressed', \
    -p camera_indices:="[1]"

# python3 ./src/alpamayo_ros/alpamayo_ros/alpamayo_node.py --ros-args \
#     -p camera_topics:="['/sensing/camera/camera3/image_raw/compressed', \
#     '/sensing/camera/camera1/image_raw/compressed', \
#     '/sensing/camera/camera4/image_raw/compressed', \
#     '/sensing/camera/camera2/image_raw/compressed']" \
#     -p camera_indices:="[0, 1, 2, 6]"