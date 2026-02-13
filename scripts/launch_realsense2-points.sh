#!/bin/bash
ros2 launch realsense2_camera rs_launch.py \
     pointcloud.enable:=true \
     pointcloud.ordered_pc:=true \
     align_depth.enable:=true \
     enable_sync:=true
