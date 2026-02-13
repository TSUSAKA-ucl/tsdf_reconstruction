#!/bin/bash
ros2 launch realsense2_camera rs_launch.py \
    align_depth.enable:=true \
    enable_accel:=true \
    enable_gyro:=true \
    unite_imu_method:=2 \
    pointcloud.enable:=true
