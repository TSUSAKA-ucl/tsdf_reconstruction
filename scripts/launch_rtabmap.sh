#!/bin/bash
ros2 launch rtabmap_launch rtabmap.launch.py \
     frame_id:=camera_link \
     args:="-d --Rtabmap/DetectionRate 0.5" \
     rgb_topic:=/camera/camera/color/image_raw \
     depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
     camera_info_topic:=/camera/camera/color/camera_info \
     approx_sync:=true \
     map_always_update:=true
