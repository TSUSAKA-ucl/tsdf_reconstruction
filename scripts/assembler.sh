ros2 run rtabmap_util map_assembler --ros-args \
    -p range_max:=1.0 \
    -p decimation:=1 \
    -p 'Grid/CellSize:="0.01"' \
    -p map_always_update:=true \
    -r mapData:=/rtabmap/rtabmap/mapData \
    -r rtabmap/get_map_data:=/rtabmap/rtabmap/get_map_data
