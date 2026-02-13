# TSDFを使用したdepth画像からのメッシュ作成

## realsenseの起動と平均化によるノイズの除去
Open3Dを使用するため `pip install open3d`が必要。
`venv`のactivateを行った後に`. /install/setup.bash`等を行う。

realsense2による撮影
```
ros2 run tsdf_reconst launch_realsense2-points.sh
```
別のウィンドウで以下と`rviz2`を起動
```
ros2 run tsdf_reconst avg_cloud_node.py
```
適当な点群が得られそうになったら、さらに別のウィンドウで
```
ros2 service call /cloud_averager/save_ply  std_srvs/srv/Trigger
```
以上でPLYを一つ手に入れて、それでテスト。
2026-02-13のHistory参照

## poisson復元とTSDFの比較検討

### poisson復元のコードのビルド
To build the project without CMake developer warnings, use the following command:
```
colcon build [--symlink-install] --cmake-args -Wno-dev
```

