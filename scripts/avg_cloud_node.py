#!/usr/bin/env python3
import os
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Empty
from std_srvs.srv import Trigger
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from collections import deque

class CloudAverager(Node):
    def __init__(self):
        super().__init__('cloud_averager')
        
        # --- パラメータ ---
        self.declare_parameter('avg_frames', 10)   # 平均化するフレーム数
        self.declare_parameter('z_min', 0.0)      # Cropping 範囲 (m)
        self.declare_parameter('z_max', 1.5)
        
        self.sub = self.create_subscription(
            PointCloud2, '/camera/camera/depth/color/points', self.callback, 10)
        self.pub = self.create_publisher(PointCloud2, '~/averaged_cloud', 10)
        
        self.buffer = deque() # 点群データを貯めるキュー
        self.get_logger().info('Cloud Averager Node Started.')
        # 1. リセットサービスの作成
        self.reset_srv = self.create_service(Empty, '~/reset_buffer', self.reset_callback)
        # 2. パラメータ更新の監視 (Jazzy/Humble標準)
        self.add_on_set_parameters_callback(self.parameter_callback)
        # PLY保存サービス
        self.save_srv = self.create_service(Trigger, '~/save_ply', self.save_callback)
        self.last_filtered_xyz = None
        self.last_filtered_rgb = None
        self.get_logger().info('Save PLY service [/save_ply] is ready.')


    def save_callback(self, request, response):
        """現在の平均化済みデータをPLYとして保存する"""
        if self.last_filtered_xyz is None or len(self.last_filtered_xyz) == 0:
            response.success = False
            response.message = "No averaged data available to save."
            return response

        try:
            # Open3D形式に変換
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.last_filtered_xyz.astype(np.float64))
            
            # RGBのデコード (前述のビットシフト処理)
            rgb_ints = self.last_filtered_rgb.view(np.uint32)
            r = ((rgb_ints >> 16) & 0xFF) / 255.0
            g = ((rgb_ints >> 8) & 0xFF) / 255.0
            b = (rgb_ints & 0xFF) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(np.stack([r, g, b], axis=1))

            # 保存ファイル名の生成 (タイムスタンプ付き)
            filename = f"averaged_cloud_{self.get_clock().now().nanoseconds}.ply"
            o3d.io.write_point_cloud(filename, pcd, write_ascii=False)
            
            response.success = True
            response.message = f"Saved successfully as: {os.path.abspath(filename)}"
            self.get_logger().info(response.message)
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to save PLY: {str(e)}"
            
        return response

    def reset_callback(self, request, response):
        """バッファを即座に空にする"""
        self.buffer.clear()
        self.get_logger().info('Buffer cleared by service call.')
        return response

    def parameter_callback(self, params):
        """avg_framesが変更されたらバッファの長さを調整する"""
        for param in params:
            if param.name == 'avg_frames' and param.type_ == param.Type.INTEGER:
                new_len = param.value
                self.get_logger().info(f'Changing buffer size to: {new_len}')
                # 現在のバッファが新しい設定より長ければ切り詰める
                while len(self.buffer) > new_len:
                    self.buffer.popleft()
        return rclpy.node.SetParametersResult(successful=True)


    def callback(self, msg):
        avg_count = self.get_parameter('avg_frames').value
        z_min = self.get_parameter('z_min').value
        z_max = self.get_parameter('z_max').value
        
        # 1. PointCloud2 を読み込み (NaNや0を含めて保持)
        gen = pc2.read_points(msg, field_names=['x', 'y', 'z', 'rgb'], skip_nans=False)
        data = np.fromiter(gen, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'f4')])
        curr_xyz = np.stack([data['x'], data['y'], data['z']], axis=1)

        # 2. バッファに貯める
        self.buffer.append(curr_xyz)
        if len(self.buffer) >= avg_count:
            # 配列化 (Frames, Points, 3)
            stacked = np.array(self.buffer)
            
            # 3. 0（無効値）を NaN に置換して統計計算から除外する
            # これをしないと、0 と 1.0m の平均が 0.5m になり原点へ引きずられる
            range_max_thresh = 2.0
            range_min_thresh = 0.5
            stacked_nan = stacked.copy()
            stacked_nan[stacked_nan == 0] = np.nan
            # stacked_nan[stacked_nan > range_max_thresh] = np.nan
            # stacked_nan[stacked_nan < range_min_thresh] = np.nan
            
            # 4. 平均と標準偏差の計算 (axis=0 は時間方向)
            avg_xyz = np.nanmean(stacked_nan, axis=0)
            std_xyz = np.nanstd(stacked_nan, axis=0)
            
            # 5. フィルタリング条件
            # - Z軸の標準偏差が 0.05 (5cm) 以下（＝ジタバタしていない）
            # - 有効なサンプル数が半分以上ある（＝たまにしか映らない点は除外）
            valid_samples = np.count_nonzero(~np.isnan(stacked_nan[:, :, 2]), axis=0)
            
            std_thresh = 0.02  # 2cm 以上の揺れがある点は不安定とみなす（調整可）
            mask = (std_xyz[:, 2] < std_thresh) & \
                   (valid_samples > (avg_count * 0.5)) & \
                   (avg_xyz[:, 2] > z_min) & \
                   (avg_xyz[:, 2] < z_max)

            # 6. マスク適用
            filtered_xyz = avg_xyz[mask]
            filtered_rgb = data['rgb'][mask]
            self.last_filtered_xyz = filtered_xyz
            self.last_filtered_rgb = filtered_rgb

            # 7. メッセージ構築 (省略: 前回と同様)
            # self.publish_cloud(msg.header, msg.fields, filtered_xyz, filtered_rgb)
            # PointCloud2 メッセージの再構築
            # 出力データを作成
            out_data = []
            for i in range(len(filtered_xyz)):
                out_data.append([filtered_xyz[i,0], filtered_xyz[i,1], filtered_xyz[i,2], filtered_rgb[i]])
            
            out_msg = pc2.create_cloud(msg.header, msg.fields, out_data)
            self.pub.publish(out_msg)
            
            self.get_logger().info(f'Published averaged cloud ({len(self.buffer)} frames)', once=False)

def main():
    rclpy.init()
    node = CloudAverager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
