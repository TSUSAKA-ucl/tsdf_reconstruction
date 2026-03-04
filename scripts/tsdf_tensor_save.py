#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import open3d as o3d
import open3d.core as o3c
import numpy as np
import h5py
from pathlib import Path
from datetime import datetime

device = o3c.Device("CUDA:0")
devcpu = o3c.Device("CPU:0")

# カメラの解像度 (D435iのデフォルト1280x720と仮定)
width = 1280
height = 720

# パラメータの定義
fx = 915.4345703125
fy = 913.7946166992188
cx = 629.7527465820312
cy = 356.7699890136719




class TSDFIntegrator(Node):
    def __init__(self):
        super().__init__('tsdf_integrator')
        
        # VoxelBlockGridの初期化
        self.voxel_size = 0.005  # 5mm
        self.block_resolution = 16  # 16^3 voxels per block
        self.reset_vbg()
        intrinsic_np = np.array([
            [915.4345703125, 0.0, 629.7527465820312],
            [0.0, 913.7946166992188, 356.7699890136719],
            [0.0, 0.0, 1.0]
            ])
        self.intrinsic_t = o3c.Tensor(intrinsic_np, dtype=o3c.float64, device=devcpu)
        self.extrinsic_t = o3c.Tensor(np.eye(4), dtype=o3c.float64, device=devcpu) # 今回は固定
        # 2つのトピックを同期してサブスクライブ
        self.sub_color = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        
        self.ts = message_filters.TimeSynchronizer([self.sub_color, self.sub_depth], 10)
        self.ts.registerCallback(self.callback)

        self.save_srv = self.create_service(Trigger, 'save_mesh', self.save_mesh_callback)
        self.reset_srv = self.create_service(Trigger, 'reset_vbg', self.reset_vbg_callback)
        self.get_logger().info('TSDF Node started. Integrating images...')

    def reset_vbg(self):
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=self.voxel_size,
            block_resolution=self.block_resolution,
            block_count=10000,
            device=device
        )
    def reset_vbg_callback(self, request, response):
        self.reset_vbg()
        rclpy.logging.get_logger('TSDFIntegrator').info("VoxelBlockGrid reset.")
        response.success = True
        response.message = "VoxelBlockGrid has been reset."
        return response

    def callback(self, color_msg, depth_msg):
        # ROS2 Image topicを確認してNumpyArrayに変換
        img_raw = depth_msg.height
        img_col = depth_msg.width
        color_1d = np.frombuffer(color_msg.data, dtype=np.uint8)
        depth_1d = np.frombuffer(depth_msg.data, dtype=np.uint16)
        color_np = color_1d.reshape((720, 1280, 3))
        depth_np = depth_1d.reshape((720, 1280))
        color_t = o3d.t.geometry.Image(o3c.Tensor(color_np, device=device))
        depth_t = o3d.t.geometry.Image(o3c.Tensor(depth_np, device=device))

        # 3. 統合ステップ
        # print(f"VBG device: {self.vbg.hashmap().device}")
        # print(f"Depth device: {depth_t.device}")
        # print(f"Intrinsic device: {self.intrinsic_t.device}")
        # print(f"Extrinsic device: {self.extrinsic_t.device}")
        # frustum_block_coords を取得して、更新が必要なボクセルを特定
        frustum_block_coords = self.vbg.compute_unique_block_coordinates(
            depth_t, self.intrinsic_t, self.extrinsic_t, depth_scale=1000.0, depth_max=3.0)
        self.vbg.integrate(frustum_block_coords, depth_t, color_t, 
                           self.intrinsic_t, self.extrinsic_t, depth_scale=1000.0, depth_max=1.5)

    def save_mesh(self, filename: str = "integrated_mesh.ply"):
        # 結果をメッシュとして抽出
        mesh = self.vbg.extract_triangle_mesh()
        legacy_mesh = mesh.to_legacy()
        o3d.io.write_triangle_mesh(filename, legacy_mesh)

    def save_mesh_callback(self, request, response):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_mesh(f"tsdf_mesh_{timestamp}.ply")
        # save raw VBG
        self.vbg.save(f"tsdf_vbg_{timestamp}.zkey")
        rclpy.logging.get_logger('TSDFIntegrator').info("Mesh and dense grid saved.")
        response.success = True
        response.message = "Mesh saved successfully."
        return response


        print(f"Saved to {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = TSDFIntegrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # node.get_logger().info("Shutting down TSDF Node...")
        node.destroy_node()
        # rclpy.shutdown()

if __name__ == '__main__':
    main()
