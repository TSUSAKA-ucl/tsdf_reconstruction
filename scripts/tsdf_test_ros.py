#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters
import open3d as o3d
import numpy as np

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
        self.bridge = CvBridge()
        
        # TSDFボリュームの初期化
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=4.0 / 512.0,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        # カメラ内参 (D435等、1280x720の標準的な値を例示。本来はCameraInfoから取得推奨)
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        
        # 2つのトピックを同期してサブスクライブ
        self.sub_color = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')
        self.sub_depth = message_filters.Subscriber(self, Image, '/camera/camera/aligned_depth_to_color/image_raw')
        
        self.ts = message_filters.TimeSynchronizer([self.sub_color, self.sub_depth], 10)
        self.ts.registerCallback(self.callback)
        
        self.get_logger().info('TSDF Node started. Integrating images...')

    def callback(self, color_msg, depth_msg):
        # ROS通信からOpen3D画像へ変換
        color_data = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        depth_data = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')

        o3d_color = o3d.geometry.Image(color_data)
        o3d_depth = o3d.geometry.Image(depth_data)

        # RGB-Dデータの作成 (depth_scaleはRealSenseなら通常1000.0)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
        )

        # 統合 (カメラ固定のため外参は単位行列)
        extrinsic = np.eye(4)
        self.volume.integrate(rgbd, self.intrinsic, extrinsic)

    def save_mesh(self):
        # 結果をメッシュとして抽出
        mesh = self.volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh("integrated_mesh.ply", mesh)
        self.get_logger().info("Mesh saved as integrated_mesh.ply")
        # 1. TSDFからメッシュを抽出
        mesh.compute_vertex_normals()
        # 2. 表面の頂点と法線を取得
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        # 3. 裏面の頂点を生成 (法線の逆方向に 5cm 押し出す例)
        thickness = 0.05  # 5cm
        offset_vertices = vertices - (normals * thickness)
        # 4. 表と裏を統合した点群を作成
        combined_vertices = np.vstack((vertices, offset_vertices))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_vertices)
        # 5. (オプション) 凸分解に回すための Alpha Shape による面貼り
        # alpha値を調整することで、エッジを保ちつつ「塊」にできる
        alpha = 0.025
        thick_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        o3d.io.write_triangle_mesh("thick_object.obj", thick_mesh)

def main(args=None):
    rclpy.init(args=args)
    node = TSDFIntegrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_mesh()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
