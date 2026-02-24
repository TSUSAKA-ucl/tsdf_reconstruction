#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import open3d as o3d
import open3d.core as o3c
import numpy as np
import copy

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
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=self.voxel_size,
            block_resolution=16,   # 16^3 voxels per block
            block_count=10000,
            device=device
        )
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

        self.srv = self.create_service(Trigger, 'save_mesh', self.save_mesh_callback)
        self.get_logger().info('TSDF Node started. Integrating images...')

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

    def save_mesh(self):
        # 結果をメッシュとして抽出
        mesh = self.vbg.extract_triangle_mesh()
        legacy_mesh = mesh.to_legacy()
        o3d.io.write_triangle_mesh("integrated_mesh_tensor.ply", legacy_mesh)

    def create_safe_thick_pcd(self, thickness=0.05):
        device = self.vbg.hashmap().device
        # voxel_size = self.vbg.get_voxel_size()
        voxel_size = self.voxel_size
        # 1. 表面のメッシュを抽出 (Tensor版)
        mesh = self.vbg.extract_triangle_mesh()
        if len(mesh.vertex.positions) == 0:
            return None
        vertices = mesh.vertex.positions # [N, 3] Tensor
        normals = mesh.vertex.normals    # [N, 3] Tensor
        # 2. 裏面候補点の計算 (P' = P - n * d)
        offset_points = vertices - normals * thickness

        # 2. 3D座標を「ボクセル整数座標」に変換
        # 座標を voxel_size で割って整数（Int32）にする
        voxel_coords = (offset_points / voxel_size).floor().to(o3c.int32)
    
        # 3. ハッシュマップからインデックスを取得
        # vbg.hashmap() は T-Hashmap オブジェクトを返します
        # find メソッドは、[N, 3] の座標に対し、[N] のインデックスと [N] のマスクを返します
        buf_indices, masks = self.vbg.hashmap().find(voxel_coords)
    
        # 3. 属性（TSDF, Weight）の取得
        # self.vbg.attribute("tsdf") は全ての有効ボクセルのTSDF値を持つ巨大な1次元Tensor
        tsdf_all = self.vbg.attribute("tsdf")
        weight_all = self.vbg.attribute("weight")
        
        # 4. クエリした座標に対応する値を抽出
        # mask はその座標にボクセルが存在するかどうかの真偽値
        # buf_indices はそのボクセルが属性Tensorの何番目にあるかのインデックス
        q_tsdf = o3c.Tensor(np.zeros(len(offset_points), dtype=np.float32), device=device)
        q_weight = o3c.Tensor(np.zeros(len(offset_points), dtype=np.float32), device=device)
    
        # 有効な（ボクセルが存在する）場所だけ値をコピー
        q_tsdf[masks] = tsdf_all[buf_indices[masks]].reshape((-1,))
        q_weight[masks] = weight_all[buf_indices[masks]].reshape((-1,))

        # 5. 安全判定 (空洞でない場所 = 重みが0、またはTSDFが0以下)
        # 条件: 重みが0 (未知) または TSDFが0以下 (内部/裏側)
        # つまり「(TSDF > 0 かつ Weight > 0) ではない」場所
        is_free_space = (q_tsdf > 0) & (q_weight > 0)
        safe_mask = is_free_space.reshape(-1).logical_not()
    
        # マスクを適用して安全な裏面頂点のみ抽出
        safe_offset_points = offset_points[safe_mask]
    
        # 5. 表面と（安全な）裏面を統合した点群を作成
        # combined_points = o3c.Tensor.concatenate([vertices, safe_offset_points], axis=0)
        # o3c.Tensor.concatenate ではなく、o3c.concatenate を使用
        combined_points = o3c.concatenate([vertices, safe_offset_points], axis=0)


        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point.positions = combined_points
        return pcd

    def alpha_shape(self, pcd, alpha=0.08):
        pcd_legacy = pcd.to_legacy()
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_legacy, alpha)

        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()

        for i in range(len(cluster_n_triangles)):
            component_mask = np.array(triangle_clusters) == i
            component_mesh = copy.deepcopy(mesh)
            component_mesh.remove_triangles_by_mask(~component_mask)
            component_mesh.remove_unreferenced_vertices()
            print(f"Component {i}: {cluster_n_triangles[i]} triangles, area: {cluster_area[i]:.4f}")
            if cluster_n_triangles[i] > 7:
                o3d.io.write_triangle_mesh(f"component_{i}.ply", component_mesh)

    def save_mesh_callback(self, request, response):
        self.save_mesh()
        pcd = self.create_safe_thick_pcd(thickness=0.05)
        if pcd is not None:
            o3d.io.write_point_cloud("safe_thick_pcd.ply", pcd.to_legacy())
            self.alpha_shape(pcd, alpha=0.08)
        response.success = True
        response.message = "Mesh saved successfully."
        return response

def main(args=None):
    rclpy.init(args=args)
    node = TSDFIntegrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_mesh()
        pcd = node.create_safe_thick_pcd(thickness=0.05)
        if pcd is not None:
            o3d.io.write_point_cloud("safe_thick_pcd.ply", pcd.to_legacy())
            node.alpha_shape(pcd, alpha=0.08)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
