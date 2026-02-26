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


def visualize_vbg_debug(voxel_pcd, filename="vbg_debug_color.ply"):
    device = voxel_pcd.device
    
    if 'tsdf' not in voxel_pcd.point:
        print("Error: TSDF attribute missing.")
        return

    tsdf = voxel_pcd.point['tsdf'].reshape(-1) # [N] に平坦化
    num_points = len(tsdf)

    # カラー用Tensorの初期化 [N, 3]
    colors = o3c.Tensor.full((num_points, 3), 0.5, o3c.float32, device)

    # マスク作成
    pos_mask = tsdf > 0  # 外部
    neg_mask = tsdf < 0  # 内部/裏側
    
    # 色の代入
    colors[pos_mask] = o3c.Tensor([0.0, 0.0, 1.0], device=device) # 青
    colors[neg_mask] = o3c.Tensor([1.0, 0.0, 0.0], device=device) # 赤

    # point 属性に 'colors' キーで追加
    voxel_pcd.point['colors'] = colors

    # 保存
    o3d.t.io.write_point_cloud(filename, voxel_pcd)
    print(f"Saved: {filename}. Red points show the thick 'back-wall'.")


class TSDFIntegrator(Node):
    def __init__(self):
        super().__init__('tsdf_integrator')
        
        # VoxelBlockGridの初期化
        self.voxel_size = 0.015  # 15mm
        self.block_res = 16
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=self.voxel_size,
            block_resolution=self.block_res,   # 16^3 voxels per block
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

    def visualize_vbg_voxels(self, vbg_to_check, filename="vbg_debug.ply"):
        device = vbg_to_check.hashmap().device
        voxel_size = self.voxel_size
        
        # 1. 有効な（データが入っている）ハッシュのエントリを取得
        active_indices = vbg_to_check.hashmap().active_buf_indices().to(o3c.int64)
        # ボクセルブロックの座標 (16x16x16単位の座標)
        block_coords = vbg_to_check.hashmap().key_tensor()[active_indices] 
        
        # 簡便な方法: 各ボクセルのTSDF値と座標を直接抜き出す
        tsdf_all = vbg_to_check.attribute("tsdf")
        weight_all = vbg_to_check.attribute("weight")
        
        # 有効なボクセルの物理座標を計算
        # (block_coord * res + local_offset) * voxel_size
        # 簡易版: 全アクティブボクセルのインデックスから座標を復元
        voxel_pcd = vbg_to_check.extract_point_cloud(weight_threshold=0.0) # 標準の点群抽出
        voxel_pcd_orig = self.vbg.extract_point_cloud(weight_threshold=0.0) # 標準の点群抽出
        
        # もし標準の extract_point_cloud が表面しか出さない場合は、以下で全ボクセルを可視化
        # 属性テンソルから直接色を付けて保存
        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point.positions = voxel_pcd.point.positions
        # visualize_vbg_debug(pcd)
        o3d.t.io.write_point_cloud(filename, pcd)
        print(f"Saved: {filename}. Red points show the thick 'back-wall'.")

        filenameo = "origianl.ply"
        pcdo = o3d.t.geometry.PointCloud(device)
        pcdo.point.positions = voxel_pcd_orig.point.positions
        o3d.t.io.write_point_cloud(filenameo, voxel_pcd_orig)
        print(f"Saved: {filenameo}. Red points show the thick 'back-wall'.")

    def extract_closed_thick_mesh(self, thickness=0.08):
        device = self.vbg.hashmap().device
        voxel_size = self.voxel_size

        # 1. 現在の表面を抽出（位置と法線を得るため）
        surface_mesh = self.vbg.extract_triangle_mesh()
        if len(surface_mesh.vertex.positions) == 0:
            return None
    
        vertices = surface_mesh.vertex.positions
        normals = surface_mesh.vertex.normals

        # 2. VBGの複製（GPU内コピー: 高速）
        # vbg_thick = self.vbg  # アルゴリズム確認のためコピーを諦める
        vbg_thick = self.vbg.to(device)

        # 3. 「内部」とみなす裏側の点を計算
        # 厚みの終着点 (P' = P - n * thickness)
        offset_points = vertices - normals * thickness
        v_coords = (offset_points / voxel_size).floor().to(o3c.int32)
        b_coords = (v_coords.to(o3c.float32) / self.block_res).floor().to(o3c.int32)
        # # 4. 「空洞チェック」: すでに何もないとわかっている場所には書き込まない
        # # 元のVBG(self.vbg)で状態を確認
        # indices, masks = self.vbg.hashmap().find(b_coords)
        # print(f"XX Total offset points: {len(offset_points)}")
        # # デフォルトは「未知(重み0)」
        # q_tsdf = o3c.Tensor.zeros((len(offset_points), 1), o3c.float32, device)
        # q_weight = o3c.Tensor.zeros((len(offset_points), 1), o3c.float32, device)
        # print(f"YY Total offset points: {len(q_tsdf)}")
        # 観測済みの場所だけ値をセット
        # valid_mask = masks.reshape(-1)
        # print(f"ZZ1 valid_mask size: {len(valid_mask)}")
        # q_tsdf[valid_mask] = self.vbg.attribute("tsdf")[indices[valid_mask].to(o3c.int64)]
        # q_weight[valid_mask] = self.vbg.attribute("weight")[indices[valid_mask].to(o3c.int64)]
        # print(f"XX Offset points: {len(offset_points)}, Valid mask count: {valid_mask.sum().item()}")
        # # 「確実な空洞」ではない場所（未知、または既に内部）のマスク
        # # (TSDF > 0 かつ Weight > 0) ではない場所
        # safe_to_write_mask = ( (q_tsdf > 0) & (q_weight > 0) ).logical_not().reshape(-1)
        # final_v_coords = v_coords[safe_to_write_mask]
        # final_b_coords = b_coords[safe_to_write_mask]
        # final_coords_to_write = voxel_coords[safe_to_write_mask]
        final_v_coords = v_coords
        final_b_coords = b_coords
        print(f"ZZ Total offset points: {len(offset_points)}, Safe to write: {len(final_v_coords)}, sizeof block coords: {len(final_b_coords)}")

        # 5. 複製したVBGに「内部(TSDF < 0)」を強制書き込み
        # if len(final_coords_to_write) > 0:
        #     # 新しい領域を確保
        #     vbg_thick.hashmap().activate(final_coords_to_write)
        #     # インデックスを再取得して値を固定
        #     new_indices, new_masks = vbg_thick.hashmap().find(final_coords_to_write)
        #     vbg_thick.attribute("tsdf")[new_indices.to(o3c.int64)] = o3c.Tensor([-1.0], device=device)
        #     vbg_thick.attribute("weight")[new_indices.to(o3c.int64)] = o3c.Tensor([10.0], device=device)
        # 5. 複製したVBGに「内部(TSDF < 0)」を強力に書き込み
        o3c.cuda.synchronize(device)
        if len(final_v_coords) > 0:
            # 新しい領域を確保
            vbg_thick.hashmap().activate(final_b_coords)
            new_indices, new_masks = vbg_thick.hashmap().find(final_b_coords)
            valid_mask = new_masks.reshape(-1)
            if valid_mask.any():
                safe_indices = new_indices[valid_mask].reshape(-1).to(o3c.int64)
                safe_b_coords = final_b_coords[valid_mask]
                safe_v_coords = final_v_coords[valid_mask]
                # ブロック内のローカルなvoxel座標を計算(0-15)
                # local_v_coords = final_v_coords % self.block_res
                local_v_coords = (safe_v_coords - (safe_b_coords * self.block_res)).to(o3c.int64)
                local_indices = (local_v_coords[:, 0] * self.block_res * self.block_res +
                                 local_v_coords[:, 1] * self.block_res +
                                 local_v_coords[:, 2]).reshape(-1)
                # 最終的なアトリビュートインデックスは、ブロックの開始インデックス + ローカルインデックス
                final_indices = (safe_indices * self.block_res**3 + local_indices).reshape(-1)

                val_tsdf = o3c.Tensor.full((len(final_indices),1), -1.0, o3c.float32, device=device)
                val_weight = o3c.Tensor.full((len(final_indices),1), 100.0, o3c.float32, device=device) # 極端に大きい
                o3c.cuda.synchronize(device)
                print(f"Activating {len(final_indices)} voxels for thickening.")
                vbg_thick.attribute("tsdf")[final_indices] = val_tsdf.reshape(-1)
                vbg_thick.attribute("weight")[final_indices] = val_weight.reshape(-1)
                o3c.cuda.synchronize(device)

        print(f"Now visualizing the thickened VBG voxels...")
        pcd = vbg_thick.extract_point_cloud(weight_threshold=0.0)
        o3d.t.io.write_point_cloud("vbg_thick_debug2.ply", pcd)
        # self.visualize_vbg_voxels(vbg_thick, filename="vbg_thick_debug2.ply")
        print(f"Saved: vbg_thick_debug2.ply. Red points show the thick 'back-wall'.")

        # # 6. GPU内でマーチングキューブ（超高速）
        # thick_mesh_gpu = vbg_thick.extract_triangle_mesh() # VRAM不足
        # # vbg_cpu = vbg_thick.cpu()
        # # thick_mesh_cpu = vbg_cpu.extract_triangle_mesh()

        # # 7. Legacy変換して保存
        # mesh_legacy = thick_mesh_gpu.to_legacy()
        # # mesh_legacy = thick_mesh_cpu.to_legacy()
        # mesh_legacy.compute_vertex_normals()
    
        # # 後処理（物体分離）へ
        # self.process_and_save_components(mesh_legacy)
        # return mesh_legacy

    def process_and_save_components(self, mesh):
        # 独立した塊（物体）ごとに分離
        triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)

        for i in range(len(cluster_n_triangles)):
            if cluster_n_triangles[i] < 10: # 小さなゴミを除去
                continue
                
            component_mask = (triangle_clusters == i)
            component_mesh = copy.deepcopy(mesh)
            component_mesh.remove_triangles_by_mask(~component_mask)
            component_mesh.remove_unreferenced_vertices()
            
            print(f"Component {i}: {cluster_n_triangles[i]} triangles, area: {cluster_area[i]:.4f}")
            
            # 保存 (CoACDに渡す用)
            o3d.io.write_triangle_mesh(f"component_m{i}.ply", component_mesh)

    def save_mesh_callback(self, request, response):
        self.save_mesh()
        self.extract_closed_thick_mesh()
        response.success = True
        response.message = "Mesh saved successfully."
        return response

def main(args=None):
    rclpy.init(args=args)
    node = TSDFIntegrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
