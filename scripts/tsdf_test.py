#!/usr/bin/env python3
import argparse
import open3d as o3d
import numpy as np

def run_tsdf_single_frame(ply_path):
    # 1. デバイスの設定 (最初はCPU)
    device = o3d.core.Device("CPU:0")
    
    # 2. PLY（平均化済み点群）の読み込み
    pcd = o3d.t.io.read_point_cloud(ply_path)
    pcd = pcd.to(device)
    pcd.estimate_normals(max_nn=30, radius=0.02)

    # 3. TSDF Voxel Grid の初期化
    # voxel_size: ボクセルの細かさ (1cm = 0.01)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3d.core.Dtype.Float32, o3d.core.Dtype.Float32, o3d.core.Dtype.Float32),
        attr_channels=((1), (1), (3)),
        voxel_size=0.01,
        block_resolution=16,
        block_count=10000,
        device=device
    )
    # 4. 視点情報（Extrinsic）と内参（Intrinsic）
    # シングルショットなので、カメラ位置は原点 [I|0] と仮定
    extrinsic = o3d.core.Tensor(np.eye(4), o3d.core.Dtype.Float32, device)
    
    # D435iの標準的な内参 (解像度に合わせて調整が必要)
    intrinsic = o3d.core.Tensor([
        [418, 0, 426.7],
        [0, 418, 241.8],
        [0, 0, 1]
    ], o3d.core.Dtype.Float32, device)

    block_coords = vbg.compute_unique_block_coordinates(pcd=pcd)
    # sdf_trunc: 表面からどれだけ厚みを持たせるか (3cm = 0.03)

    # 5. PointCloudからの統合 (Ray Castingを利用)
    # 本来はRGB-D画像からのintegrateがベストですが、PCDからも可能です
    vbg.integrate(block_coords, pcd, intrinsic, extrinsic,
                  sdf_trunc=0.03,
                  depth_max=1.5)
    # 6. メッシュの抽出 (Marching Cubes)
    mesh = vbg.extract_surface_mesh()
    
    # 7. 可視化 (Open3D Viewer)
    # メッシュと、元の点群（比較用）を表示
    o3d.visualization.draw_geometries([mesh.to_legacy()], 
                                      window_name="TSDF Result (Mesh)")

    # 8. (オプション) 凸包の生成テスト
    # meshはまだ凹みがあるので、これに対してCoACDをかける準備
    print(f"Mesh has {len(mesh.triangle.indices)} triangles.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSDF Single Frame Integration Test")
    parser.add_argument("--ply_path", type=str, default="frame_1770xxxx.ply",
                        help="Path to the input PLY file containing the point cloud.")
    args,unknown = parser.parse_known_args()
    # 前回の bag_to_ply で保存したファイルを指定
    run_tsdf_single_frame(args.ply_path)
