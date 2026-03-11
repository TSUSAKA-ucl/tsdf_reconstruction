#! /usr/bin/env python3
import sys
import open3d as o3d
import open3d.core as o3c
import numpy as np

devgpu = o3c.Device("CUDA:0")
devcpu = o3c.Device("CPU:0")

load_path = sys.argv[1] if len(sys.argv) > 1 else "tsdf_vbg.zkey.npz"

raw_data = np.load(load_path)
voxel_size_orig = raw_data['voxel_size'].item()
block_resolution_orig = raw_data['block_resolution'].item()
vbg_orig = o3d.t.geometry.VoxelBlockGrid.load(load_path)
# メッシュを作成してビジュアライズする
mesh = vbg_orig.extract_triangle_mesh()
mesh.compute_vertex_normals()
legacy_mesh = mesh.to_legacy()
o3d.visualization.draw_geometries([legacy_mesh])

vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size_orig,
            block_resolution=block_resolution_orig,
            block_count=25000,
            device=devgpu
)
# あたらしく大きなblock_countのVBGを作成して、元のVBGからアクティブなキーをコピーする
active_keys = vbg_orig.hashmap().key_tensor()
vbg.hashmap().activate(active_keys)

indices_orig, _ = vbg_orig.hashmap().find(active_keys)
indices_new, _ = vbg.hashmap().find(active_keys)
for attr_name in ['tsdf', 'weight', 'color']:
    print(f"Copying attribute '{attr_name}'...")
    # vbg.attribute(attr_name)[indices] = vbg_orig.attribute(attr_name)[indices_orig] # GPUがOOMする
    buffer_orig = vbg_orig.attribute(attr_name).cpu().numpy()
    buffer_new = vbg.attribute(attr_name).cpu().numpy()
    idx_orig = indices_orig.cpu().numpy().flatten()
    idx_new = indices_new.cpu().numpy().flatten()
    buffer_new[idx_new] = buffer_orig[idx_orig]
    vbg.attribute(attr_name)[:] = buffer_new
# 新しいvbgからメッシュを作成してビジュアライズする
mesh = vbg.extract_triangle_mesh()
mesh.compute_vertex_normals()
legacy_mesh = mesh.to_legacy()
o3d.visualization.draw_geometries([legacy_mesh])

##### ステップ1：2m角の全ブロックを確保
res_block = voxel_size_orig * block_resolution_orig
x_range = np.arange(-1.0, 1.0, res_block)
y_range = np.arange(-1.0, 1.0, res_block)
z_range = np.arange(0.0, 2.0, res_block)

X, Y, Z = np.meshgrid(x_range, y_range, z_range)
# ブロックの整数インデックスに変換
block_keys = np.stack([X, Y, Z], axis=-1).reshape(-1, 3) / res_block
block_keys = o3c.Tensor(block_keys.astype(np.int32), device=devgpu)

# ハッシュマップを拡張
vbg.hashmap().activate(block_keys)

##### ステップ2：全ボクセルの座標とインデックスを取得
# 全アクティブボクセル（既存＋新規追加分）を取得
coords, indices = vbg.voxel_coordinates_and_flattened_indices()
indices = indices.reshape((-1,)) # 1次元化

# 属性バッファへの参照
tsdf_buf = vbg.attribute("tsdf")
weight_buf = vbg.attribute("weight")

##### ステップ3：全ボクセルの座標に基づいてTSDFと重みを設定
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh.cpu())
