# ********************************
import open3d as o3d
import open3d.core as o3c
import numpy as np
import copy

def extract_closed_thick_mesh(self, thickness=0.03):
    device = self.vbg.hashmap().device
    voxel_size = self.voxel_size

    # 1. 現在の表面を抽出（位置と法線を得るため）
    surface_mesh = self.vbg.extract_triangle_mesh()
    if len(surface_mesh.vertex.positions) == 0:
        return None
    
    vertices = surface_mesh.vertex.positions
    normals = surface_mesh.vertex.normals

    # 2. VBGの複製（GPU内コピー: 高速）
    vbg_thick = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.uint8),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel_size,
        block_resolution=16,
        block_count=self.vbg.hashmap().capacity(),
        device=device
    )
    vbg_thick.hashmap().assign(self.vbg.hashmap())
    vbg_thick.attribute("tsdf").copy_from(self.vbg.attribute("tsdf"))
    vbg_thick.attribute("weight").copy_from(self.vbg.attribute("weight"))
    vbg_thick.attribute("color").copy_from(self.vbg.attribute("color"))

    # 3. 「内部」とみなす裏側の点を計算
    # 厚みの終着点 (P' = P - n * thickness)
    offset_points = vertices - normals * thickness
    voxel_coords = (offset_points / voxel_size).floor().to(o3c.int32)

    # 4. 「空洞チェック」: すでに何もないとわかっている場所には書き込まない
    # 元のVBG(self.vbg)で状態を確認
    indices, masks = self.vbg.hashmap().find(voxel_coords)
    
    # デフォルトは「未知(重み0)」
    q_tsdf = o3c.Tensor.zeros((len(offset_points), 1), o3c.float32, device)
    q_weight = o3c.Tensor.zeros((len(offset_points), 1), o3c.float32, device)
    
    # 観測済みの場所だけ値をセット
    valid_mask = masks.reshape(-1)
    q_tsdf[valid_mask] = self.vbg.attribute("tsdf")[indices[valid_mask].to(o3c.int64)]
    q_weight[valid_mask] = self.vbg.attribute("weight")[indices[valid_mask].to(o3c.int64)]

    # 「確実な空洞」ではない場所（未知、または既に内部）のマスク
    # (TSDF > 0 かつ Weight > 0) ではない場所
    safe_to_write_mask = ( (q_tsdf > 0) & (q_weight > 0) ).logical_not().reshape(-1)
    final_coords_to_write = voxel_coords[safe_to_write_mask]

    # 5. 複製したVBGに「内部(TSDF < 0)」を強制書き込み
    if len(final_coords_to_write) > 0:
        # 新しい領域を確保
        vbg_thick.hashmap().activate(final_coords_to_write)
        # インデックスを再取得して値を固定
        new_indices, new_masks = vbg_thick.hashmap().find(final_coords_to_write)
        vbg_thick.attribute("tsdf")[new_indices.to(o3c.int64)] = o3c.Tensor([-1.0], device=device)
        vbg_thick.attribute("weight")[new_indices.to(o3c.int64)] = o3c.Tensor([10.0], device=device)

    # 6. GPU内でマーチングキューブ（超高速）
    thick_mesh_gpu = vbg_thick.extract_triangle_mesh()
    
    # 7. Legacy変換して保存
    mesh_legacy = thick_mesh_gpu.to_legacy()
    mesh_legacy.compute_vertex_normals()
    
    # 後処理（物体分離）へ
    self.process_and_save_components(mesh_legacy)
    
    return mesh_legacy
