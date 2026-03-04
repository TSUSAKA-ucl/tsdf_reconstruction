#! /usr/bin/env python3
import sys
import open3d as o3d
import open3d.core as o3c
import numpy as np

load_path = sys.argv[1] if len(sys.argv) > 1 else "tsdf_vbg.zkey.npz"

vbg = o3d.t.geometry.VoxelBlockGrid.load(load_path)
# メッシュを作成してビジュアライズする
mesh = vbg.extract_triangle_mesh()
mesh.compute_vertex_normals()
legacy_mesh = mesh.to_legacy()
o3d.visualization.draw_geometries([legacy_mesh])
