#! /usr/bin/env python3
import sys
import os
import pymeshlab as ml
import numpy as np
import coacd

def run_coacd_decomposition(input_path, output_dir):
    # 1. PyMeshLabで読み込みと最終調整
    ms = ml.MeshSet()
    ms.load_new_mesh(input_path)
    
    # 向きの不整合を修正（これをしないと分解が細かくなりすぎる場合があります）
    ms.apply_filter('meshing_re_orient_faces_coherently')
    
    m = ms.current_mesh()
    v = m.vertex_matrix().astype(np.float64)
    f = m.face_matrix().astype(np.int32)

    # 2. CoACDの実行
    # threshold: 値を大きくする(0.05~0.1)と、細かい凹凸を無視して大きなパーツにまとめます
    # max_convex_hull: 生成するパーツの最大数（任意）
    mesh = coacd.Mesh(v, f)
    result = coacd.run_coacd(mesh, threshold=0.05) 

    # 3. 分割されたパーツを個別に保存
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (res_v, res_f) in enumerate(result):
        part_ms = ml.MeshSet()
        part_ms.add_mesh(ml.Mesh(res_v, res_f))
        part_ms.save_current_mesh(f"{output_dir}/part_{i}.obj")
        print(f"Saved: part_{i}.obj")

def main():
    load_path = sys.argv[1] if len(sys.argv) > 1 else "tsdf_vbg_20260304_142835-decimate-no-hole.ply"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "convex_parts"
    run_coacd_decomposition(load_path, output_dir)

if __name__ == "__main__":
    main()
