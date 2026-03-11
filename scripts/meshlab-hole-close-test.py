#! /usr/bin/env python3
import sys
import time
import open3d as o3d
import open3d.core as o3c
import pymeshlab
import numpy as np

def create_o3d_mesh_from_pymeshlab(m):
    vertices = m.vertex_matrix()
    faces = m.face_matrix()
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    return o3d_mesh

def visualize_pymeshlab(m):
    o3d_mesh = create_o3d_mesh_from_pymeshlab(m)
    o3d_mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([o3d_mesh])
                                                    
def o3d_watertight_check(m):
    faces = m.face_matrix()
    o3d_mesh = create_o3d_mesh_from_pymeshlab(m)
    is_watertight = o3d_mesh.is_watertight()
    is_edge_manifold = o3d_mesh.is_edge_manifold()
    print(f"Mesh has {len(o3d_mesh.vertices)} vertices, {len(o3d_mesh.triangles)} faces.")
    print(f"Is the mesh watertight? {is_watertight}")
    print(f"Is the mesh edge manifold? {is_edge_manifold}")
    if not is_watertight or not is_edge_manifold:
        non_manifold_edges = o3d_mesh.get_non_manifold_edges(allow_boundary_edges=False)
        non_boundary_edges = non_manifold_edges
        boundary_edges = o3d_mesh.get_non_manifold_edges(allow_boundary_edges=True)
        print(f"number of non_manifold edges: {len(non_manifold_edges)}")
        print(f"number of boundary edges: {len(boundary_edges)}")
        print(f"mesh is orientable? {o3d_mesh.is_orientable()}")
        triangle_clusters, cluster_n_triangles, cluster_area = o3d_mesh.cluster_connected_triangles()
        print(f"number of connected components: {len(cluster_n_triangles)}")
        print(f"  - component triangle counts: {np.asarray(cluster_n_triangles)}")
        # triangle_clustersのタイプとサイズを調べる
        print(f"triangle_clusters type: {type(triangle_clusters)}")
        print(f"cluster_n_triangles type: {type(cluster_n_triangles)}")
        cluster_talble = np.asarray(triangle_clusters)
        for i, count in enumerate(cluster_n_triangles):
            triangles_outside_cluster = np.where(cluster_talble != i)[0]
            print(f"len of triangles outside of Component {i}: {len(triangles_outside_cluster)}")
            # print(f"  - outside of Component {i}: triangle indices: {triangles_outside_cluster}")
            if i == 0:
                o3d_mesh.remove_triangles_by_index(triangles_outside_cluster)
                o3d_mesh.remove_unreferenced_vertices()
                # o3d_mesh_cluster = o3d_mesh.select_by_index(np.asarray(triangles_in_cluster))
                print(f"    - Component{i}  has {len(o3d_mesh.vertices)} vertices, {len(o3d_mesh.triangles)} faces.")
                print(f"    - Component {i}: is watertight? {o3d_mesh.is_watertight()}")
                print(f"    - Component {i}: is edge manifold? {o3d_mesh.is_edge_manifold()}")
                print(f"    - Component {i}: is orientable? {o3d_mesh.is_orientable()}")
                print(f"    - Component {i} is self-intersecting? {o3d_mesh.is_self_intersecting()}")
                # print(f"    - Component {i}'s volume: {o3d_mesh.get_volume()}")
        if len(non_manifold_edges) > 0:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d_mesh.vertices
            line_set.lines = non_manifold_edges
            line_set.paint_uniform_color([1, 0, 0])  # 赤色
            # o3d.visualization.draw_geometries([line_set])
            # エッジの分析
            print(f"--- Edge Analysis ---")
            print(f"Total problem edges: {len(boundary_edges)}")
            print(f"Pure boundary edges (potential holes): {len(boundary_edges) - len(non_manifold_edges)}")
            # 3. 各エッジがどの頂点をつないでいるかプリント
            # これで「4本」がバラバラか、繋がっているか分かります
            print(f"boundary edge indices: {np.asarray(boundary_edges)}")
            print(f"non boundary edge indices: {np.asarray(non_manifold_edges)}")
            # # 調査対象の頂点インデックスをユニークに抽出
            # problem_v_indices = set(non_manifold_edges.flatten())
            # print(f"Target Vertex IDs: {problem_v_indices}")

            # --- 2. PyMeshLabの現在のメッシュから面情報を抽出 ---
            problem_vertices = set(np.asarray(non_boundary_edges).flatten())
            found_faces = []
            for f_idx, face in enumerate(faces):
                intersect = set(face).intersection(problem_vertices)
                if intersect:
                    found_faces.append((f_idx, face, intersect))
            print(f"--- Faces sharing these vertices ({len(found_faces)} faces found) ---")
            # involved_verticesに found_facesのfaceに現れる頂点を全て集める
            involved_vertices = set()
            for f_idx, face, intersect in found_faces:
                involved_vertices.update(face)
                print(f"FaceID: {f_idx:6d} | Vertices: {face} | ProblemVerts: {intersect}")
            # involved_verticesを配列に変換
            involved_vertices = np.array(list(involved_vertices))
            if len(found_faces) > 0:
                visualize_problem_faces_with_backface(o3d_mesh, involved_vertices)

# sub_mesh_array = []
def visualize_problem_faces_with_backface(o3d_mesh, face_indices):
    # 1. 問題の面だけを抽出したサブメッシュを作成
    sub_mesh = o3d_mesh.select_by_index(face_indices)
#     sub_mesh_array.append(sub_mesh)  # グローバルリストに保存して、ガベージコレクションで消されないようにする
#     o3d.visualization.draw_geometries(sub_mesh_array)  # これでサブメッシュが表示されるはずです


# def submesh_visualize_problem_faces_with_backface(sub_mesh):
    # 2. 表側を「薄い緑」に塗る
    sub_mesh.paint_uniform_color([0.6, 1.0, 0.6])
    
    # 3. 裏側を強調するために、裏面用のメッシュ（法線を反転させたもの）を作成
    back_mesh = o3d.geometry.TriangleMesh()
    back_mesh.vertices = sub_mesh.vertices
    # 三角形の頂点順を入れ替えて法線を逆にする
    back_faces = np.asarray(sub_mesh.triangles)[:, [0, 2, 1]]
    back_mesh.triangles = o3d.utility.Vector3iVector(back_faces)
    # 裏側を「鮮やかな赤」に塗る
    back_mesh.paint_uniform_color([1.0, 0.2, 0.2])

    print(f"Showing {len(face_indices)} faces. Green=Front, Red=Back.")
    
    # 4. 裏表を表示
    o3d.visualization.draw_geometries([sub_mesh, back_mesh], 
                                      mesh_show_back_face=False, 
                                      window_name="Problem Area Analysis")
    # [W]キーでワイヤーフレーム表示を切り替えるとエッジがよく見えます
    o3d.visualization.draw_geometries([o3d_mesh, sub_mesh, back_mesh], 
                                      mesh_show_back_face=False, 
                                      window_name="Problem Area Analysis")


def list_faces_vertices_around_vertices(all_faces, vertices):
    found_faces = []
    for f_idx, face in enumerate(all_faces):
        intersect = set(face).intersection(vertices)
        if intersect:
            found_faces.append((f_idx, face, intersect))
    involved_vertices = set()
    for f_idx, face, intersect in found_faces:
        involved_vertices.update(face)
    involved_vertices = set(np.array(list(involved_vertices)).flatten())
    return found_faces, list(involved_vertices)

def list_non_manifold_edges_vertices(mesh):
    o3d_mesh = create_o3d_mesh_from_pymeshlab(mesh)
    if o3d_mesh.is_watertight() and o3d_mesh.is_edge_manifold():
        return [], []
    non_manifold_edges = o3d_mesh.get_non_manifold_edges(allow_boundary_edges=False)
    problem_vertices = set(np.asarray(non_manifold_edges).flatten())
    return non_manifold_edges, list(problem_vertices)

def repair_by_deletion_and_refill(ms, problem_v_indices):
    if (len(problem_v_indices) == 0):
        print("No problem vertices to repair.")
        return
    all_faces = ms.current_mesh().face_matrix()
    faces, involved_vertices = list_faces_vertices_around_vertices(all_faces, problem_v_indices)
    # 1. 問題の頂点（Open3Dで特定したもの）を「選択」状態にする
    # v_index_list は [1876, 2107, 2139, 2074] のようなリスト
    cond = " || ".join([f"vi == {i}" for i in problem_v_indices])
    print(f"Selecting vertices with condition: {cond}")
    ms.apply_filter('compute_selection_by_condition_per_vertex',
                    condselect=cond)
    ms.apply_filter('compute_selection_transfer_vertex_to_face')
    ms.apply_filter('apply_selection_dilatation') # 周囲の面も巻き込む
    # 2. 選択範囲を「面」に広げ、さらにその周囲1〜2段階の面も巻き込む
    # これにより、複雑に絡まった「谷」の部分を丸ごと空間から消去します
    # ms.apply_filter('selection_expand_face')
    # ms.apply_filter('selection_expand_face') # 2回実行して余裕を持って消すのがコツです

    # 3. 選択された「汚い構造」を削除
    ms.apply_filter('meshing_remove_selected_vertices_and_faces')
    
    # 4. 浮いた頂点を掃除
    ms.apply_filter('meshing_remove_unreferenced_vertices')

    # 5. これで「大きな、トポロジー的にクリーンな穴」が空いた状態になるので、
    # 穴埋めフィルタで平坦な蓋をする
    # maxholesize は、削除した範囲をカバーできる程度に大きく（例: 2000）
    # ms.apply_filter('meshing_close_holes', maxholesize=2000)

    # 6. 仕上げに、新しく張られた面を馴染ませる
    # ms.apply_filter('apply_coord_hc_laplacian_smoothing', iter=5)



# **************** Main Code ****************

devgpu = o3c.Device("CUDA:0")
devcpu = o3c.Device("CPU:0")

load_path = sys.argv[1] if len(sys.argv) > 1 else "tsdf_vbg.zkey.npz"

# PyMeshLabのMeshオブジェクトを作成
ms = pymeshlab.MeshSet()
ms.load_new_mesh(load_path)
print(f"vertex count of loaded mesh: {ms.current_mesh().vertex_number()}")
print(f"  face count of loaded mesh: {ms.current_mesh().face_number()}")
print(f" edge number of loaded mesh: {ms.current_mesh().edge_number()}")
ms.apply_filter('meshing_re_orient_faces_coherently')
o3d_watertight_check(ms.current_mesh())
# repairingフィルタを適用
# # 1. 非多様体エッジを直接修復（エッジを共有する面を分離・削除）
# ms.apply_filter('meshing_repair_non_manifold_edges')

# # 2. 非多様体頂点を修復（一点で接している構造を解消）
# ms.apply_filter('meshing_repair_non_manifold_vertices')

# # 3. 重複した頂点や面を掃除（これが原因で非多様体になることが多い）
# ms.apply_filter('meshing_merge_close_vertices')
# ms.apply_filter('meshing_remove_duplicate_faces')

# # 4. （オプション）浮遊している極小のパーツを削除
# # 全体の3%以下のサイズの独立したパーツを消去します
# ms.apply_filter('meshing_remove_connected_component_by_diameter', mincomponentdiag=pymeshlab.PercentageValue(3))
# # 5. （オプション）穴を埋める
# ms.apply_filter('meshing_close_holes', maxholesize=100)  # maxholesizeは埋める穴の最大サイズ（面数）
# # 追加1. 頂点のみ、あるいは面を構成していない浮遊エッジを削除
# ms.apply_filter('meshing_remove_unreferenced_vertices')
# # 追加2. 重複している面（Double Faces）を削除
# # これがあると、穴埋めアルゴリズムが「面がある」と誤認して動きません
# ms.apply_filter('meshing_remove_duplicate_faces')
# # 追加3. ゼロ面積の面（微小な縮退面）を削除
# ms.apply_filter('meshing_remove_null_faces')
# # 追加4. 非多様体エッジを再度修復（ここで4本のエッジが「切り離し」の対象になります）
# ms.apply_filter('meshing_repair_non_manifold_edges')
# ms.apply_filter('meshing_merge_close_vertices', threshold=pymeshlab.PercentageValue(0.01))
# ms.apply_filter('meshing_remove_duplicate_faces')
ms.apply_filter('meshing_repair_non_manifold_edges')
ms.apply_filter('meshing_repair_non_manifold_vertices')
ms.apply_filter('meshing_re_orient_faces_coherently')
ms.apply_filter('meshing_re_orient_faces_by_geometry')
# ms.apply_filter('meshing_repair_non_manifold_vertices')
ms.apply_filter('meshing_close_holes', maxholesize=100)
# ms.apply_filter("apply_coord_hc_laplacian_smoothing")
o3d_watertight_check(ms.current_mesh())

# problem_edges, problem_vertices = list_non_manifold_edges_vertices(ms.current_mesh())
# print(f"Number of non-manifold edges: {len(problem_edges)}")
# repair_by_deletion_and_refill(ms, problem_vertices)
o3d_watertight_check(ms.current_mesh())

visualize_pymeshlab(ms.current_mesh())
