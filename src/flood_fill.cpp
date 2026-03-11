#include <Open3D/Open3D.h>
#include <queue>
#include <unordered_set>
using namespace open3d;
using namespace open3d::t;
using namespace open3d::t::geometry;
using namespace std;
// voxel座標⽤
struct VoxelCoord {
  int x, y, z;
  bool operator==(const VoxelCoord& other) const {
    return x==other.x && y==other.y && z==other.z;
  }
};
// ハッシュ関数
namespace std {
  template <>
  struct hash<VoxelCoord> {
    size_t operator()(const VoxelCoord& v) const {
      return ((size_t)v.x * 73856093) ^
	((size_t)v.y * 19349663) ^
	((size_t)v.z * 83492791);
    }
  };
}
int main() {
  // 1. VoxelBlockGrid load
  auto vbg = VoxelBlockGrid::LoadFromFile("saved_voxelblockgrid.ply"); // 形式に応じて変更
  // 2. AABB 範囲指定
  Eigen::Vector3i min_bound(0,0,0);
  Eigen::Vector3i max_bound(399,399,399); // 400^3
  // int block_size = vbg.GetVoxelBlockSize(); // 例: 16
  int block_size = 16;
  auto hash_map = vbg.GetHashMap();
  auto active_blocks = hash_map->GetActiveIndices(); // active block のみ
  // 3. AABB 外側に接する block は weight=0
  for (auto &block_idx : active_blocks) {
    Eigen::Vector3i block_min = block_idx * block_size;
    Eigen::Vector3i block_max = block_min + Eigen::Vector3i(block_size-1,block_size-1,block_size-1);
    bool outside = (block_max.x() < min_bound.x() || block_min.x() > max_bound ||
		    block_max.y() < min_bound.y() || block_min.y() > max_bound ||
		    block_max.z() < min_bound.z() || block_min.z() > max_bound);
    if (outside) {
      vbg.GetBlockWeight(block_idx).Fill(0.0f); // 外側 weight=0vbg.GetBlockWeight(block_idx).Fill(0.0f); // 外側 weight=0
    }
  }
  // 4. flood fill BFS
							    queue<VoxelCoord> q;
							    unordered_set<VoxelCoord> visited;
							    // 外周の active voxel を enqueue
							    for (auto &block_idx : active_blocks) {
							      auto block_tsdf = vbg.GetBlockTSDF(block_idx);
							      auto block_weight = vbg.GetBlockWeight(block_idx);
							      for (int xi=0; xi<block_size; ++xi)
								for (int yi=0; yi<block_size; ++yi)
								  for (int zi=0; zi<block_size; ++zi) {
								    // weight>0 の voxel のみ
								    if (block_weight(xi,yi,zi) > 0.0f) {
								      VoxelCoord vc{block_idx.x()*block_size + xi,
										    block_idx.y()*block_size + yi,
										    block_idx.z()*block_size + zi};
								      // 外周かつ未訪問
								      if ((vc.x==min_bound.x() || vc.x==max_bound.x() ||
									   vc.y==min_bound.y() || vc.y==max_bound.y() ||
									   vc.z==min_bound.z() || vc.z==max_bound.z()) &&
									  visited.count(vc)==0) {
									q.push(vc);
									visited.insert(vc);
								      }
								    }
								  }
							    }
							    const int dx[6] = {1,-1,0,0,0,0};
							    const int dy[6] = {0,0,1,-1,0,0};
							    const int dz[6] = {0,0,0,0,1,-1};
							    while (!q.empty()) {
							      VoxelCoord v = q.front(); q.pop();
							      for (int i=0;i<6;i++) {
								VoxelCoord n{v.x+dx[i], v.y+dy[i], v.z+dz[i]};
								if (n.x < min_bound.x() || n.x > max_bound.x() ||
								    n.y < min_bound.y() || n.y > max_bound.y() ||
								    n.z < min_bound.z() || n.z > max_bound.z())
								  continue;
								// n の voxel が active block 内か確認
								Eigen::Vector3i n_block_idx = n / block_size;
								if (!hash_map->Has(n_block_idx))if (!hash_map->Has(n_block_idx))
												  continue; // 未定義ブロックは already outside
								auto block_weight = vbg.GetBlockWeight(n_block_idx);
								Eigen::Vector3i n_local = n - n_block_idx*block_size;
								if (block_weight(n_local.x(), n_local.y(), n_local.z()) > 0.0f &&
								    visited.count(n)==0) {
								  visited.insert(n);
								  q.push(n);
								}
							      }
							    }
							    // 5. flood fill の結果を TSDF に反映
							    for (auto &block_idx : active_blocks) {
							      auto block_tsdf = vbg.GetBlockTSDF(block_idx);
							      for (int xi=0; xi<block_size; ++xi)
								for (int yi=0; yi<block_size; ++yi)
								  for (int zi=0; zi<block_size; ++zi) {
								    VoxelCoord vc{block_idx.x()*block_size + xi,
										  block_idx.y()*block_size + yi,
										  block_idx.z()*block_size + zi};
								    if (visited.count(vc))
								      block_tsdf(xi,yi,zi) = 1.0f; // outside
								    else
								      block_tsdf(xi,yi,zi) = -1.0f; // inside
								  }
							    }
							    // 6. メッシュ抽出
							    auto mesh = vbg.ExtractTriangleMesh();
							    io::WriteTriangleMesh("mesh_after_floodfill.ply", *mesh);
							    return 0;
							    }
