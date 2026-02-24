cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0074 NEW) # PCL_ROOT などの変数の扱い

project(tsdf_reconst)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 依存パッケージの探索
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED) # これで include/pcl_conversions へのパスが通るはず
find_package(PCL REQUIRED COMPONENTS common io filters features surface search)

# 2. 実行ファイルの作成
add_executable(poisson_mesh_node src/poisson_mesh_node.cpp src/main.cpp)

# 3. インクルードパスの明示（念のため PCL を追加）
target_include_directories(poisson_mesh_node PRIVATE
  include
  ${PCL_INCLUDE_DIRS}
)

# 4. 依存関係のリンク（Jazzy の標準的な方法）
# ament_target_dependencies は内部で target_link_libraries を呼び出します。
# 競合を避けるため、ROSパッケージはここだけで管理するのが安全です。
ament_target_dependencies(poisson_mesh_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
)

# 5. システムライブラリ（PCL）のリンク
# PCL は ROS パッケージではないため、個別にリンクします。
target_link_libraries(poisson_mesh_node 
  ${PCL_LIBRARIES}
)

file(GLOB_RECURSE SCRIPT_FILES
  CONFIGURE_DEPENDS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "scripts/*.py"
  "scripts/*.sh"
)
install(
  FILES ${SCRIPT_FILES}
  DESTINATION lib/${PROJECT_NAME}
)
# install(
#   DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/
#   DESTINATION lib/${PROJECT_NAME}
#   FILES_MATCHING
#   PATTERN "*.sh"
# )
install(TARGETS
  poisson_mesh_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
