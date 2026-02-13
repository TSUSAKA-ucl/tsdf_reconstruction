#include <memory>
#include "poisson_mesh_node.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // マルチスレッドエグゼキューターを使用
  // これにより、CallbackGroup(Reentrant)で指定した並列処理が可能になります
  auto executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  auto node = std::make_shared<PoissonMeshNode>();

  executor->add_node(node);
  
  RCLCPP_INFO(node->get_logger(), "Poisson Mesh Node has started.");
  
  executor->spin();

  rclcpp::shutdown();
  return 0;
}
