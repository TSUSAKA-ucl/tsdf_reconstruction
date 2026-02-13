#ifndef POISSON_MESH_NODE_HPP_
#define POISSON_MESH_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <mutex>
#include <thread>
#include <atomic>

class PoissonMeshNode : public rclcpp::Node {
public:
    PoissonMeshNode();
    ~PoissonMeshNode();

private:
    // ROS通信
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

    // ワーカースレッド関連
    void worker_thread();
    std::thread processing_thread_;
    std::atomic<bool> running_;
    std::atomic<bool> has_new_data_;
    
    std::mutex data_mutex_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr latest_cloud_;

    // メッシュ生成ロジック
    void process_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
};

#endif
