#include "poisson_mesh_node.hpp"
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/poisson.h>
#include <pcl/io/ply_io.h>
#include <chrono>
#include <time.h>

PoissonMeshNode::PoissonMeshNode() : Node("poisson_mesh_node"), running_(true), has_new_data_(false) {
    auto cb_group = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    auto options = rclcpp::SubscriptionOptions();
    options.callback_group = cb_group;

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/rtabmap/cloud_map", 1, 
        std::bind(&PoissonMeshNode::cloud_callback, this, std::placeholders::_1), options);

    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("~/mesh_marker", 1);
    
    latest_cloud_ = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    processing_thread_ = std::thread(&PoissonMeshNode::worker_thread, this);
}

PoissonMeshNode::~PoissonMeshNode() {
    running_ = false;
    if (processing_thread_.joinable()) processing_thread_.join();
}

void PoissonMeshNode::cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    pcl::fromROSMsg(*msg, *latest_cloud_);
    has_new_data_ = true;
    RCLCPP_INFO(this->get_logger(), "New cloud received.");
}

void PoissonMeshNode::worker_thread() {
    while (running_) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_to_process(new pcl::PointCloud<pcl::PointXYZ>);
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            if (has_new_data_) {
                pcl::copyPointCloud(*latest_cloud_, *cloud_to_process);
                has_new_data_ = false;
            }
        }

        if (!cloud_to_process->empty()) {
            process_cloud(cloud_to_process);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void PoissonMeshNode::process_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  // 0. オフラインテストのため、日付時刻をファイル名にしたファイルに、点群をPLY ASCIIで書き出す
  auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm lt = *std::localtime(&now);
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &lt);
  const std::string s(buffer);
  std::string filename = "cloud_" + s + ".ply";
  pcl::io::savePLYFileASCII(filename, *cloud);
  
    // 1. PassThrough フィルタ (距離制限)
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(0.0, 1.0);
    pass.filter(*cloud);

    // 2. VoxelGrid (軽量化)
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    vg.filter(*cloud);

    // 3. Statistical Outlier Removal (ノイズ除去)
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud);

    // 4. 法線推定
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(20);
    ne.setViewPoint(0, 0, 0); // カメラ視点の設定
    ne.compute(*normals);

    // 法線付き点群へ結合
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    // 5. Poisson復元
    pcl::Poisson<pcl::PointNormal> poisson;
    poisson.setDepth(8); // 解像度 (8-10程度)
    poisson.setInputCloud(cloud_with_normals);
    pcl::PolygonMesh mesh;
    poisson.reconstruct(mesh);

    // 6. Marker (TRIANGLE_LIST) への変換
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "camera_link"; // RTAB-Mapの座標系に合わせる
    marker.header.stamp = this->now();
    marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 1.0; marker.scale.y = 1.0; marker.scale.z = 1.0;
    marker.color.a = 0.8; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0;

    pcl::PointCloud<pcl::PointXYZ> mesh_points;
    pcl::fromPCLPointCloud2(mesh.cloud, mesh_points);

    for (const auto& polygon : mesh.polygons) {
        for (const auto& index : polygon.vertices) {
            geometry_msgs::msg::Point p;
            p.x = mesh_points.points[index].x;
            p.y = mesh_points.points[index].y;
            p.z = mesh_points.points[index].z;
            marker.points.push_back(p);
        }
    }
    marker_pub_->publish(marker);
    RCLCPP_INFO(this->get_logger(), "Mesh published with %zu triangles.", mesh.polygons.size());
}
