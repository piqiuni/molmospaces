#include "explore_pkg/voronoi_explorer_visualization.h"
#include "explore_pkg/voronoi_explorer.h"  // 需要VoronoiNode的完整定义
#include <ros/ros.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <std_msgs/ColorRGBA.h>

namespace explore_pkg
{

VoronoiExplorerVisualization::VoronoiExplorerVisualization()
  : initialized_(false)
{
}

VoronoiExplorerVisualization::~VoronoiExplorerVisualization()
{
}

bool VoronoiExplorerVisualization::initialize(ros::NodeHandle& nh)
{
  scored_marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>(
      "/exploration_planner/scored_voronoi_markers", 1, true);
  tsp_path_pub_ = nh.advertise<visualization_msgs::Marker>(
      "/exploration_planner/tsp_exploration_path", 1, true);
  
  initialized_ = true;
  ROS_INFO("VoronoiExplorerVisualization initialized");
  return true;
}

void VoronoiExplorerVisualization::publishScoredMarkers(
    const std::vector<VoronoiNode>& nodes,
    const geometry_msgs::Pose& current_pose,
    const nav_msgs::OccupancyGrid& score_map)
{
  if (!initialized_ || nodes.empty()) {
    return;
  }
  
  visualization_msgs::MarkerArray marker_array;
  
  // 计算所有节点的评分，找到最小值和最大值用于归一化
  std::vector<double> scores;
  scores.reserve(nodes.size());
  
  for (const auto& node : nodes) {
    // 使用信息增益作为评分
    scores.push_back(node.information_gain);
  }
  
  if (scores.empty()) {
    return;
  }
  
  double min_score = *std::min_element(scores.begin(), scores.end());
  double max_score = *std::max_element(scores.begin(), scores.end());
  double score_range = max_score - min_score;
  
  // 归一化参数：柱子高度范围（米）
  const double min_height = 0.1;
  const double max_height = 1.5;
  const double height_range = max_height - min_height;
  
  // 创建Marker并归一化高度
  int marker_id = 0;
  for (size_t i = 0; i < nodes.size(); i++) {
    const auto& node = nodes[i];
    double raw_score = scores[i];
    
    // 归一化评分到 [0, 1]
    double normalized_score = 0.0;
    if (score_range > 1e-6) {
      normalized_score = (raw_score - min_score) / score_range;
    } else {
      normalized_score = 0.5;
    }
    
    // 映射到实际高度范围
    double cylinder_height = min_height + normalized_score * height_range;
    
    visualization_msgs::Marker nodes_marker;
    nodes_marker.header.frame_id = score_map.header.frame_id;
    nodes_marker.header.stamp = ros::Time::now();
    nodes_marker.ns = "scored_voronoi_nodes";
    nodes_marker.id = marker_id++;
    nodes_marker.type = visualization_msgs::Marker::CYLINDER;
    nodes_marker.action = visualization_msgs::Marker::ADD;
    nodes_marker.lifetime = ros::Duration(0);
    
    // 柱子样式
    nodes_marker.scale.x = 0.2;  // 柱子半径
    nodes_marker.scale.y = 0.2;
    nodes_marker.scale.z = cylinder_height;  // 归一化后的高度
    
    // 根据评分设置颜色（绿色渐变，评分越高越亮）
    nodes_marker.color.r = 0.0;
    nodes_marker.color.g = 0.3 + normalized_score * 0.7;  // 0.3到1.0的绿色
    nodes_marker.color.b = 0.0;
    nodes_marker.color.a = 0.8;
    
    // 设置位置：柱子底部在地面，顶部在z=height处
    nodes_marker.pose.position = node.position;
    nodes_marker.pose.position.z = cylinder_height / 2.0;  // 柱子中心在高度的一半处
    nodes_marker.pose.orientation.w = 1.0;
    
    marker_array.markers.push_back(nodes_marker);
  }
  
  // 发布Marker数组
  if (!marker_array.markers.empty()) {
    scored_marker_pub_.publish(marker_array);
    ROS_DEBUG("Published %zu scored Voronoi markers (score range: %.3f - %.3f)",
              marker_array.markers.size(), min_score, max_score);
  }
}

void VoronoiExplorerVisualization::publishTSPPath(
    const std::vector<geometry_msgs::Point>& tsp_path,
    int current_index,
    const geometry_msgs::Point& current_goal_position,
    const std::string& frame_id)
{
  if (!initialized_ || tsp_path.empty()) {
    return;
  }
  
  // 创建路径Marker（绿色线条）
  visualization_msgs::Marker path_marker;
  path_marker.header.frame_id = frame_id;
  path_marker.header.stamp = ros::Time::now();
  path_marker.ns = "tsp_exploration_path";
  path_marker.id = 0;
  path_marker.type = visualization_msgs::Marker::LINE_STRIP;
  path_marker.action = visualization_msgs::Marker::ADD;
  path_marker.pose.orientation.w = 1.0;
  
  // 路径样式：绿色线条
  path_marker.scale.x = 0.1;  // 线宽（米）
  path_marker.color.r = 0.0;
  path_marker.color.g = 1.0;  // 绿色
  path_marker.color.b = 0.0;
  path_marker.color.a = 0.8;
  
  // 添加路径点
  for (const auto& point : tsp_path) {
    geometry_msgs::Point p = point;
    p.z = 0.2;  // 稍微抬高，便于在地图上显示
    path_marker.points.push_back(p);
  }
  
  // 如果路径是闭合的，添加第一个点以形成闭环
  if (path_marker.points.size() > 2) {
    path_marker.points.push_back(path_marker.points[0]);
  }
  
  // 创建节点Marker（显示路径上的节点，已遍历和未遍历用不同颜色）
  visualization_msgs::Marker nodes_marker;
  nodes_marker.header = path_marker.header;
  nodes_marker.ns = "tsp_path_nodes";
  nodes_marker.id = 1;
  nodes_marker.type = visualization_msgs::Marker::SPHERE_LIST;
  nodes_marker.action = visualization_msgs::Marker::ADD;
  nodes_marker.pose.orientation.w = 1.0;
  
  // 节点样式：球体直径
  nodes_marker.scale.x = 0.3;  // 球体直径（米）
  nodes_marker.scale.y = 0.3;
  nodes_marker.scale.z = 0.3;
  
  // 为每个节点设置颜色（已遍历：灰色，未遍历：黄色，当前：红色）
  std_msgs::ColorRGBA visited_color;    // 已遍历：灰色
  visited_color.r = 0.5;
  visited_color.g = 0.5;
  visited_color.b = 0.5;
  visited_color.a = 0.7;
  
  std_msgs::ColorRGBA unvisited_color;  // 未遍历：黄色
  unvisited_color.r = 1.0;
  unvisited_color.g = 1.0;
  unvisited_color.b = 0.0;
  unvisited_color.a = 0.9;
  
  std_msgs::ColorRGBA current_color;   // 当前目标：红色
  current_color.r = 1.0;
  current_color.g = 0.0;
  current_color.b = 0.0;
  current_color.a = 1.0;
  
  // 添加节点位置和颜色
  for (size_t i = 0; i < tsp_path.size(); i++) {
    geometry_msgs::Point point = tsp_path[i];
    point.z = 0.2;
    nodes_marker.points.push_back(point);
    
    // 根据是否已遍历设置颜色
    if (static_cast<int>(i) < current_index) {
      nodes_marker.colors.push_back(visited_color);
    } else if (static_cast<int>(i) == current_index) {
      nodes_marker.colors.push_back(current_color);
    } else {
      nodes_marker.colors.push_back(unvisited_color);
    }
  }
  
  // 创建文本标记显示节点编号
  visualization_msgs::MarkerArray text_markers;
  for (size_t i = 0; i < tsp_path.size(); i++) {
    visualization_msgs::Marker text_marker;
    text_marker.header = path_marker.header;
    text_marker.ns = "tsp_node_labels";
    text_marker.id = static_cast<int>(i);
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;
    text_marker.pose.position.x = tsp_path[i].x;
    text_marker.pose.position.y = tsp_path[i].y;
    text_marker.pose.position.z = 0.5;  // 在节点上方显示
    text_marker.pose.orientation.w = 1.0;
    
    // 文本内容：显示TSP遍历顺序编号
    text_marker.text = std::to_string(i);
    
    // 文本样式
    text_marker.scale.z = 0.3;  // 文本高度（米）
    if (static_cast<int>(i) < current_index) {
      // 已遍历的节点：灰色文本
      text_marker.color.r = 0.5;
      text_marker.color.g = 0.5;
      text_marker.color.b = 0.5;
    } else if (static_cast<int>(i) == current_index) {
      // 当前目标节点：红色文本（高亮）
      text_marker.color.r = 1.0;
      text_marker.color.g = 0.0;
      text_marker.color.b = 0.0;
      text_marker.scale.z = 0.4;  // 当前节点文本稍大
    } else {
      // 未遍历的节点：黄色文本
      text_marker.color.r = 1.0;
      text_marker.color.g = 1.0;
      text_marker.color.b = 0.0;
    }
    text_marker.color.a = 1.0;
    
    text_markers.markers.push_back(text_marker);
  }
  
  // 发布路径、节点和文本标记
  tsp_path_pub_.publish(path_marker);
  tsp_path_pub_.publish(nodes_marker);
  scored_marker_pub_.publish(text_markers);
  
  ROS_DEBUG("Published TSP path visualization: %zu nodes (%d visited, %zu remaining)",
            tsp_path.size(), current_index, tsp_path.size() - current_index);
}

void VoronoiExplorerVisualization::clearAllMarkers(const std::string& frame_id)
{
  if (!initialized_) {
    return;
  }

  visualization_msgs::Marker delete_all_marker;
  delete_all_marker.header.frame_id = frame_id.empty() ? "map" : frame_id;
  delete_all_marker.header.stamp = ros::Time::now();
  delete_all_marker.action = visualization_msgs::Marker::DELETEALL;

  tsp_path_pub_.publish(delete_all_marker);

  visualization_msgs::MarkerArray delete_all_array;
  delete_all_array.markers.push_back(delete_all_marker);
  scored_marker_pub_.publish(delete_all_array);
}

void VoronoiExplorerVisualization::worldToMap(double wx, double wy, int& mx, int& my,
                                              const nav_msgs::OccupancyGrid& map)
{
  mx = static_cast<int>((wx - map.info.origin.position.x) / map.info.resolution);
  my = static_cast<int>((wy - map.info.origin.position.y) / map.info.resolution);
}

bool VoronoiExplorerVisualization::isValid(int mx, int my, 
                                           const nav_msgs::OccupancyGrid& map)
{
  return mx >= 0 && mx < static_cast<int>(map.info.width) &&
         my >= 0 && my < static_cast<int>(map.info.height);
}

double VoronoiExplorerVisualization::calculateDistance(const geometry_msgs::Point& p1,
                                                       const geometry_msgs::Point& p2)
{
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  double dz = p1.z - p2.z;
  return sqrt(dx * dx + dy * dy + dz * dz);
}

}  // namespace explore_pkg

