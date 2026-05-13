#include "explore_pkg/voronoi_explorer.h"
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <map>
#include <cstring>

namespace explore_pkg
{

VoronoiExplorer::VoronoiExplorer()
  : voronoi_map_received_(false)
  , voronoi_map_topic_("/struct_mapping/voronoi_map")
  , score_map_received_(false)
  , score_map_topic_("/explore_manager/score_map")
  , scene_id_grid_received_(false)
  , scene_id_scores_yaml_path_("")
  , default_scene_score_(0)
  , tsp_current_index_(0)
  , tsp_goal_reach_threshold_(1.0)
  , tsp_goal_timeout_(60.0)
  , tsp_top_num_(0.3)
  , tsp_scene_score_weight_(0.3)
  , tsp_replan_frequency_(-1.0)
  , blacklist_match_threshold_(0.2)
{
}

VoronoiExplorer::~VoronoiExplorer()
{
}

bool VoronoiExplorer::initialize(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
{
  nh_ = nh;
  private_nh_ = private_nh;
  
  // 加载参数
  private_nh_.param("voronoi/voronoi_map_topic", voronoi_map_topic_, 
                    std::string("/struct_mapping/voronoi_map"));
  private_nh_.param("voronoi/score_map_topic", score_map_topic_, 
                    std::string("/explore_manager/score_map"));
  private_nh_.param("voronoi/tsp_goal_reach_threshold", tsp_goal_reach_threshold_, 1.0);
  private_nh_.param("voronoi/tsp_goal_timeout", tsp_goal_timeout_, 60.0);
  private_nh_.param("voronoi/tsp_top_num", tsp_top_num_, 0.3);
  private_nh_.param("voronoi/tsp_scene_score_weight", tsp_scene_score_weight_, 0.3);
  private_nh_.param("voronoi/tsp_replan_frequency", tsp_replan_frequency_, -1.0);
  private_nh_.param("voronoi/blacklist_match_threshold", blacklist_match_threshold_, 0.2);
  private_nh_.param("voronoi/scene_id_grid_topic", scene_id_grid_topic_, 
                    std::string("/semantic_mapping/scene_id_grid"));
  private_nh_.param("voronoi/scene_id_scores_json_path", scene_id_scores_yaml_path_, 
                    std::string("$(find explore_pkg)/config/scene_id_scores.json"));
  
  // 订阅Voronoi图和评分地图
  voronoi_map_sub_ = nh_.subscribe(voronoi_map_topic_, 1,
                                    &VoronoiExplorer::voronoiMapCallback, this);
  score_map_sub_ = nh_.subscribe(score_map_topic_, 1,
                                 &VoronoiExplorer::scoreMapCallback, this);
  scene_id_grid_sub_ = nh_.subscribe(scene_id_grid_topic_, 1,
                                    &VoronoiExplorer::sceneIdGridCallback, this);
  
  // 初始化Scene Score Grid发布者
  scene_score_grid_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(
      "/explore_manager/scene_score_grid", 1, true);
  
  // 加载Scene ID分数配置
  if (!loadSceneIdScores()) {
    ROS_WARN("Failed to load scene ID scores, using default score: %d", default_scene_score_);
  }
  
  // 初始化可视化（在cpp文件中包含完整定义后才能使用）
  visualization_.reset(new VoronoiExplorerVisualization());
  visualization_->initialize(nh_);
  
  // 初始化TSP重新规划定时器（如果频率>=0）
  if (tsp_replan_frequency_ > 0.0) {
    tsp_replan_timer_ = nh_.createTimer(
        ros::Duration(1.0 / tsp_replan_frequency_),
        &VoronoiExplorer::tspReplanTimerCallback, this);
    ROS_INFO("TSP replan timer started with frequency: %.2f Hz", tsp_replan_frequency_);
  } else {
    ROS_INFO("TSP replan timer disabled (frequency: %.2f)", tsp_replan_frequency_);
  }
  
  ROS_INFO("VoronoiExplorer initialized");
  ROS_INFO("  TSP goal reach threshold: %.2f m", tsp_goal_reach_threshold_);
  ROS_INFO("  TSP goal timeout: %.1f s", tsp_goal_timeout_);
  ROS_INFO("  TSP top percentage: %.0f%%", tsp_top_num_ * 100.0);
  ROS_INFO("  TSP scene score weight: %.2f", tsp_scene_score_weight_);
  ROS_INFO("  TSP replan frequency: %.2f Hz", tsp_replan_frequency_);
  ROS_INFO("  Blacklist match threshold: %.2f m", blacklist_match_threshold_);
  
  return true;
}

ExplorationGoal VoronoiExplorer::selectNextGoal(
    const nav_msgs::OccupancyGrid& map,
    const geometry_msgs::Pose& current_pose)
{
  ExplorationGoal goal;
  goal.is_valid = false;
  
  current_map_ = map;
  
  // 检查必要输入
  if (!voronoi_map_received_ || voronoi_nodes_.empty()) {
    ROS_WARN_THROTTLE(2.0, "Voronoi graph not available");
    return goal;
  }
  
  if (!score_map_received_) {
    ROS_WARN_THROTTLE(2.0, "Score map not received");
    return goal;
  }
  
  // 更新节点信息增益
    for (auto& node : voronoi_nodes_) {
      node.information_gain = calculateInformationGain(node, score_map_);
  }
  
  // 使用TSP路径选择目标
  goal = selectNextGoalFromTSP(current_pose.position);
  
  // 更新可视化
  updateVisualization();
  
  return goal;
}

void VoronoiExplorer::updateDetections(const std::vector<DetectedObject>& detections)
{
  // 暂时不需要处理检测结果
  (void)detections;
}

// ========== Voronoi图构建 ==========

void VoronoiExplorer::voronoiMapCallback(const nav_msgs::OccupancyGridConstPtr& msg)
{
  voronoi_map_received_ = true;
  buildVoronoiGraph(*msg);
}

void VoronoiExplorer::scoreMapCallback(const nav_msgs::OccupancyGridConstPtr& msg)
{
  score_map_ = *msg;
  score_map_received_ = true;
}

void VoronoiExplorer::sceneIdGridCallback(const nav_msgs::OccupancyGridConstPtr& msg)
{
  scene_id_grid_ = *msg;
  scene_id_grid_received_ = true;

  // 每次收到scene_id_grid时，重新读取yaml并生成scene_score_grid
  loadSceneIdScores();
  generateSceneScoreGrid();
}

void VoronoiExplorer::buildVoronoiGraph(const nav_msgs::OccupancyGrid& voronoi_map)
{
  extractVoronoiNodes(voronoi_map);
  ROS_DEBUG("Built Voronoi graph with %zu nodes", voronoi_nodes_.size());
}

void VoronoiExplorer::extractVoronoiNodes(const nav_msgs::OccupancyGrid& voronoi_map)
{
  voronoi_nodes_.clear();
  
  int width = voronoi_map.info.width;
  int height = voronoi_map.info.height;
  double resolution = voronoi_map.info.resolution;
  double origin_x = voronoi_map.info.origin.position.x;
  double origin_y = voronoi_map.info.origin.position.y;
  
  int node_id = 0;
  std::map<std::pair<int, int>, int> grid_to_node_id;
  
  // 第一遍：提取节点
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (isVoronoiNode(x, y, voronoi_map)) {
        int degree = countNeighbors(x, y, voronoi_map);
        if (degree == 1 || degree >= 3) {
          VoronoiNode node;
          node.node_id = node_id++;
          node.position.x = origin_x + (x + 0.5) * resolution;
          node.position.y = origin_y + (y + 0.5) * resolution;
          node.position.z = 0.0;
          voronoi_nodes_.push_back(node);
          grid_to_node_id[{x, y}] = node.node_id;
        }
      }
    }
  }
  
  // 第二遍：连接相邻节点
  for (auto& node : voronoi_nodes_) {
    int mx, my;
    worldToMap(node.position.x, node.position.y, mx, my, voronoi_map);
    
    int offsets[8][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};
    for (int i = 0; i < 8; i++) {
      int nx = mx + offsets[i][0];
      int ny = my + offsets[i][1];
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        if (isVoronoiNode(nx, ny, voronoi_map)) {
          auto it = grid_to_node_id.find({nx, ny});
          if (it != grid_to_node_id.end() && it->second != node.node_id) {
            node.neighbors.push_back(it->second);
          }
        }
      }
    }
  }
  
  ROS_INFO("Extracted %zu Voronoi nodes", voronoi_nodes_.size());
}

bool VoronoiExplorer::isVoronoiNode(int x, int y, const nav_msgs::OccupancyGrid& voronoi_map)
{
  int index = y * voronoi_map.info.width + x;
  if (index < 0 || index >= static_cast<int>(voronoi_map.data.size())) {
    return false;
  }
  return voronoi_map.data[index] == 50;
}

int VoronoiExplorer::countNeighbors(int x, int y, const nav_msgs::OccupancyGrid& voronoi_map)
{
  int count = 0;
  int offsets[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
  for (int i = 0; i < 4; i++) {
    int nx = x + offsets[i][0];
    int ny = y + offsets[i][1];
    if (nx >= 0 && nx < static_cast<int>(voronoi_map.info.width) &&
        ny >= 0 && ny < static_cast<int>(voronoi_map.info.height)) {
      if (isVoronoiNode(nx, ny, voronoi_map)) {
        count++;
      }
    }
  }
  return count;
}

// ========== 节点评估 ==========

double VoronoiExplorer::calculateInformationGain(const VoronoiNode& node,
                                                 const nav_msgs::OccupancyGrid& score_map)
{
  int center_mx, center_my;
  worldToMap(node.position.x, node.position.y, center_mx, center_my, score_map);
  
  if (!isValid(center_mx, center_my, score_map)) {
    return 0.0;
  }
  
  double sensor_range = 5.0;
  double gain = 0.0;
  int radius_cells = static_cast<int>(sensor_range / score_map.info.resolution);
  
  for (int dy = -radius_cells; dy <= radius_cells; dy++) {
    for (int dx = -radius_cells; dx <= radius_cells; dx++) {
      double dist = sqrt(dx * dx + dy * dy) * score_map.info.resolution;
      if (dist > sensor_range) continue;
      
      int mx = center_mx + dx;
      int my = center_my + dy;
      if (isValid(mx, my, score_map)) {
        int idx = my * score_map.info.width + mx;
        int8_t score_value = score_map.data[idx];
        if (score_value > 0) {
          double distance_weight = 1.0 / (1.0 + dist * 0.2);
          double score_weight = static_cast<double>(score_value) / 100.0;
          gain += distance_weight * score_weight;
        }
      }
    }
  }
  
  return gain;
}

// ========== TSP路径规划 ==========

std::vector<geometry_msgs::Point> VoronoiExplorer::selectHighValueNodesForTSP(
    const nav_msgs::OccupancyGrid& score_map)
{
  std::vector<geometry_msgs::Point> selected_positions;

  // 收集节点评分（直接使用已计算好的information_gain）
  std::vector<std::pair<geometry_msgs::Point, double>> node_scores;
  for (const auto& node : voronoi_nodes_) {
    double score = node.information_gain;
    node_scores.push_back({node.position, score});
  }

  if (node_scores.empty()) {
    return selected_positions;
  }

  // 按评分排序（从高到低）
  std::sort(node_scores.begin(), node_scores.end(),
            [](const std::pair<geometry_msgs::Point, double>& a,
               const std::pair<geometry_msgs::Point, double>& b) {
              return a.second > b.second;
            });

  // 按照百分比选择节点（tsp_top_num_ 为 0.0-1.0 之间的比例）
  size_t num_selected = static_cast<size_t>(node_scores.size() * tsp_top_num_);
  if (num_selected == 0) num_selected = 1; // 避免0
  if (num_selected > node_scores.size()) num_selected = node_scores.size();

  selected_positions.reserve(num_selected);
  for (size_t i = 0; i < num_selected; i++) {
    selected_positions.push_back(node_scores[i].first);
  }

  ROS_INFO("Selected %zu high-value positions for TSP (top %.0f%% = %zu of %zu candidates)",
           selected_positions.size(), tsp_top_num_ * 100.0, num_selected, node_scores.size());

  return selected_positions;
}

std::vector<std::vector<int>> VoronoiExplorer::buildTSPDistanceMatrix(
    const std::vector<geometry_msgs::Point>& selected_positions,
                                            const geometry_msgs::Point& current_pos)
{
  size_t num_nodes = selected_positions.size() + 1;  // +1 for robot position
  std::vector<std::vector<int>> distance_matrix(num_nodes, 
                                                std::vector<int>(num_nodes, 0));
  
  // 计算位置之间的距离
  for (size_t i = 0; i < selected_positions.size(); i++) {
    // 获取目标节点的场景分数
    int target_scene_score = getSceneScoreAtPosition(selected_positions[i]);
    
    for (size_t j = 0; j < i; j++) {
      double dist = calculateDistance(selected_positions[i], selected_positions[j]);
  
      // 获取起点和终点的场景分数
      int start_scene_score = getSceneScoreAtPosition(selected_positions[j]);
      
      // 基础代价
      double base_cost = 10.0 * dist;
  
      // 根据场景分数调整代价：分数高的场景降低到达代价（更吸引人）
      // 使用起点和终点的平均分数来调整
      double avg_score = (start_scene_score + target_scene_score) / 2.0;
      // 分数越高，代价降低越多（最多降低tsp_scene_score_weight_比例）
      double score_reduction = tsp_scene_score_weight_ * (avg_score / 100.0);
      double adjusted_cost = base_cost * (1.0 - score_reduction);
      
      int cost = static_cast<int>(adjusted_cost);
      distance_matrix[i][j] = cost;
      distance_matrix[j][i] = cost;
    }
    
    // 机器人到节点的距离
    double dist_to_robot = calculateDistance(selected_positions[i], current_pos);
    
    // 基础代价
    double base_cost_robot = 10.0 * dist_to_robot;
    
    // 根据目标节点的场景分数调整代价
    // 分数高的场景（如卧室）降低到达代价，使其更容易被选择
    double score_reduction_robot = tsp_scene_score_weight_ * (target_scene_score / 100.0);
    double adjusted_cost_robot = base_cost_robot * (1.0 - score_reduction_robot);
    
    int cost_robot = static_cast<int>(adjusted_cost_robot);
    distance_matrix[i][selected_positions.size()] = cost_robot;
    distance_matrix[selected_positions.size()][i] = cost_robot;
  }
  
  return distance_matrix;
}

void VoronoiExplorer::solveTSP(const std::vector<geometry_msgs::Point>& selected_positions,
                               const geometry_msgs::Point& current_pos)
{
  if (selected_positions.empty()) {
    return;
  }
  
  // 构建距离矩阵（机器人位置在最后）
  std::vector<std::vector<int>> distance_matrix = 
      buildTSPDistanceMatrix(selected_positions, current_pos);
  
  // 求解TSP
  tsp_solver_ns::DataModel data_model;
  data_model.distance_matrix = distance_matrix;
  data_model.depot = static_cast<int>(selected_positions.size());  // 机器人位置作为depot
  
  tsp_solver_ns::TSPSolver tsp_solver(data_model);
  tsp_solver.Solve();
  
  std::vector<int> node_index;
  tsp_solver.getSolutionNodeIndex(node_index, false);
  
  // 转换为位置序列（排除机器人位置）
  tsp_path_.clear();
  for (int idx : node_index) {
    if (idx >= 0 && idx < static_cast<int>(selected_positions.size())) {
      tsp_path_.push_back(selected_positions[idx]);
}
  }
  
  tsp_current_index_ = 0;
  current_goal_position_.x = current_goal_position_.y = current_goal_position_.z = 0.0;
  current_goal_set_time_ = ros::Time(0);
  
  ROS_INFO("TSP path solved: %zu positions", tsp_path_.size());
  
  // 更新可视化
  updateVisualization();
}

void VoronoiExplorer::tspReplanTimerCallback(const ros::TimerEvent& event)
{
  // 只有在有必要的输入数据时才重新规划
  if (!voronoi_map_received_ || !score_map_received_ || voronoi_nodes_.empty()) {
    ROS_DEBUG_THROTTLE(5.0, "Cannot replan TSP: missing required data");
    return;
  }
  
  // 更新节点信息增益（确保使用最新的评分地图）
  for (auto& node : voronoi_nodes_) {
    node.information_gain = calculateInformationGain(node, score_map_);
  }
  
  // 获取当前机器人位置（从current_map_或使用默认值）
  geometry_msgs::Point current_pos;
  current_pos.x = 0.0;
  current_pos.y = 0.0;
  current_pos.z = 0.0;
  
  // 重新选择高价值节点并规划TSP路径
  std::vector<geometry_msgs::Point> selected = selectHighValueNodesForTSP(score_map_);
  if (!selected.empty()) {
    ROS_INFO("[TSP] Periodic replanning triggered, solving TSP for %zu nodes", selected.size());
    solveTSP(selected, current_pos);
    } else {
    ROS_WARN_THROTTLE(5.0, "[TSP] Periodic replanning: no high-value nodes selected");
  }
}

ExplorationGoal VoronoiExplorer::selectNextGoalFromTSP(const geometry_msgs::Point& current_pos)
{
  ExplorationGoal goal;
  goal.is_valid = false;
  purgeExpiredBlacklistedGoals();
  
  // 检查是否到达当前目标
  if (current_goal_position_.x != 0.0 || current_goal_position_.y != 0.0) {
    if (isTemporarilyBlacklisted(current_goal_position_)) {
      ROS_WARN("[TSP] Current goal is temporarily blacklisted, skipping");
      tsp_current_index_++;
      current_goal_position_.x = current_goal_position_.y = current_goal_position_.z = 0.0;
      current_goal_set_time_ = ros::Time(0);
      updateVisualization();
    } else {
    // 到达判定使用2D平面距离，避免z轴高度差影响地面导航目标切换
    double dist = calculateDistance2D(current_pos, current_goal_position_);
    ROS_INFO_THROTTLE(1.0,
                      "[TSP] 2D distance=%.3f m, threshold=%.3f m, robot=(%.2f, %.2f), goal=(%.2f, %.2f)",
                      dist, tsp_goal_reach_threshold_,
                      current_pos.x, current_pos.y,
                      current_goal_position_.x, current_goal_position_.y);
    ros::Duration elapsed = ros::Time::now() - current_goal_set_time_;
    bool reached = dist <= tsp_goal_reach_threshold_;
    bool timeout = elapsed.toSec() > tsp_goal_timeout_;
    
    if (reached || timeout) {
      if (timeout) {
        ROS_WARN("[TSP] Goal timeout (%.1f s), skipping", elapsed.toSec());
      } else {
        ROS_INFO("[TSP] Goal reached: distance=%.2f m", dist);
      }
      tsp_current_index_++;
      current_goal_position_.x = current_goal_position_.y = current_goal_position_.z = 0.0;
      current_goal_set_time_ = ros::Time(0);
      updateVisualization();
    } else {
      // 继续前往当前目标
      goal.position = current_goal_position_;
      goal.utility_score = 50.0;
      goal.reason = "TSP path node " + std::to_string(tsp_current_index_) + 
                    "/" + std::to_string(tsp_path_.size());
      goal.is_valid = true;
      return goal;
    }
    }
  }
  
  // 如果TSP路径为空或已完成，重新规划
  if (tsp_path_.empty() || tsp_current_index_ >= static_cast<int>(tsp_path_.size())) {
    // 更新节点信息增益（确保使用最新的评分地图）
    for (auto& node : voronoi_nodes_) {
      node.information_gain = calculateInformationGain(node, score_map_);
    }
    
    std::vector<geometry_msgs::Point> selected = selectHighValueNodesForTSP(score_map_);
    if (!selected.empty()) {
      solveTSP(selected, current_pos);
      } else {
      ROS_WARN("No high-value nodes selected for TSP");
      return goal;
    }
  }
  
  // 选择下一个目标
  while (tsp_current_index_ < static_cast<int>(tsp_path_.size()) &&
         isTemporarilyBlacklisted(tsp_path_[tsp_current_index_])) {
    const auto& skipped = tsp_path_[tsp_current_index_];
    ROS_WARN("[TSP] Skip blacklisted goal %d/%zu: (%.2f, %.2f, %.2f)",
             tsp_current_index_ + 1, tsp_path_.size(),
             skipped.x, skipped.y, skipped.z);
    tsp_current_index_++;
  }

  if (tsp_current_index_ < static_cast<int>(tsp_path_.size())) {
    current_goal_position_ = tsp_path_[tsp_current_index_];
    current_goal_set_time_ = ros::Time::now();
    
    goal.position = current_goal_position_;
    goal.utility_score = 50.0;
    goal.reason = "TSP path node " + std::to_string(tsp_current_index_ + 1) + 
                  "/" + std::to_string(tsp_path_.size());
    goal.is_valid = true;
  
    ROS_INFO("[TSP] Selected goal %d/%zu: (%.2f, %.2f, %.2f)",
             tsp_current_index_ + 1, tsp_path_.size(),
             goal.position.x, goal.position.y, goal.position.z);
    
    // 更新可视化
    updateVisualization();
  }
  
  return goal;
}

void VoronoiExplorer::addTemporaryBlacklistPoint(const geometry_msgs::Point& point, double duration_sec)
{
  purgeExpiredBlacklistedGoals();
  BlacklistedGoal item;
  item.point = point;
  item.expire_time = ros::Time::now() + ros::Duration(duration_sec);
  temporary_blacklisted_goals_.push_back(item);
  ROS_WARN("[TSP] Add temporary blacklisted goal for %.1f s: (%.2f, %.2f, %.2f)",
           duration_sec, point.x, point.y, point.z);
}

void VoronoiExplorer::reset()
{
  voronoi_nodes_.clear();
  tsp_path_.clear();
  temporary_blacklisted_goals_.clear();

  score_map_.data.clear();
  scene_id_grid_.data.clear();
  scene_score_grid_.data.clear();
  current_map_.data.clear();

  voronoi_map_received_ = false;
  score_map_received_ = false;
  scene_id_grid_received_ = false;
  tsp_current_index_ = 0;
  current_goal_set_time_ = ros::Time(0);
  current_goal_position_ = geometry_msgs::Point();

  if (visualization_) {
    const std::string frame_id = current_map_.header.frame_id;
    visualization_->clearAllMarkers(frame_id);
  }

  ROS_WARN("[VoronoiExplorer] Internal graph/TSP state reset");
}

void VoronoiExplorer::updateVisualization()
{
  if (!visualization_) {
    return;
  }
  
  // 发布评分节点可视化
  if (!voronoi_nodes_.empty() && score_map_received_) {
    geometry_msgs::Pose current_pose;
    current_pose.position.x = 0.0;  // 这里可以传入实际位置，暂时用0
    current_pose.position.y = 0.0;
    current_pose.position.z = 0.0;
    current_pose.orientation.w = 1.0;
    
    visualization_->publishScoredMarkers(voronoi_nodes_, current_pose, score_map_);
  }
  
  // 发布TSP路径可视化
  if (!tsp_path_.empty()) {
    std::string frame_id = current_map_.header.frame_id.empty() ? 
                          "map" : current_map_.header.frame_id;
    visualization_->publishTSPPath(tsp_path_, tsp_current_index_, 
                                  current_goal_position_, frame_id);
    }
  }
  





// ========== 接近目标路径规划 ==========

ExplorationGoal VoronoiExplorer::planApproachToTarget(
    const geometry_msgs::Point& target_pos,
    const geometry_msgs::Point& current_pos,
    const nav_msgs::OccupancyGrid& map)
{
  ExplorationGoal goal;
  goal.is_valid = false;
  
  // 检查维诺图是否可用
  if (!voronoi_map_received_ || voronoi_nodes_.empty()) {
    ROS_WARN_THROTTLE(2.0, "Voronoi graph not available for approach planning, using target directly");
    // 如果维诺图不可用，只能使用目标位置（但这种情况应该避免）
    goal.position = target_pos;
    goal.is_valid = true;
    goal.utility_score = 100.0;
    goal.reason = "Direct target (no Voronoi graph available)";
    return goal;
  }
  
  // 找到离目标最近的维诺节点（始终使用维诺节点，因为目标可能很大，如车、床等）
  double min_dist_to_target = std::numeric_limits<double>::max();
  const VoronoiNode* nearest_node = nullptr;
  
  for (const auto& node : voronoi_nodes_) {
    double dist = calculateDistance(node.position, target_pos);
    if (dist < min_dist_to_target) {
      min_dist_to_target = dist;
      nearest_node = &node;
    }
  }
  
  if (nearest_node != nullptr) {
    double dist_to_target = calculateDistance(nearest_node->position, target_pos);
    ROS_INFO("[Approach] Selected nearest Voronoi node to target: (%.2f, %.2f, %.2f), distance to target: %.2f m",
             nearest_node->position.x, nearest_node->position.y, nearest_node->position.z,
             dist_to_target);
    goal.position = nearest_node->position;
    goal.is_valid = true;
    goal.utility_score = 90.0;
    goal.reason = "Nearest Voronoi node to target (dist: " + std::to_string(dist_to_target) + " m)";
  } else {
    // 如果找不到维诺节点（异常情况），回退到直接使用目标位置
    ROS_WARN("[Approach] No Voronoi nodes found, using target directly (fallback)");
    goal.position = target_pos;
    goal.is_valid = true;
    goal.utility_score = 80.0;
    goal.reason = "Direct target (no Voronoi nodes available, fallback)";
  }
  
  return goal;
}

// ========== 辅助函数 ==========

void VoronoiExplorer::worldToMap(double wx, double wy, int& mx, int& my,
                                 const nav_msgs::OccupancyGrid& map)
{
  mx = static_cast<int>((wx - map.info.origin.position.x) / map.info.resolution);
  my = static_cast<int>((wy - map.info.origin.position.y) / map.info.resolution);
  }
  
bool VoronoiExplorer::isValid(int mx, int my, const nav_msgs::OccupancyGrid& map)
{
  return mx >= 0 && mx < static_cast<int>(map.info.width) &&
         my >= 0 && my < static_cast<int>(map.info.height);
}

double VoronoiExplorer::calculateDistance(const geometry_msgs::Point& p1,
                                         const geometry_msgs::Point& p2)
{
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  double dz = p1.z - p2.z;
  return sqrt(dx * dx + dy * dy + dz * dz);
}

double VoronoiExplorer::calculateDistance2D(const geometry_msgs::Point& p1,
                                            const geometry_msgs::Point& p2)
{
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  return sqrt(dx * dx + dy * dy);
}

void VoronoiExplorer::purgeExpiredBlacklistedGoals()
{
  ros::Time now = ros::Time::now();
  temporary_blacklisted_goals_.erase(
      std::remove_if(temporary_blacklisted_goals_.begin(),
                     temporary_blacklisted_goals_.end(),
                     [&now](const BlacklistedGoal& item) {
                       return item.expire_time <= now;
                     }),
      temporary_blacklisted_goals_.end());
}

bool VoronoiExplorer::isTemporarilyBlacklisted(const geometry_msgs::Point& point)
{
  for (const auto& item : temporary_blacklisted_goals_) {
    if (calculateDistance2D(point, item.point) <= blacklist_match_threshold_) {
      return true;
    }
  }
  return false;
}

int VoronoiExplorer::getSceneScoreAtPosition(const geometry_msgs::Point& pos)
{
  // 如果没有scene_score_grid，返回默认分数
  if (!scene_id_grid_received_ || scene_score_grid_.data.empty()) {
    return default_scene_score_;
  }
  
  // 将世界坐标转换为地图坐标
  int mx, my;
  worldToMap(pos.x, pos.y, mx, my, scene_score_grid_);
  
  // 检查坐标是否有效
  if (!isValid(mx, my, scene_score_grid_)) {
    return default_scene_score_;
  }
  
  // 获取该位置的场景分数
  int idx = my * scene_score_grid_.info.width + mx;
  int score = static_cast<int>(scene_score_grid_.data[idx]);
  
  return score;
}

// ========== Scene Score Grid ==========

bool VoronoiExplorer::loadSceneIdScores()
{
  scene_id_scores_map_.clear();
  
  // 展开ROS路径变量（如$(find explore_pkg)）
  std::string expanded_path = scene_id_scores_yaml_path_;
  size_t pos = expanded_path.find("$(find explore_pkg)");
  if (pos != std::string::npos) {
    std::string package_path = ros::package::getPath("explore_pkg");
    if (!package_path.empty()) {
      expanded_path.replace(pos, strlen("$(find explore_pkg)"), package_path);
      } else {
      ROS_ERROR("Cannot find explore_pkg package path");
      return false;
    }
  }

  // 读取JSON文件
  std::ifstream file(expanded_path);
  if (!file.is_open()) {
    ROS_WARN_THROTTLE(5.0, "Failed to open JSON file: %s", expanded_path.c_str());
    return false;
      }

  // 读取文件内容
  std::string json_content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
  file.close();
  
  // 解析JSON
  rapidjson::Document doc;
  if (doc.Parse(json_content.c_str()).HasParseError()) {
    ROS_WARN_THROTTLE(5.0, "Failed to parse JSON file: %s", expanded_path.c_str());
    return false;
  }
  
  // 检查是否有scene_scores字段
  if (!doc.HasMember("scene_scores") || !doc["scene_scores"].IsObject()) {
    ROS_WARN_THROTTLE(5.0, "JSON file missing 'scene_scores' object: %s", expanded_path.c_str());
    return false;
  }
  
  const rapidjson::Value& scene_scores = doc["scene_scores"];
  
  // 读取默认分数
  if (scene_scores.HasMember("default_score") && scene_scores["default_score"].IsInt()) {
    default_scene_score_ = scene_scores["default_score"].GetInt();
  } else {
    default_scene_score_ = 0;
  }
  
  // 读取scene_id到score的映射
  for (rapidjson::Value::ConstMemberIterator it = scene_scores.MemberBegin();
       it != scene_scores.MemberEnd(); ++it) {
    std::string key = it->name.GetString();
      
    // 跳过default_score，已经单独读取
    if (key == "default_score") {
      continue;
      }
    
    // 尝试将key转换为int（scene_id）
    int scene_id = 0;
    try {
      scene_id = std::stoi(key);
    } catch (...) {
      ROS_WARN_THROTTLE(5.0, "Invalid scene_id key in JSON: %s", key.c_str());
      continue;
  }
  
    // 读取分数值
    if (it->value.IsInt()) {
      int score = it->value.GetInt();
      scene_id_scores_map_[scene_id] = score;
      ROS_DEBUG("Loaded scene_id %d -> score %d", scene_id, score);
    } else if (it->value.IsNumber()) {
      int score = static_cast<int>(it->value.GetDouble());
      scene_id_scores_map_[scene_id] = score;
      ROS_DEBUG("Loaded scene_id %d -> score %d", scene_id, score);
    }
  }
  
  ROS_INFO_THROTTLE(10.0, "Loaded %zu scene ID scores from JSON, default score: %d", 
                    scene_id_scores_map_.size(), default_scene_score_);
  return true;
}

void VoronoiExplorer::generateSceneScoreGrid()
{
  if (!scene_id_grid_received_) {
    ROS_WARN_THROTTLE(2.0, "Scene ID grid not received, cannot generate scene score grid");
    return;
  }
  
  // 初始化scene_score_grid，使用与scene_id_grid相同的参数
  scene_score_grid_.header = scene_id_grid_.header;
  scene_score_grid_.info = scene_id_grid_.info;
  scene_score_grid_.data.resize(scene_id_grid_.data.size());
  
  int width = scene_id_grid_.info.width;
  int height = scene_id_grid_.info.height;
  
  // 遍历所有栅格，根据scene_id查找对应的分数
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      int8_t scene_id = scene_id_grid_.data[idx];
      
      // 查找scene_id对应的分数
      int score = default_scene_score_;
      if (scene_id >= 0) {
        auto it = scene_id_scores_map_.find(static_cast<int>(scene_id));
        if (it != scene_id_scores_map_.end()) {
          score = it->second;
      }
    }
      
      // 限制分数范围在0-100
      score = std::max(0, std::min(100, score));
      scene_score_grid_.data[idx] = static_cast<int8_t>(score);
    }
  }
  
  // 更新时间戳
  scene_score_grid_.header.stamp = ros::Time::now();
  
  // 发布scene_score_grid
  scene_score_grid_pub_.publish(scene_score_grid_);
  
  ROS_DEBUG_THROTTLE(5.0, "Generated and published scene_score_grid (%dx%d)", width, height);
}

}  // namespace explore_pkg
