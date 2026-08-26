#include "explore_pkg/explore_manager.h"
#include "explore_pkg/voronoi_explorer.h"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <tf/tf.h>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Twist.h>

namespace explore_pkg
{

namespace
{

std::string normalizeSemanticToken(const std::string& text)
{
  std::string normalized = text;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  std::replace(normalized.begin(), normalized.end(), ' ', '_');
  return normalized;
}

}  // namespace

ExploreManager::ExploreManager()
  : nh_()
  , private_nh_("~")
  , current_state_(NavigationState::IDLE)
  , has_target_position_(false)
  , odom_received_(false)
  , map_received_(false)
  , scene_id_grid_received_(false)
  , path_recording_enabled_(false)
  , path_sampling_distance_(0.5)
  , has_active_goal_(false)
  , rotating_in_place_(false)
  , last_goal_publish_time_(0)
  , last_move_base_failure_handle_time_(0)
  , goal_republish_interval_(2.0)
  , move_base_failure_cooldown_(1.0)
  , failed_goal_blacklist_duration_(10.0)
  , occupancy_boundary_weight_(1.0)
  , scene_known_unknown_boundary_weight_(1.0)
  , scene_different_boundary_weight_(1.5)
  , score_map_publish_rate_(2)
  , recovery_rotate_speed_(0.4)
  , recovery_rotate_rate_(10.0)
{
}

ExploreManager::~ExploreManager()
{
  stop();
}

bool ExploreManager::initialize()
{
  ROS_INFO("Initializing ExploreManager...");

  // 加载参数
  loadParameters();

  // 设置订阅者和发布者
  semantic_map_sub_ = nh_.subscribe(semantic_map_topic_, 10,
                                    &ExploreManager::semanticMapCallback, this);
  odom_sub_ = nh_.subscribe(odom_topic_, 1, &ExploreManager::odomCallback, this);
  map_sub_ = nh_.subscribe(map_topic_, 1, &ExploreManager::mapCallback, this);
  scene_id_grid_sub_ = nh_.subscribe(scene_id_grid_topic_, 1,
                                     &ExploreManager::sceneIdGridCallback, this);
  move_base_status_sub_ = nh_.subscribe(move_base_status_topic_, 10,
                                        &ExploreManager::moveBaseStatusCallback, this);
  reset_sub_ = nh_.subscribe(reset_topic_, 1, &ExploreManager::resetCallback, this);
                                     
  goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(goal_topic_, 10);
  goal_point_pub_ = nh_.advertise<geometry_msgs::PointStamped>("/explore_manager/goal_point", 10);
  cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>(cmd_vel_topic_, 10);
  state_pub_ = nh_.advertise<std_msgs::String>("/explore_manager/state", 10);
  status_pub_ = nh_.advertise<std_msgs::String>("/explore_manager/status", 10);
  exploration_path_pub_ = nh_.advertise<nav_msgs::Path>(exploration_path_topic_, 10);
  score_map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>(score_map_topic_, 1, true);

  // 初始化planner（目前只创建接口，具体实现后续添加）
  setupPlanner();

  // 初始化探索路径
  exploration_path_.header.frame_id = map_frame_;
  exploration_path_.header.stamp = ros::Time::now();

  ROS_INFO("ExploreManager initialized successfully");
  return true;
}

void ExploreManager::loadParameters()
{
  // 话题名称
  private_nh_.param("topics/semantic_map_topic", semantic_map_topic_,
                     std::string("/sem_mapping/obj_map"));
  private_nh_.param("topics/navigation/goal_topic", goal_topic_,
                     std::string("/move_base_simple/goal"));
  
  // 目标描述（从参数读取）
  private_nh_.param("target_description", target_description_, std::string(""));
  target_description_ = normalizeSemanticToken(target_description_);
  private_nh_.param("topics/navigation/odom_topic", odom_topic_,
                    std::string("/odom"));
  private_nh_.param("topics/navigation/map_topic", map_topic_,
                     std::string("/struct_mapping/occ_map"));
  private_nh_.param("topics/navigation/cmd_vel_topic", cmd_vel_topic_,
                    std::string("/cmd_vel"));
  private_nh_.param("topics/navigation/move_base_status_topic", move_base_status_topic_,
                    std::string("/move_base/status"));
  private_nh_.param("topics/exploration_path_topic", exploration_path_topic_,
                     std::string("/explore_manager/exploration_path"));

  // 坐标系
  private_nh_.param("frames/base_frame", base_frame_, std::string("base_link"));
  private_nh_.param("frames/map_frame", map_frame_, std::string("map"));

  // 阈值
  private_nh_.param("thresholds/target_reach", target_reach_threshold_, 0.5);
  private_nh_.param("thresholds/semantic_confidence", semantic_confidence_threshold_, 0.7);

  // 导航参数
  private_nh_.param("navigation/exploration_rate", exploration_rate_, 1.0);
  private_nh_.param("navigation/goal_check_rate", goal_check_rate_, 5.0);
  private_nh_.param("navigation/goal_republish_interval", goal_republish_interval_, 2.0);
  private_nh_.param("navigation/move_base_failure_cooldown", move_base_failure_cooldown_, 1.0);
  private_nh_.param("navigation/failed_goal_blacklist_duration", failed_goal_blacklist_duration_, 10.0);
  private_nh_.param("navigation/recovery_rotate_speed", recovery_rotate_speed_, 0.4);
  private_nh_.param("navigation/recovery_rotate_rate", recovery_rotate_rate_, 10.0);

  // 探索参数
  private_nh_.param("exploration/algorithm_type", exploration_algorithm_type_, std::string("frontier"));

  // 路径记录参数
  private_nh_.param("path_recording/enable", path_recording_enabled_param_, true);
  private_nh_.param("path_recording/sampling_distance", path_sampling_distance_param_, 0.5);

  // 评分地图参数
  private_nh_.param("topics/scene_id_grid_topic", scene_id_grid_topic_,
                     std::string("/semantic_mapping/scene_id_grid"));
  private_nh_.param("topics/score_map_topic", score_map_topic_,
                     std::string("/explore_manager/score_map"));
  private_nh_.param("topics/reset_topic", reset_topic_,
                     std::string("/nav_system/reset"));
  private_nh_.param("score_map/occupancy_boundary_weight", occupancy_boundary_weight_, 1.0);
  private_nh_.param("score_map/scene_known_unknown_boundary_weight", scene_known_unknown_boundary_weight_, 1.0);
  private_nh_.param("score_map/scene_different_boundary_weight", scene_different_boundary_weight_, 1.5);
  private_nh_.param("score_map/publish_rate", score_map_publish_rate_, 2);

  path_recording_enabled_ = path_recording_enabled_param_;
  path_sampling_distance_ = path_sampling_distance_param_;

  ROS_INFO("Parameters loaded:");
  ROS_INFO("  Target description: %s", target_description_.empty() ? "(empty, general exploration)" : target_description_.c_str());
  ROS_INFO("  Target reach threshold: %.2f m", target_reach_threshold_);
  ROS_INFO("  Semantic confidence threshold: %.2f", semantic_confidence_threshold_);
  ROS_INFO("  Exploration algorithm: %s", exploration_algorithm_type_.c_str());
  ROS_INFO("  Path recording: %s", path_recording_enabled_ ? "enabled" : "disabled");
  ROS_INFO("  Goal republish interval: %.2f s", goal_republish_interval_);
  ROS_INFO("  Move base failure cooldown: %.2f s", move_base_failure_cooldown_);
  ROS_INFO("  Failed goal blacklist duration: %.2f s", failed_goal_blacklist_duration_);
  ROS_INFO("  Recovery rotate speed: %.2f rad/s", recovery_rotate_speed_);
  ROS_INFO("  Recovery rotate rate: %.2f Hz", recovery_rotate_rate_);
}

void ExploreManager::setupPlanner()
{
  ROS_INFO("Initializing VoronoiExplorer");
  if (!explorer_.initialize(nh_, private_nh_)) {
    ROS_ERROR("Failed to initialize VoronoiExplorer");
  } else {
    ROS_INFO("VoronoiExplorer initialized successfully");
  }
}

void ExploreManager::start()
{
  ROS_INFO("Starting ExploreManager...");
  
  // 创建定时器
  exploration_timer_ = nh_.createTimer(
      ros::Duration(1.0 / exploration_rate_),
      &ExploreManager::executeExploration, this);
  
  goal_check_timer_ = nh_.createTimer(
      ros::Duration(1.0 / goal_check_rate_),
      &ExploreManager::checkGoalReached, this);
  
  state_machine_timer_ = nh_.createTimer(
      ros::Duration(0.1),  // 10Hz状态机循环
      &ExploreManager::stateMachineLoop, this);
  
  // 评分地图更新定时器
  score_map_timer_ = nh_.createTimer(
      ros::Duration(1.0 / score_map_publish_rate_),
      &ExploreManager::generateScoreMap, this);

  // 原地旋转恢复定时器（默认不自动启动，通过startRecoveryRotation触发）
  recovery_rotate_timer_ = nh_.createTimer(
      ros::Duration(1.0 / recovery_rotate_rate_),
      &ExploreManager::recoveryRotateCallback, this,
      false, false);

  // 如果没有目标描述，自动进入探索状态
  updateState(NavigationState::EXPLORING);

  ROS_INFO("ExploreManager started");
}

void ExploreManager::stop()
{
  exploration_timer_.stop();
  goal_check_timer_.stop();
  state_machine_timer_.stop();
  score_map_timer_.stop();
  recovery_rotate_timer_.stop();
  stopRecoveryRotation();
  
  ROS_INFO("ExploreManager stopped");
}

// ========== 回调函数实现 ==========

void ExploreManager::semanticMapCallback(const std_msgs::StringConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  processSemanticMap(msg->data);
}

void ExploreManager::odomCallback(const nav_msgs::OdometryConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  current_odom_ = *msg;
  current_pose_.header = msg->header;
  current_pose_.pose = msg->pose.pose;
  
  // 使用/odometry中的平面坐标（x, y）进行探索与到达判定。
  // 注：ai2thor_ros.py已将AI2-THOR的地面坐标映射到odom.x/odom.y。
  last_robot_position_.x = current_odom_.pose.pose.position.x;
  last_robot_position_.y = current_odom_.pose.pose.position.y;
  last_robot_position_.z = 0.0;
  current_pose_.pose.position.x = last_robot_position_.x;
  current_pose_.pose.position.y = last_robot_position_.y;
  current_pose_.pose.position.z = 0.0;
  ROS_INFO_THROTTLE(1.0,
                    "[ExploreManager] Odom raw(x,y,z)=(%.2f, %.2f, %.2f), planar used(x,y,z)=(%.2f, %.2f, %.2f)",
                    current_odom_.pose.pose.position.x,
                    current_odom_.pose.pose.position.y,
                    current_odom_.pose.pose.position.z,
                    current_pose_.pose.position.x,
                    current_pose_.pose.position.y,
                    current_pose_.pose.position.z);
  
  odom_received_ = true;
  
  // 更新探索路径
  if (path_recording_enabled_) {
    updateExplorationPath();
  }
}

void ExploreManager::mapCallback(const nav_msgs::OccupancyGridConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  current_map_ = *msg;
  map_received_ = true;
}

void ExploreManager::sceneIdGridCallback(const nav_msgs::OccupancyGridConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  scene_id_grid_ = *msg;
  scene_id_grid_received_ = true;
}

void ExploreManager::moveBaseStatusCallback(const actionlib_msgs::GoalStatusArrayConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (msg->status_list.empty()) {
    return;
  }

  const actionlib_msgs::GoalStatus& latest_status = msg->status_list.back();
  const uint8_t status = latest_status.status;
  if (status != actionlib_msgs::GoalStatus::ABORTED &&
      status != actionlib_msgs::GoalStatus::REJECTED) {
    return;
  }

  if (current_state_ != NavigationState::EXPLORING) {
    return;
  }

  ros::Time now = ros::Time::now();
  if ((now - last_move_base_failure_handle_time_).toSec() < move_base_failure_cooldown_) {
    return;
  }

  const std::string& goal_id = latest_status.goal_id.id;
  if (!goal_id.empty() && goal_id == last_failed_goal_id_) {
    return;
  }

  last_move_base_failure_handle_time_ = now;
  last_failed_goal_id_ = goal_id;
  if (has_active_goal_) {
    explorer_.addTemporaryBlacklistPoint(current_goal_.pose.position, failed_goal_blacklist_duration_);
  }
  has_active_goal_ = false;
  last_goal_publish_time_ = ros::Time(0);
  startRecoveryRotation();

  ROS_WARN("[ExploreManager] move_base reported planning failure (status=%u), "
           "forcing goal reselection in next exploration cycle", status);
}

void ExploreManager::resetCallback(const std_msgs::EmptyConstPtr& msg)
{
  (void)msg;
  std::lock_guard<std::mutex> lock(state_mutex_);

  stopRecoveryRotation();

  semantic_objects_.clear();
  visited_points_.clear();
  exploration_path_.poses.clear();
  exploration_path_.header.stamp = ros::Time::now();
  score_map_.data.clear();
  current_map_.data.clear();
  scene_id_grid_.data.clear();

  has_target_position_ = false;
  has_active_goal_ = false;
  rotating_in_place_ = false;
  map_received_ = false;
  scene_id_grid_received_ = false;
  last_goal_publish_time_ = ros::Time(0);
  last_move_base_failure_handle_time_ = ros::Time(0);
  last_failed_goal_id_.clear();

  geometry_msgs::Twist stop_cmd;
  cmd_vel_pub_.publish(stop_cmd);

  explorer_.reset();

  updateState(NavigationState::EXPLORING);
  publishStatus();

  ROS_WARN("[ExploreManager] Received reset signal on %s, exploration state cleared", reset_topic_.c_str());
}

// ========== 核心功能函数实现 ==========

void ExploreManager::stateMachineLoop(const ros::TimerEvent& event)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  
  switch (current_state_) {
    case NavigationState::IDLE:
      // 等待目标描述
      break;
      
    case NavigationState::EXPLORING:
      // 探索循环由exploration_timer_触发
      break;
      
    case NavigationState::APPROACHING:
      // 检查是否到达目标由goal_check_timer_触发
      break;
      
    case NavigationState::REACHED:
      // 到达目标，保持状态
      break;
      
    case NavigationState::FAILED:
      // 失败状态，可以在这里实现重试逻辑
      break;
  }
  
  publishStatus();
}

void ExploreManager::processSemanticMap(const std::string& json_data)
{
  try {
    rapidjson::Document doc;
    doc.Parse(json_data.c_str());
    
    if (doc.HasParseError() || !doc.IsArray()) {
      ROS_WARN("Failed to parse semantic map JSON");
      return;
    }
    
    semantic_objects_.clear();
    
    for (rapidjson::SizeType i = 0; i < doc.Size(); i++) {
      const rapidjson::Value& obj = doc[i];
      
      if (!obj.HasMember("semantic_name") || !obj.HasMember("conf")) {
        continue;
      }
      
      DetectedObject semantic_obj;
      semantic_obj.semantic_name = obj["semantic_name"].GetString();
      semantic_obj.confidence = obj["conf"].GetDouble();
      
      // 解析3D坐标
      if (obj.HasMember("coord") && obj["coord"].IsArray()) {
        const rapidjson::Value& coord = obj["coord"];
        if (coord.Size() >= 3) {
          semantic_obj.coord_3d.x = coord[0].GetDouble();
          semantic_obj.coord_3d.y = coord[1].GetDouble();
          semantic_obj.coord_3d.z = coord[2].GetDouble();
        }
      }
      
      // 解析环境状态
      if (obj.HasMember("env_status")) {
        semantic_obj.env_status = obj["env_status"].GetString();
      }

      // 检查是否匹配目标
      if (!target_description_.empty() && 
          normalizeSemanticToken(semantic_obj.semantic_name) == target_description_ && 
          semantic_obj.confidence >= semantic_confidence_threshold_) {
        target_position_ = semantic_obj.coord_3d;
        has_target_position_ = true;
        ROS_INFO("Target detected: %s at (%.2f, %.2f, %.2f), confidence: %.2f",
                 target_description_.c_str(),
                 target_position_.x, target_position_.y, target_position_.z,
                 semantic_obj.confidence);
        
        updateState(NavigationState::APPROACHING);
        planToTarget();
      }
      
      semantic_objects_.push_back(semantic_obj);
    }
    
    // 更新explorer的检测结果
    explorer_.updateDetections(semantic_objects_);
    
    last_semantic_map_time_ = ros::Time::now();
    
  } catch (const std::exception& e) {
    ROS_ERROR("Error processing semantic map: %s", e.what());
  }
}

void ExploreManager::checkGoalReached(const ros::TimerEvent& event)
{
  if (!odom_received_ || !has_target_position_) {
    return;
  }
  
  if (current_state_ != NavigationState::APPROACHING) {
    return;
  }
  
  double distance = calculateDistance(last_robot_position_, target_position_);
  
  if (distance <= target_reach_threshold_) {
    ROS_INFO("Target reached! Distance: %.2f m", distance);
    updateState(NavigationState::REACHED);
  }
}

bool ExploreManager::planToTarget()
{
  if (!has_target_position_ || !odom_received_) {
    ROS_WARN("Cannot plan to target: missing target position or odometry");
    return false;
  }
  
  if (!map_received_) {
    ROS_WARN("Cannot plan to target: map not received");
    return false;
  }
  
  // 使用维诺图规划接近目标的路径
  ExplorationGoal approach_goal = explorer_.planApproachToTarget(
      target_position_,
      last_robot_position_,
      current_map_);
  
  if (!approach_goal.is_valid) {
    ROS_WARN("Failed to plan approach path to target");
    return false;
  }
  
  geometry_msgs::PoseStamped goal = createGoalPose(approach_goal.position);
  stopRecoveryRotation();
  
  ROS_INFO("Publishing approach goal to: (%.2f, %.2f, %.2f), reason: %s",
           goal.pose.position.x, goal.pose.position.y, goal.pose.position.z,
           approach_goal.reason.c_str());
  
  goal_pub_.publish(goal);
  geometry_msgs::PointStamped goal_point_msg;
  goal_point_msg.header = goal.header;
  goal_point_msg.point = goal.pose.position;
  goal_point_pub_.publish(goal_point_msg);
  last_goal_publish_time_ = ros::Time::now();
  current_goal_ = goal;
  has_active_goal_ = true;
  
  return true;
}

void ExploreManager::executeExploration(const ros::TimerEvent& event)
{
  // 记录当前导航状态
  ROS_INFO("[ExploreManager] Exploration loop - Navigation State: %s", 
           stateToString(current_state_).c_str());
  
  // 只在EXPLORING状态下执行探索，其他状态（如APPROACHING）不应该执行探索
  if (current_state_ != NavigationState::EXPLORING) {
    stopRecoveryRotation();
    ROS_DEBUG_THROTTLE(5.0, "[ExploreManager] 不在EXPLORING状态，当前状态: %s, 跳过探索", 
                      stateToString(current_state_).c_str());
    return;
  }
  
  if (!map_received_ || !odom_received_) {
    stopRecoveryRotation();
    ROS_WARN_THROTTLE(5.0, "Waiting for map or odometry... (map: %d, odom: %d)", 
                      map_received_, odom_received_);
    return;
  }
  
  // 调用explorer选择下一个探索目标
  ExplorationGoal goal = explorer_.selectNextGoal(
      current_map_,
      current_pose_.pose
  );
  
  if (goal.is_valid) {
    // 检查目标是否与当前目标相同（避免频繁重新发布相同目标）
    bool is_same_goal = false;
    if (has_active_goal_) {
      double dist_to_current = calculateDistance(
          goal.position,
          geometry_msgs::Point(current_goal_.pose.position)
      );
      const double goal_same_threshold = 0.1;  // 10cm内认为是同一个目标
      if (dist_to_current < goal_same_threshold) {
        is_same_goal = true;
        ros::Duration since_last_publish = ros::Time::now() - last_goal_publish_time_;
        if (since_last_publish.toSec() >= goal_republish_interval_) {
          goal_pub_.publish(current_goal_);
          geometry_msgs::PointStamped goal_point_msg;
          goal_point_msg.header = current_goal_.header;
          goal_point_msg.point = current_goal_.pose.position;
          goal_point_pub_.publish(goal_point_msg);
          last_goal_publish_time_ = ros::Time::now();
          ROS_WARN("[ExploreManager] TSP目标未改变，已间隔%.2f秒，重发当前目标: (%.2f, %.2f, %.2f)",
                   since_last_publish.toSec(),
                   current_goal_.pose.position.x,
                   current_goal_.pose.position.y,
                   current_goal_.pose.position.z);
        } else {
          ROS_INFO("[ExploreManager] TSP目标未改变 (%.2f, %.2f, %.2f), 距离当前目标 %.2f m, 跳过重新发布",
                            goal.position.x, goal.position.y, goal.position.z, dist_to_current);
        }
      }
    }
    
    if (!is_same_goal) {
      stopRecoveryRotation();
      ROS_INFO("Selected exploration goal: (%.2f, %.2f, %.2f), score: %.2f, reason: %s",
               goal.position.x, goal.position.y, goal.position.z,
               goal.utility_score, goal.reason.c_str());
      
      // 发布探索目标
      geometry_msgs::PoseStamped exploration_goal = createGoalPose(goal.position);
      goal_pub_.publish(exploration_goal);
      geometry_msgs::PointStamped goal_point_msg;
      goal_point_msg.header = exploration_goal.header;
      goal_point_msg.point = exploration_goal.pose.position;
      goal_point_pub_.publish(goal_point_msg);
      last_goal_publish_time_ = ros::Time::now();
      current_goal_ = exploration_goal;
      has_active_goal_ = true;
      
      ROS_INFO("Published navigation goal to %s: (%.2f, %.2f, %.2f)", 
               goal_topic_.c_str(), 
               exploration_goal.pose.position.x,
               exploration_goal.pose.position.y,
               exploration_goal.pose.position.z);
      
      // 注意：visited_points_由planner内部管理（TSP路径会跟踪已访问节点）
      // 这里不再自动添加，避免干扰TSP路径的节点访问跟踪
    }
  } else {
    startRecoveryRotation();
    ROS_WARN_THROTTLE(2.0, "No valid exploration goal found (state: %s, map: %d, odom: %d)", 
                      stateToString(current_state_).c_str(), map_received_, odom_received_);
  }
}

void ExploreManager::updateExplorationPath()
{
  if (!odom_received_) {
    return;
  }
  
  // 检查是否需要记录新点（采样间隔）
  if (exploration_path_.poses.empty()) {
    // 第一个点，直接添加
    geometry_msgs::PoseStamped pose;
    pose.header = current_pose_.header;
    pose.pose = current_pose_.pose;
    exploration_path_.poses.push_back(pose);
    last_recorded_position_ = last_robot_position_;
    exploration_path_.header.stamp = ros::Time::now();
    exploration_path_pub_.publish(exploration_path_);
    return;
  }
  
  // 计算距离上次记录点的距离
  double distance = calculateDistance(last_robot_position_, last_recorded_position_);
  
  if (distance >= path_sampling_distance_) {
    // 添加新点
    geometry_msgs::PoseStamped pose;
    pose.header = current_pose_.header;
    pose.pose = current_pose_.pose;
    exploration_path_.poses.push_back(pose);
    last_recorded_position_ = last_robot_position_;
    exploration_path_.header.stamp = ros::Time::now();
    exploration_path_pub_.publish(exploration_path_);
  }
}

void ExploreManager::recoveryRotateCallback(const ros::TimerEvent& event)
{
  if (!rotating_in_place_ || current_state_ != NavigationState::EXPLORING) {
    return;
  }

  geometry_msgs::Twist cmd_vel;
  cmd_vel.linear.x = 0.0;
  cmd_vel.linear.y = 0.0;
  cmd_vel.linear.z = 0.0;
  cmd_vel.angular.x = 0.0;
  cmd_vel.angular.y = 0.0;
  cmd_vel.angular.z = recovery_rotate_speed_;
  cmd_vel_pub_.publish(cmd_vel);
}

void ExploreManager::startRecoveryRotation()
{
  if (rotating_in_place_) {
    return;
  }

  rotating_in_place_ = true;
  recovery_rotate_timer_.start();
  ROS_WARN_THROTTLE(2.0, "[ExploreManager] No valid exploration goal, rotating in place...");
}

void ExploreManager::stopRecoveryRotation()
{
  if (!rotating_in_place_) {
    return;
  }

  rotating_in_place_ = false;
  recovery_rotate_timer_.stop();

  // 停止原地旋转，避免残留角速度命令
  geometry_msgs::Twist stop_cmd;
  stop_cmd.linear.x = 0.0;
  stop_cmd.linear.y = 0.0;
  stop_cmd.linear.z = 0.0;
  stop_cmd.angular.x = 0.0;
  stop_cmd.angular.y = 0.0;
  stop_cmd.angular.z = 0.0;
  cmd_vel_pub_.publish(stop_cmd);
}

// ========== 辅助函数实现 ==========

void ExploreManager::updateState(NavigationState new_state)
{
  if (current_state_ != new_state) {
    NavigationState old_state = current_state_;
    current_state_ = new_state;
    ROS_INFO("State changed: %s -> %s", 
             stateToString(old_state).c_str(),
             stateToString(new_state).c_str());
  }
}

void ExploreManager::publishStatus()
{
  std_msgs::String state_msg;
  state_msg.data = stateToString(current_state_);
  state_pub_.publish(state_msg);
  
  // 发布详细状态（JSON格式）
  rapidjson::Document status_doc;
  status_doc.SetObject();
  rapidjson::Document::AllocatorType& allocator = status_doc.GetAllocator();
  
  status_doc.AddMember("state", rapidjson::Value(stateToString(current_state_).c_str(), allocator), allocator);
  status_doc.AddMember("target", rapidjson::Value(target_description_.c_str(), allocator), allocator);
  status_doc.AddMember("has_target", has_target_position_, allocator);
  
  if (has_target_position_) {
    rapidjson::Value coord(rapidjson::kObjectType);
    coord.AddMember("x", target_position_.x, allocator);
    coord.AddMember("y", target_position_.y, allocator);
    coord.AddMember("z", target_position_.z, allocator);
    status_doc.AddMember("target_position", coord, allocator);
    
    if (odom_received_) {
      double distance = calculateDistance(last_robot_position_, target_position_);
      status_doc.AddMember("distance_to_target", distance, allocator);
    }
  }
  
  // 添加路径信息
  if (path_recording_enabled_) {
    status_doc.AddMember("path_length", static_cast<int>(exploration_path_.poses.size()), allocator);
    if (!exploration_path_.poses.empty()) {
      status_doc.AddMember("path_distance", calculatePathDistance(), allocator);
    }
  }
  
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  status_doc.Accept(writer);
  
  std_msgs::String status_msg;
  status_msg.data = buffer.GetString();
  status_pub_.publish(status_msg);
}

geometry_msgs::PoseStamped ExploreManager::createGoalPose(const geometry_msgs::Point& position)
{
  geometry_msgs::PoseStamped goal;
  goal.header.frame_id = map_frame_;
  goal.header.stamp = ros::Time::now();
  goal.pose.position = position;
  goal.pose.orientation.w = 1.0;  // 默认朝向
  
  return goal;
}

double ExploreManager::calculateDistance(const geometry_msgs::Point& p1, const geometry_msgs::Point& p2)
{
  double dx = p1.x - p2.x;
  double dy = p1.y - p2.y;
  double dz = p1.z - p2.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double ExploreManager::calculatePathDistance()
{
  if (exploration_path_.poses.size() < 2) {
    return 0.0;
  }
  
  double total_distance = 0.0;
  for (size_t i = 1; i < exploration_path_.poses.size(); i++) {
    const auto& p1 = exploration_path_.poses[i-1].pose.position;
    const auto& p2 = exploration_path_.poses[i].pose.position;
    total_distance += calculateDistance(p1, p2);
  }
  return total_distance;
}

std::string ExploreManager::stateToString(NavigationState state)
{
  switch (state) {
    case NavigationState::IDLE: return "IDLE";
    case NavigationState::EXPLORING: return "EXPLORING";
    case NavigationState::APPROACHING: return "APPROACHING";
    case NavigationState::REACHED: return "REACHED";
    case NavigationState::FAILED: return "FAILED";
    default: return "UNKNOWN";
  }
}

// ========== 评分地图相关函数实现 ==========

void ExploreManager::generateScoreMap(const ros::TimerEvent& event)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  
  // 检查必要的地图是否已接收
  if (!map_received_) {
    ROS_WARN_THROTTLE(5.0, "Occupancy map not received, cannot generate score map");
    return;
  }
  
  // 初始化评分地图（基于占据地图）
  score_map_ = current_map_;
  score_map_.header.frame_id = map_frame_;
  score_map_.header.stamp = ros::Time::now();
  
  // 清零评分地图
  std::fill(score_map_.data.begin(), score_map_.data.end(), 0);
  
  // 统一检测所有边界（占据边界和场景边界）
  // 如果scene_id_grid可用且兼容，则同时检测场景边界；否则只检测占据边界
  const nav_msgs::OccupancyGrid* scene_grid_ptr = nullptr;
  if (scene_id_grid_received_ && areGridsCompatible(scene_id_grid_, current_map_)) {
    scene_grid_ptr = &scene_id_grid_;
  } else if (scene_id_grid_received_) {
    ROS_WARN_THROTTLE(5.0, "Scene ID grid incompatible with occupancy map, skipping scene boundary detection");
  }
  detectAllBoundaries(score_map_, current_map_, scene_grid_ptr);
  
  // 发布评分地图
  score_map_pub_.publish(score_map_);
}

void ExploreManager::detectAllBoundaries(nav_msgs::OccupancyGrid& score_map,
                                         const nav_msgs::OccupancyGrid& occ_map,
                                         const nav_msgs::OccupancyGrid* scene_id_grid)
{
  int width = occ_map.info.width;
  int height = occ_map.info.height;
  
  // 4邻域偏移
  int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  
  // 一次遍历完成所有边界检测，提高效率
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      int8_t cell_value = occ_map.data[idx];
      
      // 只处理已知自由区域（值为0）
      if (cell_value != 0) {
        continue;
      }
      
      // ========== 1. 检测占据边界（已知自由和未知边界） ==========
      bool is_occupancy_boundary = false;
      bool has_wall_neighbor = false;  // 是否有墙壁邻域（用于过滤场景边界）
      
      for (int i = 0; i < 4; i++) {
        int nx = x + offsets[i][0];
        int ny = y + offsets[i][1];
        
        if (isValidMapCoord(nx, ny, occ_map)) {
          int nidx = ny * width + nx;
          int8_t neighbor_value = occ_map.data[nidx];
          
          // 检查是否有未知区域
          if (neighbor_value == -1) {
            is_occupancy_boundary = true;
          }
          
          // 检查是否有占据区域（墙壁）
          if (neighbor_value > 0) {
            has_wall_neighbor = true;
          }
        }
      }
      
      // 如果是占据边界，增加评分
      if (is_occupancy_boundary) {
        int score = static_cast<int>(occupancy_boundary_weight_ * 50.0);
        score_map.data[idx] = std::min(100, std::max(0, score));
      }
      
      // ========== 2. 检测场景边界（如果scene_id_grid可用且当前栅格不在墙壁边缘） ==========
      // 关键优化：如果当前栅格邻域中有墙壁，则跳过场景边界检测，避免误判墙壁边缘
      if (scene_id_grid != nullptr && !has_wall_neighbor) {
        int scene_idx = idx;
        if (scene_idx < static_cast<int>(scene_id_grid->data.size())) {
          int8_t scene_id = scene_id_grid->data[scene_idx];
          
          bool is_known_to_unknown_boundary = false;
          bool is_different_scene_boundary = false;
          
          for (int i = 0; i < 4; i++) {
            int nx = x + offsets[i][0];
            int ny = y + offsets[i][1];
            
            if (isValidMapCoord(nx, ny, *scene_id_grid)) {
              int nidx = ny * width + nx;
              
              // 跳过占据区域（墙壁）
              if (nidx < static_cast<int>(occ_map.data.size()) && occ_map.data[nidx] != 0) {
                continue;
              }
              
              // 检查场景ID边界
              if (nidx < static_cast<int>(scene_id_grid->data.size())) {
                int8_t neighbor_scene_id = scene_id_grid->data[nidx];
                
                // 情况1：已知环境属性到未知环境属性边界
                if (scene_id >= 0 && neighbor_scene_id == -1) {
                  is_known_to_unknown_boundary = true;
                } else if (scene_id == -1 && neighbor_scene_id >= 0) {
                  is_known_to_unknown_boundary = true;
                }
                
                // 情况2：不同环境属性边界（都是已知的，但ID不同）
                if (scene_id >= 0 && neighbor_scene_id >= 0 && scene_id != neighbor_scene_id) {
                  is_different_scene_boundary = true;
                }
              }
            }
          }
          
          // 如果是场景边界，累加评分
          if (is_known_to_unknown_boundary || is_different_scene_boundary) {
            int base_score = 0;
            
            if (is_known_to_unknown_boundary) {
              base_score = static_cast<int>(scene_known_unknown_boundary_weight_ * 50.0);
            } else if (is_different_scene_boundary) {
              base_score = static_cast<int>(scene_different_boundary_weight_ * 50.0);
            }
            
            // 累加到现有评分（可能已有占据边界评分）
            int current_score = static_cast<int>(score_map.data[idx]);
            int new_score = std::min(100, current_score + base_score);
            score_map.data[idx] = static_cast<int8_t>(new_score);
          }
        }
      }
    }
  }
}

bool ExploreManager::areGridsCompatible(const nav_msgs::OccupancyGrid& grid1,
                                         const nav_msgs::OccupancyGrid& grid2)
{
  // 检查分辨率
  if (std::abs(grid1.info.resolution - grid2.info.resolution) > 1e-6) {
    return false;
  }
  
  // 检查尺寸
  if (grid1.info.width != grid2.info.width || grid1.info.height != grid2.info.height) {
    return false;
  }
  
  // 检查原点（允许小的误差）
  double origin_diff_x = std::abs(grid1.info.origin.position.x - grid2.info.origin.position.x);
  double origin_diff_y = std::abs(grid1.info.origin.position.y - grid2.info.origin.position.y);
  if (origin_diff_x > grid1.info.resolution || origin_diff_y > grid1.info.resolution) {
    return false;
  }
  
  return true;
}

void ExploreManager::worldToMap(double wx, double wy, int& mx, int& my,
                                const nav_msgs::OccupancyGrid& map)
{
  mx = static_cast<int>((wx - map.info.origin.position.x) / map.info.resolution);
  my = static_cast<int>((wy - map.info.origin.position.y) / map.info.resolution);
}

bool ExploreManager::isValidMapCoord(int mx, int my, const nav_msgs::OccupancyGrid& map)
{
  return mx >= 0 && mx < static_cast<int>(map.info.width) &&
         my >= 0 && my < static_cast<int>(map.info.height);
}

}  // namespace explore_pkg
