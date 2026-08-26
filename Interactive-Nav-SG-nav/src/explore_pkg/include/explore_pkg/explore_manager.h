#ifndef EXPLORE_MANAGER_H_
#define EXPLORE_MANAGER_H_

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/String.h>
#include <std_msgs/Empty.h>
#include <actionlib_msgs/GoalStatusArray.h>
#include <memory>
#include <vector>
#include <string>
#include <mutex>

#include "explore_pkg/voronoi_explorer.h"
#include "explore_pkg/exploration_types.h"

namespace explore_pkg
{

/**
 * @brief 导航状态枚举
 */
enum class NavigationState
{
  IDLE,          // 空闲，等待目标
  EXPLORING,     // 探索中，寻找目标
  APPROACHING,   // 接近目标
  REACHED,       // 已到达目标
  FAILED         // 导航失败
};

/**
 * @class ExploreManager
 * @brief 零样本目标导航管理器
 * 
 * 负责：
 * - 状态机管理
 * - 接收语义地图（由感知模块处理）
 * - 目标验证与确认
 * - 导航目标调度
 * - 探索路径记录
 * - ROS接口
 */
class ExploreManager
{
public:
  ExploreManager();
  ~ExploreManager();

  /**
   * @brief 初始化管理器
   */
  bool initialize();

  /**
   * @brief 启动管理器
   */
  void start();

  /**
   * @brief 停止管理器
   */
  void stop();

  /**
   * @brief 获取当前状态
   */
  NavigationState getCurrentState() const { return current_state_; }

private:
  // ========== 初始化函数 ==========
  void loadParameters();
  void setupPlanner();

  // ========== 回调函数 ==========
  void semanticMapCallback(const std_msgs::StringConstPtr& msg);
  void odomCallback(const nav_msgs::OdometryConstPtr& msg);
  void mapCallback(const nav_msgs::OccupancyGridConstPtr& msg);
  void sceneIdGridCallback(const nav_msgs::OccupancyGridConstPtr& msg);
  void moveBaseStatusCallback(const actionlib_msgs::GoalStatusArrayConstPtr& msg);
  void resetCallback(const std_msgs::EmptyConstPtr& msg);

  // ========== 核心功能函数 ==========
  void stateMachineLoop(const ros::TimerEvent& event);
  void processSemanticMap(const std::string& json_data);
  void checkGoalReached(const ros::TimerEvent& event);
  bool planToTarget();
  void executeExploration(const ros::TimerEvent& event);
  void updateExplorationPath();
  void recoveryRotateCallback(const ros::TimerEvent& event);
  
  // ========== 辅助函数 ==========
  void updateState(NavigationState new_state);
  void publishStatus();
  geometry_msgs::PoseStamped createGoalPose(const geometry_msgs::Point& position);
  double calculateDistance(const geometry_msgs::Point& p1, const geometry_msgs::Point& p2);
  double calculatePathDistance();  // 计算路径总长度
  std::string stateToString(NavigationState state);
  void startRecoveryRotation();
  void stopRecoveryRotation();
  
  // ========== 评分地图相关函数 ==========
  /**
   * @brief 生成评分地图（定时器回调）
   * @details 评分地图由两部分组成：
   *          1. 占据边界（已知自由和未知边界）
   *          2. 环境属性边界（已知环境属性和未知环境属性边界，以及已知不同环境属性的边界）
   */
  void generateScoreMap(const ros::TimerEvent& event);
  
  /**
   * @brief 统一检测所有边界（占据边界和场景边界）
   * @param score_map 输出的评分地图（会被修改）
   * @param occ_map 占据栅格地图
   * @param scene_id_grid 场景ID栅格地图（可选，如果为空则只检测占据边界）
   * 
   * 优化说明：
   * - 合并三个边界检测函数为一个，一次遍历完成所有检测，提高效率
   * - 添加墙壁检测：如果栅格邻域中有占据区域（墙壁），则跳过场景边界评分，避免误判墙壁边缘
   */
  void detectAllBoundaries(nav_msgs::OccupancyGrid& score_map,
                           const nav_msgs::OccupancyGrid& occ_map,
                           const nav_msgs::OccupancyGrid* scene_id_grid = nullptr);
  
  /**
   * @brief 检查两个栅格地图是否兼容（分辨率、尺寸、原点相同）
   */
  bool areGridsCompatible(const nav_msgs::OccupancyGrid& grid1,
                          const nav_msgs::OccupancyGrid& grid2);
  
  /**
   * @brief 世界坐标转地图坐标
   */
  void worldToMap(double wx, double wy, int& mx, int& my,
                   const nav_msgs::OccupancyGrid& map);
  
  /**
   * @brief 检查地图坐标是否有效
   */
  bool isValidMapCoord(int mx, int my, const nav_msgs::OccupancyGrid& map);

  // ========== ROS节点句柄 ==========
  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;

  // ========== 订阅者 ==========
  ros::Subscriber semantic_map_sub_;
  ros::Subscriber odom_sub_;
  ros::Subscriber map_sub_;
  ros::Subscriber scene_id_grid_sub_;
  ros::Subscriber move_base_status_sub_;
  ros::Subscriber reset_sub_;

  // ========== 发布者 ==========
  ros::Publisher goal_pub_;
  ros::Publisher goal_point_pub_;
  ros::Publisher cmd_vel_pub_;
  ros::Publisher state_pub_;
  ros::Publisher status_pub_;
  ros::Publisher exploration_path_pub_;
  ros::Publisher score_map_pub_;

  // ========== 探索算法 ==========
  VoronoiExplorer explorer_;

  // ========== 状态管理 ==========
  NavigationState current_state_;
  std::mutex state_mutex_;

  // ========== 目标管理 ==========
  std::string target_description_;
  geometry_msgs::Point target_position_;
  bool has_target_position_;

  // ========== 语义地图相关 ==========
  std::vector<DetectedObject> semantic_objects_;  // 当前语义地图中的物体
  ros::Time last_semantic_map_time_;

  // ========== 机器人状态 ==========
  nav_msgs::Odometry current_odom_;
  geometry_msgs::PoseStamped current_pose_;
  geometry_msgs::Point last_robot_position_;
  bool odom_received_;

  // ========== 地图和探索 ==========
  nav_msgs::OccupancyGrid current_map_;
  nav_msgs::OccupancyGrid scene_id_grid_;
  nav_msgs::OccupancyGrid score_map_;
  std::vector<geometry_msgs::Point> visited_points_;
  bool map_received_;
  bool scene_id_grid_received_;

  // ========== 探索路径记录 ==========
  nav_msgs::Path exploration_path_;
  geometry_msgs::Point last_recorded_position_;
  bool path_recording_enabled_;
  double path_sampling_distance_;

  // ========== 导航相关 ==========
  geometry_msgs::PoseStamped current_goal_;
  bool has_active_goal_;
  bool rotating_in_place_;
  ros::Time last_goal_publish_time_;
  std::string last_failed_goal_id_;
  ros::Time last_move_base_failure_handle_time_;

  // ========== 定时器 ==========
  ros::Timer exploration_timer_;
  ros::Timer goal_check_timer_;
  ros::Timer state_machine_timer_;
  ros::Timer score_map_timer_;  // 评分地图更新定时器
  ros::Timer recovery_rotate_timer_;  // 原地旋转恢复定时器

  // ========== 参数 ==========
  // 话题名称
  std::string semantic_map_topic_;
  std::string goal_topic_;
  std::string odom_topic_;
  std::string map_topic_;
  std::string cmd_vel_topic_;
  std::string move_base_status_topic_;
  std::string exploration_path_topic_;
  std::string scene_id_grid_topic_;
  std::string score_map_topic_;
  std::string reset_topic_;
  
  // 坐标系
  std::string base_frame_;
  std::string map_frame_;
  
  // 阈值
  double target_reach_threshold_;
  double semantic_confidence_threshold_;
  
  // 导航参数
  double exploration_rate_;
  double goal_check_rate_;
  double goal_republish_interval_;  // 同一目标重发间隔（秒）
  double move_base_failure_cooldown_;  // 处理move_base失败状态的最小间隔（秒）
  double failed_goal_blacklist_duration_;  // 失败目标黑名单持续时间（秒）
  double recovery_rotate_speed_;  // 原地旋转角速度（rad/s）
  double recovery_rotate_rate_;   // 原地旋转控制发布频率（Hz）
  
  // 探索参数
  std::string exploration_algorithm_type_;
  
  // 路径记录参数
  bool path_recording_enabled_param_;
  double path_sampling_distance_param_;
  
  // 评分地图参数
  double occupancy_boundary_weight_;                    // 占据边界权重
  double scene_known_unknown_boundary_weight_;          // 已知/未知环境属性边界权重
  double scene_different_boundary_weight_;             // 不同环境属性边界权重
  int score_map_publish_rate_;                         // 评分地图发布频率（Hz）
};

}  // namespace explore_pkg

#endif  // EXPLORE_MANAGER_H_

