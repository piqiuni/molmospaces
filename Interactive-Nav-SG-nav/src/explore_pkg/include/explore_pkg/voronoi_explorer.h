#ifndef VORONOI_EXPLORER_H_
#define VORONOI_EXPLORER_H_

#include "explore_pkg/tsp_solver.h"
#include "explore_pkg/exploration_types.h"
#include "explore_pkg/voronoi_explorer_visualization.h"  // 包含完整定义以支持unique_ptr
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>
#include <ros/ros.h>
#include <vector>
#include <string>
#include <memory>

namespace explore_pkg
{

/**
 * @brief Voronoi节点结构（仅用于构建图）
 */
struct VoronoiNode
{
  geometry_msgs::Point position;
  int node_id;
  std::vector<int> neighbors;
  double information_gain;  // 信息增益（基于评分地图）
  
  VoronoiNode() : node_id(-1), information_gain(0.0) {}
};

/**
 * @class VoronoiExplorer
 * @brief 基于Voronoi图的探索算法
 * 
 * 核心功能：
 * - 从Voronoi图中提取节点
 * - 基于评分地图选择高价值节点
 * - 使用TSP求解器规划访问顺序
 * - 返回下一个探索目标
 */
class VoronoiExplorer
{
public:
  VoronoiExplorer();
  ~VoronoiExplorer();  // 在cpp文件中实现，需要包含可视化头文件

  /**
   * @brief 初始化
   */
  bool initialize(ros::NodeHandle& nh, ros::NodeHandle& private_nh);
  
  /**
   * @brief 选择下一个探索目标
   */
  ExplorationGoal selectNextGoal(
      const nav_msgs::OccupancyGrid& map,
      const geometry_msgs::Pose& current_pose);

  /**
   * @brief 更新检测结果（可选）
   */
  void updateDetections(const std::vector<DetectedObject>& detections);

  /**
   * @brief 为接近目标规划路径（使用维诺图）
   * @param target_pos 目标位置
   * @param current_pos 当前位置
   * @param map 占据栅格地图（用于检查目标是否在自由空间）
   * @return 探索目标（如果找到最近的维诺节点，返回该节点位置；否则返回目标位置）
   * 
   * 策略：
   * 1. 找到离目标最近的维诺节点作为导航目标
   * 2. 始终使用维诺节点（不使用目标直接位置），因为目标可能很大（如车、床等），
   *    需要导航到附近的维诺节点再接近目标
   */
  ExplorationGoal planApproachToTarget(
      const geometry_msgs::Point& target_pos,
      const geometry_msgs::Point& current_pos,
      const nav_msgs::OccupancyGrid& map);

  /**
   * @brief 添加临时黑名单目标点（在持续时间内不会被选为TSP目标）
   */
  void addTemporaryBlacklistPoint(const geometry_msgs::Point& point, double duration_sec);

private:
  // ========== Voronoi图构建 ==========
  void buildVoronoiGraph(const nav_msgs::OccupancyGrid& voronoi_map);
  void extractVoronoiNodes(const nav_msgs::OccupancyGrid& voronoi_map);
  bool isVoronoiNode(int x, int y, const nav_msgs::OccupancyGrid& voronoi_map);
  int countNeighbors(int x, int y, const nav_msgs::OccupancyGrid& voronoi_map);
  
  // ========== 节点评估 ==========
  double calculateInformationGain(const VoronoiNode& node, 
                                 const nav_msgs::OccupancyGrid& score_map);
  
  // ========== TSP路径规划 ==========
  std::vector<geometry_msgs::Point> selectHighValueNodesForTSP(
      const nav_msgs::OccupancyGrid& score_map);
  std::vector<std::vector<int>> buildTSPDistanceMatrix(
      const std::vector<geometry_msgs::Point>& selected_positions,
      const geometry_msgs::Point& current_pos);
  void solveTSP(const std::vector<geometry_msgs::Point>& selected_positions,
                const geometry_msgs::Point& current_pos);
  ExplorationGoal selectNextGoalFromTSP(const geometry_msgs::Point& current_pos);
  
  // ========== 辅助函数 ==========
  void worldToMap(double wx, double wy, int& mx, int& my,
                  const nav_msgs::OccupancyGrid& map);
  bool isValid(int mx, int my, const nav_msgs::OccupancyGrid& map);
  double calculateDistance(const geometry_msgs::Point& p1, 
                          const geometry_msgs::Point& p2);
  double calculateDistance2D(const geometry_msgs::Point& p1,
                             const geometry_msgs::Point& p2);
  void purgeExpiredBlacklistedGoals();
  bool isTemporarilyBlacklisted(const geometry_msgs::Point& point);
  int getSceneScoreAtPosition(const geometry_msgs::Point& pos);  // 获取位置的场景分数
  
  // ========== Scene Score Grid ==========
  void sceneIdGridCallback(const nav_msgs::OccupancyGridConstPtr& msg);
  void generateSceneScoreGrid();
  bool loadSceneIdScores();
  
  // ========== TSP重新规划 ==========
  void tspReplanTimerCallback(const ros::TimerEvent& event);
  
  // ========== 回调函数 ==========
  void voronoiMapCallback(const nav_msgs::OccupancyGridConstPtr& msg);
  void scoreMapCallback(const nav_msgs::OccupancyGridConstPtr& msg);
  
  // ========== 成员变量 ==========
  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  
  // Voronoi图
  std::vector<VoronoiNode> voronoi_nodes_;
  bool voronoi_map_received_;
  std::string voronoi_map_topic_;
  ros::Subscriber voronoi_map_sub_;
  
  // 评分地图
  nav_msgs::OccupancyGrid score_map_;
  bool score_map_received_;
  std::string score_map_topic_;
  std::string scene_id_grid_topic_;
  nav_msgs::OccupancyGrid scene_id_grid_;
  bool scene_id_grid_received_;
  ros::Subscriber score_map_sub_;
  ros::Subscriber scene_id_grid_sub_;
  
  // Scene Score Grid
  nav_msgs::OccupancyGrid scene_score_grid_;
  ros::Publisher scene_score_grid_pub_;
  std::string scene_id_scores_yaml_path_;  // JSON配置文件路径（变量名保持兼容）
  std::map<int, int> scene_id_scores_map_;  // scene_id -> score映射
  int default_scene_score_;  // 默认分数
  
  // TSP路径（独立存储，只存储位置，不依赖Voronoi节点）
  std::vector<geometry_msgs::Point> tsp_path_;  // TSP路径中的位置序列
  int tsp_current_index_;                       // 当前在TSP路径中的索引
  geometry_msgs::Point current_goal_position_; // 当前目标位置
  ros::Time current_goal_set_time_;             // 当前目标设置时间
  double tsp_goal_reach_threshold_;            // 到达阈值（米）
  double tsp_goal_timeout_;                    // 超时时间（秒）
  
  // 参数
  double tsp_top_num_;  // TSP节点选择比例（0.0-1.0，默认0.3表示30%）
  double tsp_scene_score_weight_;  // 场景分数对TSP代价的影响权重（0.0-1.0，默认0.3）
  double tsp_replan_frequency_;  // TSP重新规划频率（Hz），-1表示不按时间重新规划
  double blacklist_match_threshold_;  // 黑名单点匹配阈值（米）

  struct BlacklistedGoal
  {
    geometry_msgs::Point point;
    ros::Time expire_time;
  };
  std::vector<BlacklistedGoal> temporary_blacklisted_goals_;
  
  // TSP重新规划定时器
  ros::Timer tsp_replan_timer_;
  
  // 当前地图（用于坐标转换）
  nav_msgs::OccupancyGrid current_map_;
  
  // ========== 可视化 ==========
  std::unique_ptr<VoronoiExplorerVisualization> visualization_;
  
  // 可视化辅助函数
  void updateVisualization();
};

}  // namespace explore_pkg

#endif  // VORONOI_EXPLORER_H_
