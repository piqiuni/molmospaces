#ifndef VORONOI_EXPLORER_VISUALIZATION_H_
#define VORONOI_EXPLORER_VISUALIZATION_H_

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <vector>

namespace explore_pkg
{

// 前向声明（在cpp文件中包含完整定义）
struct VoronoiNode;

/**
 * @class VoronoiExplorerVisualization
 * @brief Voronoi探索器的可视化类
 * 
 * 负责发布可视化标记：
 * - Voronoi节点评分可视化（柱状图）
 * - TSP路径可视化（路径线和节点）
 */
class VoronoiExplorerVisualization
{
public:
  VoronoiExplorerVisualization();
  ~VoronoiExplorerVisualization();

  /**
   * @brief 初始化可视化发布者
   */
  bool initialize(ros::NodeHandle& nh);

  /**
   * @brief 发布评分节点的可视化（柱状图）
   * @param nodes Voronoi节点列表
   * @param current_pose 当前机器人位姿（用于计算评分）
   * @param score_map 评分地图
   */
  void publishScoredMarkers(const std::vector<VoronoiNode>& nodes,
                            const geometry_msgs::Pose& current_pose,
                            const nav_msgs::OccupancyGrid& score_map);

  /**
   * @brief 发布TSP路径可视化
   * @param tsp_path TSP路径位置序列
   * @param current_index 当前路径索引
   * @param current_goal_position 当前目标位置
   * @param frame_id 坐标系ID
   */
  void publishTSPPath(const std::vector<geometry_msgs::Point>& tsp_path,
                      int current_index,
                      const geometry_msgs::Point& current_goal_position,
                      const std::string& frame_id);

private:
  ros::Publisher scored_marker_pub_;
  ros::Publisher tsp_path_pub_;
  
  bool initialized_;
  
  // 辅助函数
  void worldToMap(double wx, double wy, int& mx, int& my,
                  const nav_msgs::OccupancyGrid& map);
  bool isValid(int mx, int my, const nav_msgs::OccupancyGrid& map);
  double calculateDistance(const geometry_msgs::Point& p1, 
                          const geometry_msgs::Point& p2);
};

}  // namespace explore_pkg

#endif  // VORONOI_EXPLORER_VISUALIZATION_H_

