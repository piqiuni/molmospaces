#ifndef EXPLORATION_TYPES_H_
#define EXPLORATION_TYPES_H_

#include <geometry_msgs/Point.h>
#include <string>

namespace explore_pkg
{

/**
 * @brief 探索目标结构
 */
struct ExplorationGoal
{
  geometry_msgs::Point position;
  double utility_score;
  std::string reason;
  bool is_valid;
  
  ExplorationGoal() : utility_score(0.0), is_valid(false) {}
};

/**
 * @brief 检测到的物体结构
 */
struct DetectedObject
{
  std::string semantic_name;
  double confidence;
  geometry_msgs::Point coord_3d;
  std::string env_status;
  
  DetectedObject() : confidence(0.0) {}
};

}  // namespace explore_pkg

#endif  // EXPLORATION_TYPES_H_

