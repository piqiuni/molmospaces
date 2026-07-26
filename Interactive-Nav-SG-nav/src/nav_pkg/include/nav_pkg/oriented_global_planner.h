#pragma once

#include <memory>
#include <string>
#include <vector>

#include <costmap_2d/costmap_2d_ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <global_planner/planner_core.h>
#include <nav_core/base_global_planner.h>
#include <ros/ros.h>

namespace nav_pkg {

class OrientedGlobalPlanner : public nav_core::BaseGlobalPlanner {
 public:
  OrientedGlobalPlanner();
  OrientedGlobalPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros);

  void initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros) override;

  bool makePlan(const geometry_msgs::PoseStamped& start,
                const geometry_msgs::PoseStamped& goal,
                std::vector<geometry_msgs::PoseStamped>& plan) override;

 private:
  void applyPathTangentialOrientation(
      std::vector<geometry_msgs::PoseStamped>& plan) const;
  void applyTerminalOrientation(const geometry_msgs::PoseStamped& goal,
                                std::vector<geometry_msgs::PoseStamped>& plan) const;
  void publishCorrectedPlan(const std::vector<geometry_msgs::PoseStamped>& plan) const;

  bool initialized_;
  bool preserve_goal_orientation_;
  bool orient_path_tangents_;
  int terminal_orientation_window_size_;
  std::unique_ptr<global_planner::GlobalPlanner> delegate_;
  ros::Publisher corrected_plan_pub_;
};

}  // namespace nav_pkg
