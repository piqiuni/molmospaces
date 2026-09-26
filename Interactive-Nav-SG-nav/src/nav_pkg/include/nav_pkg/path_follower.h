#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <costmap_2d/costmap_2d_ros.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_core/base_local_planner.h>
#include <nav_msgs/Path.h>
#include <nav_pkg/PathFollowerConfig.h>
#include <ros/ros.h>
#include <tf2_ros/buffer.h>

namespace nav_pkg {

class PathFollower : public nav_core::BaseLocalPlanner {
 public:
  PathFollower();
  void initialize(std::string name, tf2_ros::Buffer* tf,
                  costmap_2d::Costmap2DROS* costmap_ros) override;
  bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan) override;
  bool computeVelocityCommands(geometry_msgs::Twist& command) override;
  bool isGoalReached() override;

 private:
  struct Point {
    double x;
    double y;
  };
  struct Projection {
    double distance;
    double progress;
    Point point;
    std::size_t segment;
  };

  Projection project(Point position, std::size_t start_segment) const;
  Point atDistance(double distance) const;
  bool safeCommand(const geometry_msgs::Twist& command,
                   const geometry_msgs::PoseStamped& robot,
                   const Projection& start, nav_msgs::Path* trajectory) const;
  bool transformPose(const geometry_msgs::PoseStamped& pose,
                     const std::string& frame,
                     geometry_msgs::PoseStamped* transformed) const;
  void updateTolerances(PathFollowerConfig& config, uint32_t level);

  tf2_ros::Buffer* tf_;
  costmap_2d::Costmap2DROS* costmap_ros_;
  std::vector<geometry_msgs::PoseStamped> plan_;
  std::vector<double> lengths_;
  ros::Publisher local_plan_pub_;
  ros::Publisher legacy_local_plan_pub_;
  std::unique_ptr<dynamic_reconfigure::Server<PathFollowerConfig>> configuration_server_;
  std::size_t last_segment_;
  double last_progress_;
  double last_linear_speed_;
  double last_angular_speed_;
  double lookahead_;
  double lookahead_time_;
  double max_lookahead_;
  double max_lateral_accel_;
  double corridor_;
  double control_dt_;
  double max_linear_speed_;
  double max_angular_speed_;
  double max_linear_accel_;
  double max_angular_accel_;
  std::atomic<double> xy_tolerance_;
  std::atomic<double> yaw_tolerance_;
  double turn_threshold_;
  double turn_exit_threshold_;
  bool initialized_;
  bool position_reached_;
  bool goal_reached_;
  bool rotating_to_path_;
};

}
