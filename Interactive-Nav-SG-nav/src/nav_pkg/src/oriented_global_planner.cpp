#include <nav_pkg/oriented_global_planner.h>

#include <algorithm>
#include <cmath>

#include <nav_msgs/Path.h>
#include <pluginlib/class_list_macros.h>

namespace {

double yawFromQuaternion(const geometry_msgs::Quaternion& quaternion) {
  const double siny_cosp =
      2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y);
  const double cosy_cosp =
      1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

double shortestAngularDistance(double from, double to) {
  return std::atan2(std::sin(to - from), std::cos(to - from));
}

geometry_msgs::Quaternion quaternionFromYaw(double yaw) {
  geometry_msgs::Quaternion quaternion;
  quaternion.z = std::sin(0.5 * yaw);
  quaternion.w = std::cos(0.5 * yaw);
  return quaternion;
}

}  // namespace

namespace nav_pkg {

OrientedGlobalPlanner::OrientedGlobalPlanner()
    : initialized_(false),
      preserve_goal_orientation_(true),
      orient_path_tangents_(true),
      terminal_orientation_window_size_(8) {}

OrientedGlobalPlanner::OrientedGlobalPlanner(
    std::string name, costmap_2d::Costmap2DROS* costmap_ros)
    : OrientedGlobalPlanner() {
  initialize(std::move(name), costmap_ros);
}

void OrientedGlobalPlanner::initialize(
    std::string name, costmap_2d::Costmap2DROS* costmap_ros) {
  if (initialized_) {
    return;
  }
  ros::NodeHandle private_nh("~/" + name);
  private_nh.param(
      "preserve_goal_orientation", preserve_goal_orientation_, true);
  private_nh.param("orient_path_tangents", orient_path_tangents_, true);
  private_nh.param(
      "terminal_orientation_window_size", terminal_orientation_window_size_, 8);
  terminal_orientation_window_size_ =
      std::max(1, terminal_orientation_window_size_);

  delegate_ = std::make_unique<global_planner::GlobalPlanner>();
  delegate_->initialize("GlobalPlanner", costmap_ros);
  ros::NodeHandle move_base_private("~");
  corrected_plan_pub_ =
      move_base_private.advertise<nav_msgs::Path>("GlobalPlanner/plan", 1);
  initialized_ = true;
}

bool OrientedGlobalPlanner::makePlan(
    const geometry_msgs::PoseStamped& start,
    const geometry_msgs::PoseStamped& goal,
    std::vector<geometry_msgs::PoseStamped>& plan) {
  if (!initialized_ || delegate_ == nullptr) {
    ROS_ERROR("OrientedGlobalPlanner has not been initialized");
    return false;
  }
  if (!delegate_->makePlan(start, goal, plan)) {
    return false;
  }
  if (orient_path_tangents_) {
    applyPathTangentialOrientation(plan);
  }
  if (preserve_goal_orientation_) {
    applyTerminalOrientation(goal, plan);
  }
  publishCorrectedPlan(plan);
  return true;
}

void OrientedGlobalPlanner::applyPathTangentialOrientation(
    std::vector<geometry_msgs::PoseStamped>& plan) const {
  if (plan.size() < 2) {
    return;
  }
  for (std::size_t index = 0; index < plan.size(); ++index) {
    std::size_t next_index = index + 1;
    while (next_index < plan.size()) {
      const double dx = plan[next_index].pose.position.x -
                        plan[index].pose.position.x;
      const double dy = plan[next_index].pose.position.y -
                        plan[index].pose.position.y;
      if (std::hypot(dx, dy) > 1e-6) {
        plan[index].pose.orientation = quaternionFromYaw(std::atan2(dy, dx));
        break;
      }
      ++next_index;
    }
    if (next_index < plan.size()) {
      continue;
    }
    std::size_t previous_index = index;
    while (previous_index > 0) {
      --previous_index;
      const double dx = plan[index].pose.position.x -
                        plan[previous_index].pose.position.x;
      const double dy = plan[index].pose.position.y -
                        plan[previous_index].pose.position.y;
      if (std::hypot(dx, dy) > 1e-6) {
        plan[index].pose.orientation = quaternionFromYaw(std::atan2(dy, dx));
        break;
      }
    }
  }
}

void OrientedGlobalPlanner::applyTerminalOrientation(
    const geometry_msgs::PoseStamped& goal,
    std::vector<geometry_msgs::PoseStamped>& plan) const {
  if (plan.empty()) {
    return;
  }
  const std::size_t last_index = plan.size() - 1;
  if (last_index == 0) {
    plan.back().pose.orientation = goal.pose.orientation;
    return;
  }
  const std::size_t window_size = std::min<std::size_t>(
      static_cast<std::size_t>(terminal_orientation_window_size_), last_index);
  const std::size_t start_index = last_index - window_size;
  const double start_yaw = yawFromQuaternion(plan[start_index].pose.orientation);
  const double goal_yaw = yawFromQuaternion(goal.pose.orientation);
  const double yaw_delta = shortestAngularDistance(start_yaw, goal_yaw);

  for (std::size_t index = start_index + 1; index <= last_index; ++index) {
    const double ratio = static_cast<double>(index - start_index) /
                         static_cast<double>(last_index - start_index);
    plan[index].pose.orientation =
        quaternionFromYaw(start_yaw + ratio * yaw_delta);
  }
  plan.back().pose.orientation = quaternionFromYaw(goal_yaw);
}

void OrientedGlobalPlanner::publishCorrectedPlan(
    const std::vector<geometry_msgs::PoseStamped>& plan) const {
  if (plan.empty()) {
    return;
  }
  nav_msgs::Path message;
  message.header = plan.front().header;
  message.poses = plan;
  corrected_plan_pub_.publish(message);
}

}  // namespace nav_pkg

PLUGINLIB_EXPORT_CLASS(nav_pkg::OrientedGlobalPlanner, nav_core::BaseGlobalPlanner)
