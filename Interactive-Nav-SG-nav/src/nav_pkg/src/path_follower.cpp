#include <nav_pkg/path_follower.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <base_local_planner/costmap_model.h>
#include <pluginlib/class_list_macros.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace {

double wrapAngle(double value) {
  return std::atan2(std::sin(value), std::cos(value));
}

double distance(double x1, double y1, double x2, double y2) {
  return std::hypot(x1 - x2, y1 - y2);
}

geometry_msgs::Quaternion rotation(double yaw) {
  tf2::Quaternion quaternion;
  quaternion.setRPY(0.0, 0.0, yaw);
  return tf2::toMsg(quaternion);
}

}

namespace nav_pkg {

PathFollower::PathFollower()
    : tf_(nullptr), costmap_ros_(nullptr), last_segment_(0), last_progress_(0.0),
      last_linear_speed_(0.0), last_angular_speed_(0.0), lookahead_(0.2),
      corridor_(0.18), control_dt_(0.2), max_linear_speed_(0.35),
      max_angular_speed_(1.1), max_linear_accel_(2.5), max_angular_accel_(1.2),
      xy_tolerance_(0.2), yaw_tolerance_(0.2), turn_threshold_(0.7),
      initialized_(false), position_reached_(false), goal_reached_(false) {}

void PathFollower::initialize(std::string name, tf2_ros::Buffer* tf,
                              costmap_2d::Costmap2DROS* costmap_ros) {
  if (initialized_) return;
  tf_ = tf;
  costmap_ros_ = costmap_ros;
  ros::NodeHandle parameters("~/" + name);
  parameters.param("lookahead_m", lookahead_, lookahead_);
  parameters.param("max_path_deviation_m", corridor_, corridor_);
  parameters.param("control_dt_s", control_dt_, control_dt_);
  parameters.param("max_vel_x", max_linear_speed_, max_linear_speed_);
  parameters.param("max_vel_theta", max_angular_speed_, max_angular_speed_);
  parameters.param("acc_lim_x", max_linear_accel_, max_linear_accel_);
  parameters.param("acc_lim_theta", max_angular_accel_, max_angular_accel_);
  double xy_tolerance = xy_tolerance_.load();
  double yaw_tolerance = yaw_tolerance_.load();
  parameters.param("xy_goal_tolerance", xy_tolerance, xy_tolerance);
  parameters.param("yaw_goal_tolerance", yaw_tolerance, yaw_tolerance);
  xy_tolerance_.store(xy_tolerance);
  yaw_tolerance_.store(yaw_tolerance);
  parameters.param("turn_in_place_threshold", turn_threshold_, turn_threshold_);
  ros::NodeHandle move_base("~");
  local_plan_pub_ = parameters.advertise<nav_msgs::Path>("local_plan", 1);
  legacy_local_plan_pub_ = move_base.advertise<nav_msgs::Path>("DWAPlannerROS/local_plan", 1);
  initialized_ = tf_ && costmap_ros_ && lookahead_ > 0.0 && corridor_ > 0.0 &&
                 control_dt_ > 0.0 && max_linear_speed_ > 0.0 && max_angular_speed_ > 0.0;
  if (!initialized_) ROS_ERROR("PathFollower invalid initialization or parameters");
  if (initialized_) {
    configuration_server_ = std::make_unique<dynamic_reconfigure::Server<PathFollowerConfig>>(parameters);
    configuration_server_->setCallback([this](PathFollowerConfig& config, uint32_t level) {
      updateTolerances(config, level);
    });
  }
}

void PathFollower::updateTolerances(PathFollowerConfig& config, uint32_t) {
  xy_tolerance_.store(config.xy_goal_tolerance);
  yaw_tolerance_.store(config.yaw_goal_tolerance);
}

bool PathFollower::setPlan(const std::vector<geometry_msgs::PoseStamped>& plan) {
  if (!initialized_ || plan.empty() || plan.front().header.frame_id.empty()) return false;
  for (const auto& pose : plan) {
    if (pose.header.frame_id != plan.front().header.frame_id ||
        !std::isfinite(pose.pose.position.x) || !std::isfinite(pose.pose.position.y)) return false;
  }
  const bool same_goal = !plan_.empty() &&
      distance(plan_.back().pose.position.x, plan_.back().pose.position.y,
               plan.back().pose.position.x, plan.back().pose.position.y) < 0.05 &&
      std::abs(wrapAngle(tf2::getYaw(plan_.back().pose.orientation) -
                         tf2::getYaw(plan.back().pose.orientation))) < 0.1;
  plan_ = plan;
  lengths_.assign(plan_.size(), 0.0);
  for (std::size_t index = 1; index < plan_.size(); ++index) {
    lengths_[index] = lengths_[index - 1] + distance(
        plan_[index - 1].pose.position.x, plan_[index - 1].pose.position.y,
        plan_[index].pose.position.x, plan_[index].pose.position.y);
  }
  last_segment_ = 0;
  last_progress_ = 0.0;
  if (!same_goal) {
    position_reached_ = false;
    goal_reached_ = false;
    last_linear_speed_ = 0.0;
    last_angular_speed_ = 0.0;
  }
  return true;
}

PathFollower::Projection PathFollower::project(Point position, std::size_t start_segment) const {
  Projection best{std::numeric_limits<double>::infinity(), 0.0, position, 0};
  if (plan_.size() == 1) {
    best.point = {plan_.front().pose.position.x, plan_.front().pose.position.y};
    best.distance = distance(position.x, position.y, best.point.x, best.point.y);
    return best;
  }
  for (std::size_t index = start_segment; index + 1 < plan_.size(); ++index) {
    const auto& first = plan_[index].pose.position;
    const auto& second = plan_[index + 1].pose.position;
    const double dx = second.x - first.x;
    const double dy = second.y - first.y;
    const double squared = dx * dx + dy * dy;
    const double fraction = squared > 1e-12
        ? std::clamp(((position.x - first.x) * dx + (position.y - first.y) * dy) / squared,
                     0.0, 1.0)
        : 0.0;
    const Point closest{first.x + fraction * dx, first.y + fraction * dy};
    const double separation = distance(position.x, position.y, closest.x, closest.y);
    const double progress = lengths_[index] + fraction * std::sqrt(squared);
    if (separation < best.distance - 1e-6 ||
        (std::abs(separation - best.distance) < 1e-6 && progress > best.progress)) {
      best = {separation, progress, closest, index};
    }
  }
  return best;
}

PathFollower::Point PathFollower::atDistance(double progress) const {
  if (progress >= lengths_.back()) {
    return {plan_.back().pose.position.x, plan_.back().pose.position.y};
  }
  auto next = std::upper_bound(lengths_.begin(), lengths_.end(), progress);
  const std::size_t index = std::max<std::size_t>(1, next - lengths_.begin());
  const double span = lengths_[index] - lengths_[index - 1];
  const double fraction = span > 1e-9 ? (progress - lengths_[index - 1]) / span : 0.0;
  return {plan_[index - 1].pose.position.x + fraction *
              (plan_[index].pose.position.x - plan_[index - 1].pose.position.x),
          plan_[index - 1].pose.position.y + fraction *
              (plan_[index].pose.position.y - plan_[index - 1].pose.position.y)};
}

bool PathFollower::transformPose(const geometry_msgs::PoseStamped& pose,
                                 const std::string& frame,
                                 geometry_msgs::PoseStamped* transformed) const {
  try {
    tf_->transform(pose, *transformed, frame, ros::Duration(0.05));
    return true;
  } catch (const tf2::TransformException& error) {
    ROS_WARN_THROTTLE(2.0, "PathFollower transform failed: %s", error.what());
    return false;
  }
}

bool PathFollower::safeCommand(const geometry_msgs::Twist& command,
                               const geometry_msgs::PoseStamped& robot,
                               const Projection& start, nav_msgs::Path* trajectory) const {
  const auto* costmap = costmap_ros_->getCostmap();
  base_local_planner::CostmapModel collision(*costmap);
  const auto footprint = costmap_ros_->getRobotFootprint();
  geometry_msgs::PoseStamped position = robot;
  geometry_msgs::PoseStamped in_path;
  const std::string& frame = plan_.front().header.frame_id;
  trajectory->header.frame_id = robot.header.frame_id;
  trajectory->header.stamp = ros::Time::now();
  trajectory->poses.clear();
  trajectory->poses.push_back(robot);
  double yaw = tf2::getYaw(position.pose.orientation);
  const int steps = std::max(3, static_cast<int>(std::ceil(0.5 / control_dt_)));
  for (int step = 0; step < steps; ++step) {
    const double step_heading = yaw;
    const int subdivisions = std::max({1,
        static_cast<int>(std::ceil(std::abs(command.linear.x) * control_dt_ / 0.025)),
        static_cast<int>(std::ceil(std::abs(command.angular.z) * control_dt_ / 0.08))});
    const double sample_dt = control_dt_ / subdivisions;
    for (int sample = 0; sample < subdivisions; ++sample) {
      position.pose.position.x += command.linear.x * sample_dt * std::cos(step_heading);
      position.pose.position.y += command.linear.x * sample_dt * std::sin(step_heading);
      yaw = wrapAngle(yaw + command.angular.z * sample_dt);
      position.pose.orientation = rotation(yaw);
      if (collision.footprintCost(position.pose.position.x, position.pose.position.y,
                                  yaw, footprint, 0.0, 0.0) < 0.0) return false;
      if (!transformPose(position, frame, &in_path)) return false;
      const Projection now = project({in_path.pose.position.x, in_path.pose.position.y},
                                     start.segment > 2 ? start.segment - 2 : 0);
      if (now.distance > std::max(corridor_, start.distance - 0.005) + 0.02 ||
          now.progress + 0.12 < std::max(start.progress, last_progress_)) return false;
      trajectory->poses.push_back(position);
    }
  }
  return true;
}

bool PathFollower::computeVelocityCommands(geometry_msgs::Twist& command) {
  command = geometry_msgs::Twist();
  if (!initialized_ || plan_.empty()) return false;
  geometry_msgs::PoseStamped robot;
  if (!costmap_ros_->getRobotPose(robot)) return false;
  geometry_msgs::PoseStamped in_path;
  if (!transformPose(robot, plan_.front().header.frame_id, &in_path)) return false;
  const Point current{in_path.pose.position.x, in_path.pose.position.y};
  Projection nearest = project(current, last_segment_ > 3 ? last_segment_ - 3 : 0);
  if (nearest.distance > corridor_ + 0.25) {
    ROS_WARN_THROTTLE(2.0, "PathFollower too far from path: %.3f m", nearest.distance);
    return false;
  }
  if (nearest.progress + 0.15 < last_progress_) {
    nearest = project(current, last_segment_);
  }
  last_segment_ = nearest.segment;
  last_progress_ = std::max(last_progress_, nearest.progress);
  const auto& destination = plan_.back().pose;
  const double distance_to_goal = distance(current.x, current.y,
                                           destination.position.x, destination.position.y);
  if (distance_to_goal <= xy_tolerance_.load()) position_reached_ = true;
  if (position_reached_) {
    const double yaw_error = wrapAngle(tf2::getYaw(destination.orientation) -
                                       tf2::getYaw(in_path.pose.orientation));
    if (std::abs(yaw_error) <= yaw_tolerance_.load()) {
      goal_reached_ = true;
      last_linear_speed_ = 0.0;
      last_angular_speed_ = 0.0;
      return true;
    }
  }
  const Point target = atDistance(std::min(lengths_.back(), nearest.progress + lookahead_));
  const double path_heading = std::atan2(target.y - current.y, target.x - current.x);
  const double heading_error = wrapAngle(path_heading - tf2::getYaw(in_path.pose.orientation));
  const bool rotate_only = position_reached_ || std::abs(heading_error) > turn_threshold_;
  const double target_error = position_reached_
      ? wrapAngle(tf2::getYaw(destination.orientation) - tf2::getYaw(in_path.pose.orientation))
      : heading_error;
  const double angular_target = std::clamp(2.5 * target_error,
                                           -max_angular_speed_, max_angular_speed_);
  const double max_angular_change = max_angular_accel_ * control_dt_;
  const double angular = std::clamp(angular_target,
                                    last_angular_speed_ - max_angular_change,
                                    last_angular_speed_ + max_angular_change);
  const double speed_limit = std::min(max_linear_speed_,
      std::max(0.08, std::min(distance_to_goal, lookahead_) / control_dt_));
  const double desired_speed = rotate_only ? 0.0 :
      speed_limit * std::max(0.3, std::cos(heading_error));
  const double linear = std::clamp(desired_speed,
                                   std::max(0.0, last_linear_speed_ - max_linear_accel_ * control_dt_),
                                   last_linear_speed_ + max_linear_accel_ * control_dt_);
  const double angular_candidates[] = {angular, angular * 0.5, 0.0};
  const double linear_candidates[] = {linear, linear * 0.5, 0.0};
  nav_msgs::Path trajectory;
  bool found = false;
  for (double speed : linear_candidates) {
    for (double turn : angular_candidates) {
      if (rotate_only && speed > 1e-6) continue;
      if (std::abs(speed) < 1e-6 && std::abs(turn) < 1e-4 && !position_reached_) continue;
      command.linear.x = speed;
      command.angular.z = turn;
      if (safeCommand(command, robot, nearest, &trajectory)) {
        found = true;
        break;
      }
    }
    if (found) break;
  }
  if (!found && !rotate_only) {
    const double escape_speed = std::min({linear * 0.5, 0.10, max_linear_speed_});
    const double steering_change = std::min(0.35, max_angular_change);
    for (double turn : {angular + steering_change, angular - steering_change}) {
      turn = std::clamp(turn, last_angular_speed_ - max_angular_change,
                        last_angular_speed_ + max_angular_change);
      turn = std::clamp(turn, -max_angular_speed_, max_angular_speed_);
      if (escape_speed < 0.02 || std::abs(turn - angular) < 0.08) continue;
      command.linear.x = escape_speed;
      command.angular.z = turn;
      if (safeCommand(command, robot, nearest, &trajectory)) {
        found = true;
        break;
      }
    }
  }
  if (!found && !position_reached_) {
    const double reverse_speed = std::max(-std::min(0.08, max_linear_speed_ * 0.5),
                                          last_linear_speed_ - max_linear_accel_ * control_dt_);
    if (reverse_speed < -0.02) {
      const double turn_change = std::min(max_angular_speed_, max_angular_change);
      for (double turn : {0.0, turn_change, -turn_change}) {
        turn = std::clamp(turn, last_angular_speed_ - max_angular_change,
                          last_angular_speed_ + max_angular_change);
        command.linear.x = reverse_speed;
        command.angular.z = turn;
        if (safeCommand(command, robot, nearest, &trajectory)) {
          found = true;
          break;
        }
      }
    }
  }
  if (!found) {
    command = geometry_msgs::Twist();
    ROS_WARN_THROTTLE(2.0, "PathFollower no collision-free path-following command");
    return false;
  }
  last_linear_speed_ = command.linear.x;
  last_angular_speed_ = command.angular.z;
  local_plan_pub_.publish(trajectory);
  legacy_local_plan_pub_.publish(trajectory);
  return true;
}

bool PathFollower::isGoalReached() {
  return goal_reached_;
}

}

PLUGINLIB_EXPORT_CLASS(nav_pkg::PathFollower, nav_core::BaseLocalPlanner)
