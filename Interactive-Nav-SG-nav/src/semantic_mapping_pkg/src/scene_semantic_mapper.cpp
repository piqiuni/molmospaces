#include "semantic_mapping_pkg/scene_semantic_mapper.h"
#include "semantic_mapping_pkg/common/utils.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <algorithm>
#include <cmath>

double SceneSemanticMapper::computeYawAngle(const geometry_msgs::Point& point)
{
    // 计算偏航角（yaw angle）
    // 相机坐标系：+X右, +Y下, +Z前
    // 偏航角 = atan2(x, z) * 180 / PI
    // 相对于相机前向（+Z），左为负，右为正
    
    if (std::abs(point.z) < 1e-6 && std::abs(point.x) < 1e-6) {
        // 如果点在相机正前方（z=0, x=0），返回0度
        return 0.0;
    }
    
    double angle_rad = std::atan2(point.x, point.z);
    double angle_deg = angle_rad * 180.0 / M_PI;
    
    return angle_deg;
}

bool SceneSemanticMapper::isOccluded(const geometry_msgs::Point& start,
                                     const geometry_msgs::Point& end) const
{
    if (!has_occupancy_grid_) {
        return false;  // 如果没有occupancy grid，不进行遮挡检测
    }
    
    // 计算射线方向
    double dx = end.x - start.x;
    double dy = end.y - start.y;
    double dist = std::sqrt(dx * dx + dy * dy);
    
    if (dist < 0.01) {
        return false;  // 距离太近，不检查
    }
    
    // 归一化方向
    dx /= dist;
    dy /= dist;
    
    // 获取occupancy grid参数
    const auto& info = occupancy_grid_.info;
    double resolution = info.resolution;
    double origin_x = info.origin.position.x;
    double origin_y = info.origin.position.y;
    
    // Ray casting：沿着射线采样
    double step = resolution * 0.5;  // 步长为分辨率的一半
    int steps = static_cast<int>(dist / step) + 1;
    
    for (int i = 1; i < steps; ++i) {  // 从起点后开始检查
        double t = i * step;
        double x = start.x + dx * t;
        double y = start.y + dy * t;
        
        // 转换到grid坐标
        int grid_x = static_cast<int>((x - origin_x) / resolution);
        int grid_y = static_cast<int>((y - origin_y) / resolution);
        
        // 检查是否在范围内
        if (grid_x >= 0 && grid_x < static_cast<int>(info.width) &&
            grid_y >= 0 && grid_y < static_cast<int>(info.height)) {
            
            size_t idx = grid_y * info.width + grid_x;
            int8_t occ_value = occupancy_grid_.data[idx];
            
            // 如果遇到障碍物，返回被遮挡
            if (occ_value > occlusion_threshold_) {
                return true;
            }
        }
    }
    
    return false;  // 未被遮挡
}

bool SceneSemanticMapper::isKnownFreeSpace(const geometry_msgs::Point& point) const
{
    if (!has_occupancy_grid_) {
        return true;  // 如果没有occupancy grid，允许更新（向后兼容）
    }
    
    // 获取occupancy grid参数
    const auto& info = occupancy_grid_.info;
    double resolution = info.resolution;
    double origin_x = info.origin.position.x;
    double origin_y = info.origin.position.y;
    
    // 转换到grid坐标
    int grid_x = static_cast<int>((point.x - origin_x) / resolution);
    int grid_y = static_cast<int>((point.y - origin_y) / resolution);
    
    // 检查是否在范围内
    if (grid_x < 0 || grid_x >= static_cast<int>(info.width) ||
        grid_y < 0 || grid_y >= static_cast<int>(info.height)) {
        return false;  // 超出地图范围
    }
    
    size_t idx = grid_y * info.width + grid_x;
    int8_t occ_value = occupancy_grid_.data[idx];
    
    // 只有值为0（自由空间）才认为是已知的自由空间
    // -1表示未知，>0表示占用或障碍物
    return (occ_value == 0);
}

void SceneSemanticMapper::updateFromPointCloud(const sensor_msgs::PointCloud2& pointcloud,
                                              const geometry_msgs::Pose& camera_pose,
                                              const std::string& scene_attribute,
                                              float confidence,
                                              double min_yaw_angle,
                                              double max_yaw_angle)
{
    if (!map_) {
        ROS_ERROR("SceneSemanticMapper::updateFromPointCloud: map is null!");
        return;
    }
    
    if (scene_attribute.empty()) {
        ROS_WARN("SceneSemanticMapper::updateFromPointCloud: empty scene attribute, skipping");
        return;
    }
    
    // 获取或创建scene_id
    int scene_id = map_->getOrCreateSceneId(scene_attribute);
    if (scene_id < 0) {
        ROS_ERROR("SceneSemanticMapper::updateFromPointCloud: failed to get scene_id for '%s'", 
                  scene_attribute.c_str());
        return;
    }
    
    // 转换点云格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(pointcloud, *cloud);
    
    if (cloud->empty()) {
        ROS_WARN("SceneSemanticMapper::updateFromPointCloud: empty pointcloud");
        return;
    }
    
    // 获取相机位置（用于遮挡检测）
    geometry_msgs::Point camera_pos = camera_pose.position;
    
    // 检查是否启用偏航角过滤
    bool use_yaw_filter = std::isfinite(min_yaw_angle) && std::isfinite(max_yaw_angle);
    
    // 统计更新的点数量
    int updated_count = 0;
    int skipped_count = 0;
    int occluded_count = 0;
    int yaw_filtered_count = 0;
    int unknown_space_count = 0;
    
    // 遍历点云中的每个点
    for (const auto& point : cloud->points) {
        // 检查点是否有效（不是NaN）
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
            skipped_count++;
            continue;
        }
        
        // 计算距离（相机坐标系下的距离）
        double distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        if (distance > max_range_ || distance < min_range_) {
            skipped_count++;
            continue;
        }
        
        // 偏航角过滤（如果提供了角度范围）
        if (use_yaw_filter) {
            geometry_msgs::Point camera_point;
            camera_point.x = point.x;
            camera_point.y = point.y;
            camera_point.z = point.z;
            
            double yaw_angle = computeYawAngle(camera_point);
            
            // 检查角度是否在范围内
            if (yaw_angle < min_yaw_angle || yaw_angle > max_yaw_angle) {
                yaw_filtered_count++;
                continue;
            }
        }
        
        // 转换到世界坐标系
        geometry_msgs::Point camera_point;
        camera_point.x = point.x;
        camera_point.y = point.y;
        camera_point.z = point.z;
        
        geometry_msgs::Point world_point = common::transformToWorld(camera_point, camera_pose);
        
        // 投影到地图平面（z=0，不考虑高度）
        double map_x = world_point.x;
        double map_y = world_point.y;
        
        // 检查坐标有效性
        if (!map_->isValidCoordinate(map_x, map_y)) {
            skipped_count++;
            continue;
        }
        
        // 遮挡检测（如果设置了occupancy grid）
        if (has_occupancy_grid_) {
            geometry_msgs::Point map_point;
            map_point.x = map_x;
            map_point.y = map_y;
            map_point.z = 0.0;
            
            if (isOccluded(camera_pos, map_point)) {
                occluded_count++;
                continue;
            }
            
            // 检查目标点是否在已知的自由空间
            if (!isKnownFreeSpace(map_point)) {
                unknown_space_count++;
                continue;
            }
        }
        
        // 更新地图网格
        map_->setSceneLabel(map_x, map_y, scene_id, scene_attribute, confidence);
        updated_count++;
    }
    
    ROS_DEBUG("SceneSemanticMapper::updateFromPointCloud: scene='%s', updated=%d cells, "
              "skipped=%d points, occluded=%d, yaw_filtered=%d, unknown_space=%d, yaw_range=[%.1f, %.1f]",
              scene_attribute.c_str(), updated_count, skipped_count, occluded_count, yaw_filtered_count, unknown_space_count,
              use_yaw_filter ? min_yaw_angle : 0.0, use_yaw_filter ? max_yaw_angle : 0.0);
}
