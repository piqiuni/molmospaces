#ifndef SCENE_SEMANTIC_MAPPER_H_
#define SCENE_SEMANTIC_MAPPER_H_

#include "semantic_mapping_pkg/scene_semantic_map.h"
#include "semantic_mapping_pkg/common/utils.h"
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/OccupancyGrid.h>
#include <ros/ros.h>
#include <string>
#include <limits>

/**
 * @class SceneSemanticMapper
 * @brief 场景语义建图算法
 * 
 * 负责：
 * 1. 从点云更新场景地图
 * 2. 遮挡检测（ray casting）
 * 3. 竖直角度过滤（支持指定角度范围）
 */
class SceneSemanticMapper
{
public:
    /**
     * @brief 构造函数
     * @param map 场景语义地图指针
     */
    explicit SceneSemanticMapper(SceneSemanticMap* map)
        : map_(map)
        , max_range_(10.0)
        , min_range_(0.1)
        , occlusion_threshold_(50)  // occupancy grid中>50视为障碍
        , has_occupancy_grid_(false)
    {
        if (!map_) {
            ROS_ERROR("SceneSemanticMapper: map pointer is null!");
        }
        ROS_INFO("SceneSemanticMapper initialized");
    }
    
    ~SceneSemanticMapper() = default;
    
    /**
     * @brief 从点云更新场景地图
     * @param pointcloud 点云数据（sensor_msgs/PointCloud2，相机坐标系）
     * @param camera_pose 相机在世界坐标系中的位姿
     * @param scene_attribute BLIP返回的场景属性字符串
     * @param confidence 置信度 [0.0, 1.0]，默认1.0
     * @param min_yaw_angle 最小偏航角（度），相对于相机前向，左为负，右为正。如果不需要角度过滤，设为NaN
     * @param max_yaw_angle 最大偏航角（度）。如果不需要角度过滤，设为NaN
     * 
     * 流程：
     * 1. 遍历点云中的每个点
     * 2. 距离过滤（min_range_ 到 max_range_）
     * 3. 偏航角过滤（如果提供了角度范围）
     * 4. 转换到世界坐标系
     * 5. 遮挡检测（如果设置了occupancy grid）
     * 6. 投影到地图平面（z=0）
     * 7. 更新场景属性
     * 
     * 注意：角度范围可能受BLIP输出影响，每次调用可以传入不同的角度范围
     */
    void updateFromPointCloud(const sensor_msgs::PointCloud2& pointcloud,
                             const geometry_msgs::Pose& camera_pose,
                             const std::string& scene_attribute,
                             float confidence = 1.0f,
                             double min_yaw_angle = std::numeric_limits<double>::quiet_NaN(),
                             double max_yaw_angle = std::numeric_limits<double>::quiet_NaN());
    
    /**
     * @brief 设置最大有效距离
     */
    void setMaxRange(double max_range) { max_range_ = max_range; }
    
    /**
     * @brief 设置最小有效距离
     */
    void setMinRange(double min_range) { min_range_ = min_range; }
    
    /**
     * @brief 设置遮挡检测阈值（occupancy grid值）
     */
    void setOcclusionThreshold(int8_t threshold) { occlusion_threshold_ = threshold; }
    
    /**
     * @brief 设置occupancy grid用于遮挡检测
     */
    void setOccupancyGrid(const nav_msgs::OccupancyGrid& occ_grid) {
        occupancy_grid_ = occ_grid;
        has_occupancy_grid_ = true;
    }

private:
    /**
     * @brief Ray casting遮挡检测
     * @param start 起点（世界坐标）
     * @param end 终点（世界坐标）
     * @return 如果射线被遮挡返回true
     */
    bool isOccluded(const geometry_msgs::Point& start,
                   const geometry_msgs::Point& end) const;
    
    /**
     * @brief 检查点是否在已知的自由空间
     * @param point 点（世界坐标）
     * @return 如果点在已知的自由空间（occupancy grid值为0）返回true，否则返回false
     */
    bool isKnownFreeSpace(const geometry_msgs::Point& point) const;
    
    /**
     * @brief 计算点的偏航角（yaw angle，度）
     * @param point 点（相机坐标系）
     * @return 角度（度），相对于相机前向（+Z），左为负，右为正
     * 
     * 相机坐标系：+X右, +Y下, +Z前
     * 偏航角 = atan2(x, z) * 180 / PI
     */
    static double computeYawAngle(const geometry_msgs::Point& point);
    
    SceneSemanticMap* map_;  // 场景语义地图
    
    // 参数
    double max_range_;           // 最大有效距离（米）
    double min_range_;           // 最小有效距离（米）
    int8_t occlusion_threshold_; // 遮挡检测阈值
    
    // 遮挡检测用的occupancy grid（可选）
    nav_msgs::OccupancyGrid occupancy_grid_;
    bool has_occupancy_grid_;
};

#endif // SCENE_SEMANTIC_MAPPER_H_

