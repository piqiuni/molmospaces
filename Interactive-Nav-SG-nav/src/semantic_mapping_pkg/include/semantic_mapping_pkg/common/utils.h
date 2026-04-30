#ifndef SEMANTIC_MAPPING_UTILS_H_
#define SEMANTIC_MAPPING_UTILS_H_

#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/ColorRGBA.h>
#include <string>
#include <map>
#include <functional>
#include <cmath>

namespace common
{

/**
 * @brief 计算两点之间的3D距离
 * @param p1 点1
 * @param p2 点2
 * @return 距离（米）
 */
inline double calculateDistance(const geometry_msgs::Point& p1,
                                const geometry_msgs::Point& p2)
{
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * @brief 计算两个尺寸的相对差异
 * @param size1 尺寸1
 * @param size2 尺寸2
 * @return 相对差异 [0, 1]，0表示完全相同，1表示完全不同
 */
inline double calculateSizeDifference(const geometry_msgs::Vector3& size1,
                                     const geometry_msgs::Vector3& size2)
{
    // 计算体积
    double vol1 = size1.x * size1.y * size1.z;
    double vol2 = size2.x * size2.y * size2.z;
    
    if (vol1 == 0 && vol2 == 0) {
        return 0.0;
    }
    if (vol1 == 0 || vol2 == 0) {
        return 1.0;  // 完全不同的尺寸
    }
    
    // 相对差异
    double diff = std::abs(vol1 - vol2) / std::max(vol1, vol2);
    return diff;
}

/**
 * @brief 将点从相机坐标系变换到世界坐标系
 * @param camera_pos 相机坐标系中的点
 * @param camera_pose 相机在世界坐标系中的位姿
 * @return 世界坐标系中的点
 */
geometry_msgs::Point transformToWorld(const geometry_msgs::Point& camera_pos,
                                     const geometry_msgs::Pose& camera_pose);

/**
 * @brief 将点从世界坐标系变换到相机坐标系
 * @param world_pos 世界坐标系中的点
 * @param camera_pose 相机在世界坐标系中的位姿
 * @return 相机坐标系中的点
 */
geometry_msgs::Point transformToCamera(const geometry_msgs::Point& world_pos,
                                      const geometry_msgs::Pose& camera_pose);

/**
 * @brief 根据场景类型获取颜色（硬编码映射）
 * @param normalized_scene_type 标准化后的场景类型字符串
 * @return ColorRGBA，如果不在预定义列表中返回 r=-1.0 表示无效
 */
std_msgs::ColorRGBA getColorForSceneType(const std::string& normalized_scene_type);

/**
 * @brief 根据场景ID获取颜色
 * @param scene_id 场景ID
 * @param scene_id_to_type 场景ID到类型的映射表
 * @param normalize_func 场景类型标准化函数（接受原始类型，返回标准化类型）
 * @return ColorRGBA
 */
std_msgs::ColorRGBA getColorForSceneId(int scene_id,
                                      const std::map<int, std::string>& scene_id_to_type,
                                      const std::function<std::string(const std::string&)>& normalize_func);

} // namespace common

#endif // SEMANTIC_MAPPING_UTILS_H_

