#include "semantic_mapping_pkg/common/utils.h"
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <algorithm>
#include <cmath>

namespace common
{

geometry_msgs::Point transformToWorld(const geometry_msgs::Point& camera_pos,
                                     const geometry_msgs::Pose& camera_pose)
{
    // 创建绕Z轴旋转90度的四元数
    tf::Quaternion q_z_90 = tf::createQuaternionFromRPY(0, 0, M_PI / 2);
    
    // 将输入点转换为TF向量
    tf::Vector3 camera_point(camera_pos.x, camera_pos.y, camera_pos.z);
    
    // 应用旋转（绕原点旋转）
    tf::Vector3 rotated_point = tf::quatRotate(q_z_90, camera_point);
    
    // 转换回geometry_msgs::Point
    geometry_msgs::Point result;
    result.x = rotated_point.x();
    result.y = rotated_point.y();
    result.z = rotated_point.z();
    
    return result;

    // // 提取旋转和平移
    // tf::Quaternion q(
    //     camera_pose.orientation.x,
    //     camera_pose.orientation.y,
    //     camera_pose.orientation.z,
    //     camera_pose.orientation.w
    // );
    // tf::Vector3 t(
    //     camera_pose.position.x,
    //     camera_pose.position.y,
    //     camera_pose.position.z
    // );

    // // tf::Quaternion q_z_90 = tf::createQuaternionFromRPY(0, 0, M_PI / 2);
    // // tf::Quaternion q_final = q_z_90 * q;

    // // 构建变换矩阵
    // tf::Transform transform(q, t);
    
    // // 变换点
    // tf::Vector3 camera_point(camera_pos.x, camera_pos.y, camera_pos.z);
    // tf::Vector3 world_point = transform * camera_point;
    
    // geometry_msgs::Point result;
    // result.x = world_point.x();
    // result.y = world_point.y(); 
    // result.z = world_point.z();
    
    // return result;
}

geometry_msgs::Point transformToCamera(const geometry_msgs::Point& world_pos,
                                       const geometry_msgs::Pose& camera_pose)
{
    return world_pos;

    // 提取旋转和平移
    tf::Quaternion q(
        camera_pose.orientation.x,
        camera_pose.orientation.y,
        camera_pose.orientation.z,
        camera_pose.orientation.w
    );
    tf::Vector3 t(
        camera_pose.position.x,
        camera_pose.position.y,
        camera_pose.position.z
    );
    
    // 构建变换矩阵（世界到相机需要逆变换）
    tf::Transform transform(q, t);
    tf::Transform inverse_transform = transform.inverse();
    
    // 变换点（注意：输入的world_pos的Y已经是修正过的，需要先反转回去）
    tf::Vector3 world_point(world_pos.x, world_pos.y, world_pos.z);  // Y轴取反，与transformToWorld对应
    tf::Vector3 camera_point = inverse_transform * world_point;
    
    geometry_msgs::Point result;
    result.x = camera_point.x();
    result.y = camera_point.y();
    result.z = camera_point.z();

    
    return result;
}

std_msgs::ColorRGBA getColorForSceneType(const std::string& normalized_scene_type)
{
    std_msgs::ColorRGBA color;
    color.a = 1.0;
    
    // 常见场景类型的固定颜色映射
    if (normalized_scene_type == "bathroom") {
        color.r = 0.0; color.g = 0.5; color.b = 1.0;  // 蓝色
        return color;
    }
    if (normalized_scene_type == "corridor") {
        color.r = 1.0; color.g = 1.0; color.b = 0.0;  // 黄色
        return color;
    }
    if (normalized_scene_type == "kitchen") {
        color.r = 1.0; color.g = 0.5; color.b = 0.0;  // 橙色
        return color;
    }
    if (normalized_scene_type == "bedroom") {
        color.r = 0.5; color.g = 0.0; color.b = 1.0;  // 紫色
        return color;
    }
    if (normalized_scene_type == "livingroom") {
        color.r = 0.0; color.g = 1.0; color.b = 0.5;  // 绿色
        return color;
    }
    if (normalized_scene_type == "dinningroom") {
        color.r = 1.0; color.g = 0.8; color.b = 0.0;  // 金黄色
        return color;
    }
    if (normalized_scene_type == "laundry") {
        color.r = 0.8; color.g = 0.5; color.b = 1.0;  // 浅紫色
        return color;
    }
    if (normalized_scene_type == "storage") {
        color.r = 0.6; color.g = 0.4; color.b = 0.2;  // 棕色
        return color;
    }
    if (normalized_scene_type == "garden") {
        color.r = 0.0; color.g = 0.8; color.b = 0.0;  // 深绿色
        return color;
    }
    if (normalized_scene_type == "parkinglot") {
        color.r = 0.3; color.g = 0.3; color.b = 0.3;  // 深灰色
        return color;
    }
    
    // 如果不在预定义列表中，返回空颜色（将由scene_id生成）
    color.r = -1.0; color.g = -1.0; color.b = -1.0;
    return color;
}

std_msgs::ColorRGBA getColorForSceneId(int scene_id,
                                      const std::map<int, std::string>& scene_id_to_type,
                                      const std::function<std::string(const std::string&)>& normalize_func)
{
    std_msgs::ColorRGBA color;
    color.a = 1.0;
    
    // 特殊ID使用固定颜色（SCENE_UNKNOWN = -1）
    if (scene_id == -1) {
        color.r = 0.5; color.g = 0.5; color.b = 0.5;  // 灰色 - 未知
        return color;
    }
    
    // 查找场景类型名称
    auto it = scene_id_to_type.find(scene_id);
    if (it != scene_id_to_type.end()) {
        // 标准化场景类型
        std::string normalized = normalize_func(it->second);
        // 尝试使用场景类型的固定颜色
        std_msgs::ColorRGBA type_color = getColorForSceneType(normalized);
        if (type_color.r >= 0.0) {  // 如果找到了预定义颜色
            return type_color;
        }
    }
    
    // 如果场景类型没有预定义颜色，使用HSV颜色空间根据scene_id生成颜色
    float hue = static_cast<float>((scene_id - 3) % 120) / 120.0f;  // 0-1范围（对应0-120度）
    float saturation = 0.8f;
    float value = 0.9f;
    
    // HSV to RGB转换
    float c = value * saturation;
    float x = c * (1.0f - std::abs(std::fmod(hue * 6.0f, 2.0f) - 1.0f));
    float m = value - c;
    
    float r, g, b;
    if (hue < 1.0f/6.0f) {
        r = c; g = x; b = 0.0f;
    } else if (hue < 2.0f/6.0f) {
        r = x; g = c; b = 0.0f;
    } else if (hue < 3.0f/6.0f) {
        r = 0.0f; g = c; b = x;
    } else if (hue < 4.0f/6.0f) {
        r = 0.0f; g = x; b = c;
    } else if (hue < 5.0f/6.0f) {
        r = x; g = 0.0f; b = c;
    } else {
        r = c; g = 0.0f; b = x;
    }
    
    color.r = r + m;
    color.g = g + m;
    color.b = b + m;
    return color;
}

} // namespace common

