#ifndef OBJECT_SEMANTIC_MAPPER_H_
#define OBJECT_SEMANTIC_MAPPER_H_

#include "semantic_mapping_pkg/object_semantic_map.h"
#include "semantic_mapping_pkg/common/utils.h"
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <ros/ros.h>
#include <vector>
#include <string>
#include <map>

/**
 * @brief 检测结果结构
 */
struct Detection {
    geometry_msgs::Point position;      // 3D位置（相机坐标系）
    geometry_msgs::Vector3 size;        // 物体尺寸
    std::string semantic_class;         // 语义类别
    double confidence;                  // 置信度 [0, 1]
    std::string instance_id;            // 实例ID（如果检测器提供）
    ros::Time timestamp;                // 时间戳
    
    Detection() : confidence(0.0) {
        position.x = position.y = position.z = 0.0;
        size.x = size.y = size.z = 0.0;
        timestamp = ros::Time::now();
    }
};

/**
 * @class ObjectSemanticMapper
 * @brief 物体语义建图算法
 * 
 * 负责：
 * 1. 数据关联（将检测结果关联到地图中的物体）
 * 2. 坐标变换（相机坐标系 -> 世界坐标系）
 * 3. 地图更新（贝叶斯更新、融合等）
 */
class ObjectSemanticMapper
{
public:
    /**
     * @brief 构造函数
     * @param map 物体语义地图指针
     */
    explicit ObjectSemanticMapper(ObjectSemanticMap* map)
        : map_(map)
        , position_threshold_(0.5)      // 0.5米
        , size_threshold_(0.3)          // 30%差异
        , time_window_(5.0)             // 5秒
        , confidence_threshold_(0.5)    // 0.5
        , next_object_id_(1)
    {
        if (!map_) {
            ROS_ERROR("ObjectSemanticMapper: map pointer is null!");
        }
        ROS_INFO("ObjectSemanticMapper initialized");
    }
    
    ~ObjectSemanticMapper() = default;
    
    /**
     * @brief 更新检测结果到地图
     * @param detections 检测结果列表
     * @param camera_pose 相机在世界坐标系中的位姿
     */
    void updateDetections(const std::vector<Detection>& detections,
                          const geometry_msgs::Pose& camera_pose);
    
    // ========== 参数设置 ==========
    
    /**
     * @brief 设置位置匹配阈值（米）
     */
    void setPositionThreshold(double threshold) { position_threshold_ = threshold; }
    
    /**
     * @brief 设置尺寸匹配阈值
     */
    void setSizeThreshold(double threshold) { size_threshold_ = threshold; }
    
    /**
     * @brief 设置时间窗口（秒）
     */
    void setTimeWindow(double window) { time_window_ = window; }
    
    /**
     * @brief 设置置信度阈值
     */
    void setConfidenceThreshold(double threshold) { confidence_threshold_ = threshold; }

private:
    // ========== 数据关联 ==========
    
    /**
     * @brief 关联检测结果到地图中的物体
     * @param detection 检测结果（已转换到世界坐标系）
     * @param world_pos 世界坐标位置
     * @return 匹配的物体ID，如果没有匹配返回0
     */
    uint64_t associateDetection(const Detection& detection,
                                const geometry_msgs::Point& world_pos);
    
    /**
     * @brief 基于位置和语义的匹配
     */
    uint64_t matchByPositionAndSemantic(const Detection& detection,
                                       const geometry_msgs::Point& world_pos);
    
    /**
     * @brief 基于实例ID的匹配（如果检测器提供）
     */
    uint64_t matchByInstanceId(const std::string& instance_id);
    
    // ========== 地图更新 ==========
    
    /**
     * @brief 更新已有物体
     * @param object_id 物体ID
     * @param detection 检测结果
     * @param world_pos 世界坐标位置
     */
    void updateExistingObject(uint64_t object_id,
                              const Detection& detection,
                              const geometry_msgs::Point& world_pos);
    
    /**
     * @brief 创建新物体
     * @param detection 检测结果
     * @param world_pos 世界坐标位置
     * @return 新创建的物体ID
     */
    uint64_t createNewObject(const Detection& detection,
                             const geometry_msgs::Point& world_pos);
    
    /**
     * @brief 更新单个检测结果（内部使用）
     */
    void updateDetection(const Detection& detection,
                        const geometry_msgs::Pose& camera_pose);
    
    // ========== 成员变量 ==========
    
    ObjectSemanticMap* map_;  // 物体语义地图
    
    // 参数
    double position_threshold_;   // 位置匹配阈值（米），默认0.5m
    double size_threshold_;       // 尺寸差异阈值，默认0.3
    double time_window_;          // 时间窗口（秒），默认5.0s
    double confidence_threshold_; // 置信度阈值，默认0.5
    
    // 物体ID管理
    uint64_t next_object_id_;  // 下一个可用的物体ID
    std::map<std::string, uint64_t> instance_id_to_object_id_;  // 实例ID到物体ID的映射
};

#endif // OBJECT_SEMANTIC_MAPPER_H_

