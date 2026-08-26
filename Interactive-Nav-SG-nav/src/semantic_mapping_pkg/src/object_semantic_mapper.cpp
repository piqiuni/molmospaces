#include "semantic_mapping_pkg/object_semantic_mapper.h"
#include <algorithm>
#include <limits>

void ObjectSemanticMapper::updateDetections(const std::vector<Detection>& detections,
                                           const geometry_msgs::Pose& camera_pose)
{
    // for debug
    // map_->clear();
    for (const auto& detection : detections) {
        updateDetection(detection, camera_pose);
        // if (!detection.semantic_class.empty())
        // {
        //     ROS_WARN("semantic class: %s", detection.semantic_class.c_str());
        // }
        // if (detection.semantic_class == "Painting") {
            
        //     ROS_WARN("painting id: %s, position: %f, %f, %f", detection.instance_id.c_str(), detection.position.x, detection.position.y, detection.position.z);
        // }

    }
}

void ObjectSemanticMapper::updateDetection(const Detection& detection,
                                          const geometry_msgs::Pose& camera_pose)
{
    if (!map_) {
        ROS_ERROR("ObjectSemanticMapper: map is null!");
        return;
    }
    
    // 检查置信度阈值
    if (detection.confidence < confidence_threshold_) {
        ROS_DEBUG("Detection confidence %.2f below threshold, skipping", detection.confidence);
        return;
    }
    
    // 1. 坐标变换：相机坐标系 -> 世界坐标系
    geometry_msgs::Point world_pos = common::transformToWorld(detection.position, camera_pose);
    
    // 2. 数据关联：查找匹配的物体
    uint64_t matched_object_id = associateDetection(detection, world_pos);
    
    if (matched_object_id == 0) {
        // 3. 创建新物体
        createNewObject(detection, world_pos);
    } else {
        // 4. 更新已有物体
        updateExistingObject(matched_object_id, detection, world_pos);
    }
}

uint64_t ObjectSemanticMapper::associateDetection(const Detection& detection,
                                                  const geometry_msgs::Point& world_pos)
{
    // 优先级1：实例ID匹配（如果可用）
    if (!detection.instance_id.empty()) {
        uint64_t id = matchByInstanceId(detection.instance_id);
        if (id > 0) {
            return id;
        }
    }
    
    // 优先级2：位置和语义匹配
    return matchByPositionAndSemantic(detection, world_pos);
}

uint64_t ObjectSemanticMapper::matchByInstanceId(const std::string& instance_id)
{
    auto it = instance_id_to_object_id_.find(instance_id);
    if (it != instance_id_to_object_id_.end()) {
        return it->second;
    }
    return 0;
}

uint64_t ObjectSemanticMapper::matchByPositionAndSemantic(const Detection& detection,
                                                         const geometry_msgs::Point& world_pos)
{
    // 获取附近的所有体素
    std::vector<SemanticVoxel*> candidates = map_->getVoxelsInRadius(
        world_pos.x, world_pos.y, world_pos.z, position_threshold_ * 2.0);
    
    if (candidates.empty()) {
        return 0;
    }
    
    ros::Time current_time = ros::Time::now();
    double min_distance = std::numeric_limits<double>::max();
    uint64_t best_match = 0;
    
    for (auto* voxel : candidates) {
        // 检查语义类别
        if (voxel->semantic_class != detection.semantic_class) {
            continue;
        }
        
        // 检查时间窗口
        double time_diff = (current_time - voxel->last_updated).toSec();
        if (time_diff > time_window_) {
            continue;  // 太久没更新，可能是旧物体
        }
        
        // 计算位置距离
        double distance = common::calculateDistance(world_pos, voxel->position);
        
        // 检查尺寸相似性
        double size_diff = common::calculateSizeDifference(detection.size, voxel->size);
        
        // 综合评分：距离越近、尺寸越相似，匹配度越高
        if (distance < position_threshold_ && 
            size_diff < size_threshold_ &&
            distance < min_distance) {
            min_distance = distance;
            best_match = voxel->object_id;
        }
    }
    
    return best_match;
}

void ObjectSemanticMapper::updateExistingObject(uint64_t object_id,
                                               const Detection& detection,
                                               const geometry_msgs::Point& world_pos)
{
    SemanticVoxel* voxel = map_->getVoxelByObjectId(object_id);
    if (!voxel) {
        ROS_WARN("Object %lu not found in map, creating new object", object_id);
        createNewObject(detection, world_pos);
        return;
    }
    
    // 贝叶斯更新位置（加权平均）
    double weight_old = static_cast<double>(voxel->observation_count) / 
                       (voxel->observation_count + 1.0);
    double weight_new = 1.0 / (voxel->observation_count + 1.0);
    
    voxel->position.x = weight_old * voxel->position.x + weight_new * world_pos.x;
    voxel->position.y = weight_old * voxel->position.y + weight_new * world_pos.y;
    voxel->position.z = weight_old * voxel->position.z + weight_new * world_pos.z;
    
    // 更新尺寸（加权平均）
    voxel->size.x = weight_old * voxel->size.x + weight_new * detection.size.x;
    voxel->size.y = weight_old * voxel->size.y + weight_new * detection.size.y;
    voxel->size.z = weight_old * voxel->size.z + weight_new * detection.size.z;
    
    // 更新置信度（多次观测增加置信度）
    voxel->confidence = std::min(1.0, voxel->confidence + 0.1 * detection.confidence);
    
    // 更新时间和计数
    voxel->last_updated = ros::Time::now();
    voxel->observation_count++;
    
    // 更新实例ID映射
    if (!detection.instance_id.empty()) {
        instance_id_to_object_id_[detection.instance_id] = object_id;
    }
    
    ROS_DEBUG("Updated object %lu: observations=%d, confidence=%.2f", 
              object_id, voxel->observation_count, voxel->confidence);
}

uint64_t ObjectSemanticMapper::createNewObject(const Detection& detection,
                                              const geometry_msgs::Point& world_pos)
{
    uint64_t new_id = next_object_id_++;
    
    SemanticVoxel voxel;
    voxel.object_id = new_id;
    voxel.position = world_pos;
    voxel.size = detection.size;
    voxel.semantic_class = detection.semantic_class;
    voxel.confidence = detection.confidence;
    voxel.first_observed = detection.timestamp;
    voxel.last_updated = detection.timestamp;
    voxel.observation_count = 1;
    // 存储到地图
    map_->setVoxel(world_pos.x, world_pos.y, world_pos.z, voxel);
    
    // 更新实例ID映射
    if (!detection.instance_id.empty()) {
        instance_id_to_object_id_[detection.instance_id] = new_id;
    }
    
    ROS_INFO("Created new object %lu: %s at (%.2f, %.2f, %.2f)", 
             new_id, detection.semantic_class.c_str(),
             world_pos.x, world_pos.y, world_pos.z);
    
    return new_id;
}
