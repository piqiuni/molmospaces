#include "semantic_mapping_pkg/object_semantic_map.h"
#include <algorithm>
#include <cmath>

VoxelIndex ObjectSemanticMap::worldToVoxel(double x, double y, double z) const
{
    return VoxelIndex(
        static_cast<int>(std::floor(x / voxel_size_)),
        static_cast<int>(std::floor(y / voxel_size_)),
        static_cast<int>(std::floor(z / voxel_size_))
    );
}

Eigen::Vector3d ObjectSemanticMap::voxelToWorld(const VoxelIndex& idx) const
{
    return Eigen::Vector3d(
        (idx.x + 0.5) * voxel_size_,
        (idx.y + 0.5) * voxel_size_,
        (idx.z + 0.5) * voxel_size_
    );
}

SemanticVoxel* ObjectSemanticMap::getVoxel(double x, double y, double z)
{
    VoxelIndex idx = worldToVoxel(x, y, z);
    auto it = voxel_map_.find(idx);
    return (it != voxel_map_.end()) ? &(it->second) : nullptr;
}

bool ObjectSemanticMap::hasVoxel(double x, double y, double z) const
{
    VoxelIndex idx = worldToVoxel(x, y, z);
    return voxel_map_.find(idx) != voxel_map_.end();
}

void ObjectSemanticMap::setVoxel(double x, double y, double z, const SemanticVoxel& voxel)
{
    VoxelIndex idx = worldToVoxel(x, y, z);
    voxel_map_[idx] = voxel;
    
    // 更新物体ID映射
    if (voxel.object_id > 0) {
        object_id_to_voxel_[voxel.object_id] = idx;
    }
}

SemanticVoxel* ObjectSemanticMap::updateVoxel(double x, double y, double z, const SemanticVoxel& voxel)
{
    VoxelIndex idx = worldToVoxel(x, y, z);
    auto& stored_voxel = voxel_map_[idx];
    
    // 如果体素已存在，更新；否则创建
    if (stored_voxel.object_id == 0) {
        stored_voxel = voxel;
        // 设置位置为体素中心
        Eigen::Vector3d world_pos = voxelToWorld(idx);
        stored_voxel.position.x = world_pos.x();
        stored_voxel.position.y = world_pos.y();
        stored_voxel.position.z = world_pos.z();
    } else {
        // 更新已有体素
        stored_voxel = voxel;
        Eigen::Vector3d world_pos = voxelToWorld(idx);
        stored_voxel.position.x = world_pos.x();
        stored_voxel.position.y = world_pos.y();
        stored_voxel.position.z = world_pos.z();
    }
    
    // 更新物体ID映射
    if (voxel.object_id > 0) {
        object_id_to_voxel_[voxel.object_id] = idx;
    }
    
    return &stored_voxel;
}

std::vector<SemanticVoxel*> ObjectSemanticMap::getVoxelsInRadius(double x, double y, double z, double radius)
{
    std::vector<SemanticVoxel*> result;
    
    // 计算搜索范围（体素单位）
    int radius_voxels = static_cast<int>(std::ceil(radius / voxel_size_));
    VoxelIndex center_idx = worldToVoxel(x, y, z);
    
    for (int dx = -radius_voxels; dx <= radius_voxels; dx++) {
        for (int dy = -radius_voxels; dy <= radius_voxels; dy++) {
            for (int dz = -radius_voxels; dz <= radius_voxels; dz++) {
                VoxelIndex idx(center_idx.x + dx, center_idx.y + dy, center_idx.z + dz);
                
                auto it = voxel_map_.find(idx);
                if (it != voxel_map_.end()) {
                    // 检查实际距离
                    Eigen::Vector3d voxel_pos = voxelToWorld(idx);
                    double distance = std::sqrt(
                        std::pow(voxel_pos.x() - x, 2) +
                        std::pow(voxel_pos.y() - y, 2) +
                        std::pow(voxel_pos.z() - z, 2)
                    );
                    
                    if (distance <= radius) {
                        result.push_back(&(it->second));
                    }
                }
            }
        }
    }
    
    return result;
}

std::vector<SemanticVoxel*> ObjectSemanticMap::getVoxelsByClass(const std::string& semantic_class)
{
    std::vector<SemanticVoxel*> result;
    
    for (auto& [idx, voxel] : voxel_map_) {
        if (voxel.semantic_class == semantic_class) {
            result.push_back(&voxel);
        }
    }
    
    return result;
}

SemanticVoxel* ObjectSemanticMap::getVoxelByObjectId(uint64_t object_id)
{
    auto it = object_id_to_voxel_.find(object_id);
    if (it != object_id_to_voxel_.end()) {
        auto voxel_it = voxel_map_.find(it->second);
        if (voxel_it != voxel_map_.end()) {
            return &(voxel_it->second);
        }
    }
    return nullptr;
}

std::vector<SemanticVoxel*> ObjectSemanticMap::getAllVoxels()
{
    std::vector<SemanticVoxel*> result;
    result.reserve(voxel_map_.size());
    
    for (auto& [idx, voxel] : voxel_map_) {
        result.push_back(&voxel);
    }
    
    return result;
}

void ObjectSemanticMap::clear()
{
    voxel_map_.clear();
    object_id_to_voxel_.clear();
}

visualization_msgs::MarkerArray ObjectSemanticMap::toMarkerArray(const std::string& frame_id)
{
    visualization_msgs::MarkerArray marker_array;
    
    int marker_id = 0;
    
    for (const auto& [idx, voxel] : voxel_map_) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "semantic_objects";
        marker.id = marker_id++;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        
        marker.pose.position = voxel.position;
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = std::max(voxel.size.x, voxel_size_);
        marker.scale.y = std::max(voxel.size.y, voxel_size_);
        marker.scale.z = std::max(voxel.size.z, voxel_size_);
        
        // 根据语义类别设置颜色
        marker.color.r = 0.5;
        marker.color.g = 0.5;
        marker.color.b = 0.5;
        marker.color.a = voxel.confidence;
        
        marker.lifetime = ros::Duration(0);
        
        marker_array.markers.push_back(marker);
        
        // 添加文本标签
        visualization_msgs::Marker text_marker;
        text_marker.header = marker.header;
        text_marker.ns = "semantic_labels";
        text_marker.id = marker_id++;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        text_marker.pose.position = voxel.position;
        text_marker.pose.position.z += voxel.size.z / 2.0 + 0.1;
        text_marker.pose.orientation.w = 1.0;
        text_marker.scale.z = 0.2;
        text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        text_marker.text = voxel.semantic_class + " (" + std::to_string(voxel.object_id) + ")";
        text_marker.lifetime = ros::Duration(0);
        
        marker_array.markers.push_back(text_marker);
    }
    
    return marker_array;
}

