#include "semantic_mapping_pkg/scene_semantic_map.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <cctype>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/MarkerArray.h>
#include <ros/ros.h>
#include <xmlrpcpp/XmlRpcValue.h>

// 静态常量定义（满足ODR要求）
const int8_t SceneSemanticMap::CONFIDENCE_UNKNOWN;
const int8_t SceneSemanticMap::SCENE_UNKNOWN;

SceneSemanticMap::SceneSemanticMap(double resolution, 
                   unsigned int width, 
                   unsigned int height,
                   double origin_x,
                   double origin_y)
    : resolution_(resolution)
    , width_(width)
    , height_(height)
    , origin_x_(origin_x)
    , origin_y_(origin_y)
    , scene_type_mapping_initialized_(false)
{
    // 初始化scene_id_grid_
    scene_id_grid_.info.resolution = resolution_;
    scene_id_grid_.info.width = width_;
    scene_id_grid_.info.height = height_;
    scene_id_grid_.info.origin.position.x = origin_x_;
    scene_id_grid_.info.origin.position.y = origin_y_;
    scene_id_grid_.info.origin.position.z = 0.0;
    scene_id_grid_.info.origin.orientation.w = 1.0;
    scene_id_grid_.data.resize(width_ * height_, SCENE_UNKNOWN);
    
    // 初始化confidence_grid_（与scene_id_grid_相同的尺寸）
    confidence_grid_ = scene_id_grid_;
    confidence_grid_.data.resize(width_ * height_, CONFIDENCE_UNKNOWN);
    
    ROS_INFO("SceneSemanticMap initialized: resolution=%.3f, size=%dx%d, origin=(%.2f, %.2f)", 
             resolution_, width_, height_, origin_x_, origin_y_);
    ROS_INFO("Scene ID mapping: -1=unknown(未观测), 0-9=semantic_class_index(直接对应通道4-13)");
    ROS_INFO("Semantic indices: 0=livingroom, 1=bedroom, 2=kitchen, 3=bathroom, 4=balcony, "
             "5=storage, 6=door, 7=wall, 8=entrance, 9=outside");
}

SceneSemanticMap::~SceneSemanticMap()
{
}

void SceneSemanticMap::initializeFromOccupancyGrid(const nav_msgs::OccupancyGrid& occ_grid)
{
    resolution_ = occ_grid.info.resolution;
    width_ = occ_grid.info.width;
    height_ = occ_grid.info.height;
    origin_x_ = occ_grid.info.origin.position.x;
    origin_y_ = occ_grid.info.origin.position.y;
    
    // 初始化scene_id_grid_
    scene_id_grid_.info = occ_grid.info;
    scene_id_grid_.data.resize(width_ * height_, SCENE_UNKNOWN);
    
    // 初始化confidence_grid_
    confidence_grid_.info = occ_grid.info;
    confidence_grid_.data.resize(width_ * height_, CONFIDENCE_UNKNOWN);
    
    
    ROS_INFO("SceneSemanticMap initialized from OccupancyGrid: resolution=%.3f, size=%dx%d", 
             resolution_, width_, height_);
}

bool SceneSemanticMap::worldToGrid(double x, double y, int& grid_x, int& grid_y) const
{
    grid_x = static_cast<int>((x - origin_x_) / resolution_);
    grid_y = static_cast<int>((y - origin_y_) / resolution_);
    return isValidGrid(grid_x, grid_y);
}

void SceneSemanticMap::gridToWorld(int grid_x, int grid_y, double& x, double& y) const
{
    x = origin_x_ + (grid_x + 0.5) * resolution_;
    y = origin_y_ + (grid_y + 0.5) * resolution_;
}

bool SceneSemanticMap::isValidGrid(int grid_x, int grid_y) const
{
    return (grid_x >= 0 && grid_x < static_cast<int>(width_) &&
            grid_y >= 0 && grid_y < static_cast<int>(height_));
}

bool SceneSemanticMap::isValidCoordinate(double x, double y) const
{
    int grid_x, grid_y;
    return worldToGrid(x, y, grid_x, grid_y);
}

bool SceneSemanticMap::getKnownRegionBounds(double& min_x, double& min_y, double& max_x, double& max_y) const
{
    bool found_known = false;
    int min_grid_x = static_cast<int>(width_), min_grid_y = static_cast<int>(height_);
    int max_grid_x = -1, max_grid_y = -1;
    
    // 遍历所有栅格cell，找到已知区域的边界
    for (unsigned int y = 0; y < height_; ++y) {
        for (unsigned int x = 0; x < width_; ++x) {
            size_t idx = gridToIndex(x, y);
            int scene_id = static_cast<int>(scene_id_grid_.data[idx]);
            
            // 如果cell不是未知状态，记录其位置
            if (scene_id != SCENE_UNKNOWN) {
                found_known = true;
                int grid_x = static_cast<int>(x);
                int grid_y = static_cast<int>(y);
                if (grid_x < min_grid_x) min_grid_x = grid_x;
                if (grid_x > max_grid_x) max_grid_x = grid_x;
                if (grid_y < min_grid_y) min_grid_y = grid_y;
                if (grid_y > max_grid_y) max_grid_y = grid_y;
            }
        }
    }
    
    if (!found_known) {
        return false;
    }
    
    // 将栅格坐标转换为世界坐标
    gridToWorld(min_grid_x, min_grid_y, min_x, min_y);
    gridToWorld(max_grid_x, max_grid_y, max_x, max_y);
    
    return true;
}

size_t SceneSemanticMap::gridToIndex(int grid_x, int grid_y) const
{
    return static_cast<size_t>(grid_y * width_ + grid_x);
}

void SceneSemanticMap::setSceneLabel(double x, double y, int scene_id, 
                             const std::string& scene_type, 
                             float confidence)
{
    int grid_x, grid_y;
    if (!worldToGrid(x, y, grid_x, grid_y)) {
        ROS_WARN("SceneSemanticMap::setSceneLabel: coordinate (%.2f, %.2f) out of bounds", x, y);
        return;
    }
    
    // 检查scene_id范围
    // 允许值: -1(未知), 0-9(语义类别索引，直接对应通道4-13)
    if (scene_id < -1 || scene_id > 9) {
        ROS_WARN("SceneSemanticMap::setSceneLabel: scene_id %d out of range [-1, 9], clamping", scene_id);
        scene_id = std::max(-1, std::min(9, scene_id));
    }
    
    size_t idx = gridToIndex(grid_x, grid_y);
    
    // 获取现有的scene_id和confidence
    int8_t old_scene_id = scene_id_grid_.data[idx];
    int8_t old_confidence = confidence_grid_.data[idx];
    
    // 置信度更新策略：基于观测一致性
    // BLIP模型不支持confidence，因此使用累积更新机制
    int8_t new_confidence;
    
    // 判断是否为首次观测：confidence未知 或 scene_id从未知变为有效值
    bool is_first_observation = (old_confidence == CONFIDENCE_UNKNOWN) || 
                                (old_scene_id == SCENE_UNKNOWN && scene_id != SCENE_UNKNOWN);
    
    if (is_first_observation) {
        // 首次观测：设置初始置信度（1对应0.01，表示非常保守的初始置信度）
        new_confidence = 1;
    } else {
        // 检查新观测与现有观测是否一致
        if (old_scene_id == static_cast<int8_t>(scene_id)) {
            // 相同观测：置信度+1（上限100，对应1.0）
            new_confidence = std::min(100, static_cast<int>(old_confidence) + 1);
        } else {
            // 不同观测：置信度-1（下限0，对应0.0）
            new_confidence = std::max(0, static_cast<int>(old_confidence) - 1);
        }
    }
    
    // 设置scene_id
    scene_id_grid_.data[idx] = static_cast<int8_t>(scene_id);
    
    // 设置confidence
    confidence_grid_.data[idx] = static_cast<int8_t>(new_confidence);
    
    // 更新场景类型映射表（可选，用于记录）
    if (!scene_type.empty() && scene_id >= 0) {
        scene_id_to_type_[scene_id] = scene_type;
    }
}

int SceneSemanticMap::getSceneId(double x, double y) const
{
    int grid_x, grid_y;
    if (!worldToGrid(x, y, grid_x, grid_y)) {
        return SCENE_UNKNOWN;
    }
    
    size_t idx = gridToIndex(grid_x, grid_y);
    return static_cast<int>(scene_id_grid_.data[idx]);
}

float SceneSemanticMap::getConfidence(double x, double y) const
{
    int grid_x, grid_y;
    if (!worldToGrid(x, y, grid_x, grid_y)) {
        return -1.0f;
    }
    
    size_t idx = gridToIndex(grid_x, grid_y);
    int8_t conf_val = confidence_grid_.data[idx];
    
    if (conf_val == CONFIDENCE_UNKNOWN) {
        return -1.0f;
    }
    
    return static_cast<float>(conf_val) / 100.0f;
}

std::string SceneSemanticMap::getSceneType(double x, double y) const
{
    int scene_id = getSceneId(x, y);
    if (scene_id == SCENE_UNKNOWN) {
        return "unknown";
    }
    
    auto it = scene_id_to_type_.find(scene_id);
    if (it != scene_id_to_type_.end()) {
        return it->second;
    }
    
    return "unknown";
}

void SceneSemanticMap::updateSceneLabels(const std::vector<double>& world_x,
                                 const std::vector<double>& world_y,
                                 const std::vector<int>& scene_ids,
                                 const std::vector<std::string>& scene_types,
                                 const std::vector<float>& confidences)
{
    if (world_x.size() != world_y.size() || world_x.size() != scene_ids.size()) {
        ROS_ERROR("SceneSemanticMap::updateSceneLabels: input vectors size mismatch");
        return;
    }
    
    bool has_types = !scene_types.empty() && scene_types.size() == scene_ids.size();
    bool has_confidences = !confidences.empty() && confidences.size() == scene_ids.size();
    
    for (size_t i = 0; i < world_x.size(); ++i) {
        std::string scene_type = has_types ? scene_types[i] : "";
        float confidence = has_confidences ? confidences[i] : 1.0f;
        setSceneLabel(world_x[i], world_y[i], scene_ids[i], scene_type, confidence);
    }
}

void SceneSemanticMap::registerSceneType(int scene_id, const std::string& scene_type)
{
    if (scene_id > 0) {
        scene_id_to_type_[scene_id] = scene_type;
    }
}

std::string SceneSemanticMap::getSceneTypeById(int scene_id) const
{
    auto it = scene_id_to_type_.find(scene_id);
    if (it != scene_id_to_type_.end()) {
        return it->second;
    }
    return "unknown";
}

int SceneSemanticMap::getCellCountBySceneId(int scene_id) const
{
    int count = 0;
    for (size_t i = 0; i < scene_id_grid_.data.size(); ++i) {
        if (static_cast<int>(scene_id_grid_.data[i]) == scene_id) {
            count++;
        }
    }
    return count;
}

int SceneSemanticMap::getCellCountBySceneType(const std::string& scene_type) const
{
    int count = 0;
    // 找到所有匹配scene_type的scene_id
    std::vector<int> matching_ids;
    for (const auto& pair : scene_id_to_type_) {
        if (pair.second == scene_type) {
            matching_ids.push_back(pair.first);
        }
    }
    
    // 统计这些scene_id的cell数量
    for (size_t i = 0; i < scene_id_grid_.data.size(); ++i) {
        int id = static_cast<int>(scene_id_grid_.data[i]);
        if (std::find(matching_ids.begin(), matching_ids.end(), id) != matching_ids.end()) {
            count++;
        }
    }
    
    return count;
}

void SceneSemanticMap::clear()
{
    // 重置所有cell为未知状态
    std::fill(scene_id_grid_.data.begin(), scene_id_grid_.data.end(), SCENE_UNKNOWN);
    std::fill(confidence_grid_.data.begin(), confidence_grid_.data.end(), CONFIDENCE_UNKNOWN);
    
    // 清空场景类型映射表（可选，根据需求决定）
    // scene_id_to_type_.clear();
}

nav_msgs::OccupancyGrid SceneSemanticMap::getSceneIdGrid(const std::string& frame_id) const
{
    nav_msgs::OccupancyGrid grid = scene_id_grid_;
    grid.header.frame_id = frame_id;
    grid.header.stamp = ros::Time::now();
    return grid;
}

nav_msgs::OccupancyGrid SceneSemanticMap::getConfidenceGrid(const std::string& frame_id) const
{
    nav_msgs::OccupancyGrid grid = confidence_grid_;
    grid.header.frame_id = frame_id;
    grid.header.stamp = ros::Time::now();
    return grid;
}

int SceneSemanticMap::getOrCreateSceneId(const std::string& scene_attribute)
{
    if (scene_attribute.empty()) {
        ROS_WARN("SceneSemanticMap::getOrCreateSceneId: empty scene attribute");
        return -1;
    }
    
    // 标准化场景类型
    std::string normalized = normalizeSceneType(scene_attribute);
    
    // 硬编码的场景类型到语义类别索引映射（0-9对应通道4-13）
    static const std::map<std::string, int> SCENE_TYPE_TO_SEMANTIC_INDEX = {
        {"livingroom", 0},  // 通道4
        {"bedroom", 1},     // 通道5
        {"kitchen", 2},     // 通道6
        {"bathroom", 3},    // 通道7
        {"balcony", 4},     // 通道8
        {"storage", 5},    // 通道9
        {"door", 6},       // 通道10
        {"wall", 7},       // 通道11
        {"entrance", 8},   // 通道12
        {"outside", 9}     // 通道13
    };
    
    // 查找映射
    auto it = SCENE_TYPE_TO_SEMANTIC_INDEX.find(normalized);
    if (it != SCENE_TYPE_TO_SEMANTIC_INDEX.end()) {
        int semantic_idx = it->second;
        // 记录场景类型（用于记录和可视化）
        scene_id_to_type_[semantic_idx] = scene_attribute;
        ROS_DEBUG("SceneSemanticMap: mapped scene type '%s' (normalized: '%s') -> semantic_index=%d", 
                 scene_attribute.c_str(), normalized.c_str(), semantic_idx);
        return semantic_idx;
    }
    
    // 即使不在预定义列表中，也注册到映射表中（用于记录和可视化）
    // 但返回-1表示无法映射到语义类别索引
    // 查找是否已存在该场景类型（使用相同的标准化函数进行比较）
    for (const auto& pair : scene_id_to_type_) {
        std::string existing_normalized = normalizeSceneType(pair.second);
        if (existing_normalized == normalized) {
            ROS_DEBUG("SceneSemanticMap: found existing scene type '%s' (normalized: '%s') with scene_id=%d (not mapped to semantic index)", 
                     scene_attribute.c_str(), normalized.c_str(), pair.first);
            return -1;  // 已注册但不在预定义列表中
        }
    }
    
    // 注册新的场景类型（使用一个临时ID，但返回-1表示无法映射）
    // 注意：由于只有0-9是有效的语义类别索引，这里使用一个临时ID来存储
    // 但实际使用时，这个ID不会被映射到语义通道
    static int temp_id_counter = 10;  // 从10开始，避免与0-9冲突
    int temp_id = temp_id_counter++;
    scene_id_to_type_[temp_id] = scene_attribute;
    ROS_INFO("SceneSemanticMap: registered scene type '%s' (normalized: '%s') with temp_id=%d (not mapped to semantic index [0-9])", 
             scene_attribute.c_str(), normalized.c_str(), temp_id);
    return -1;  // 返回-1表示无法映射到语义类别索引
}

void SceneSemanticMap::initializeSceneTypeMapping(ros::NodeHandle& nh)
{
    if (scene_type_mapping_initialized_) {
        return;  // 已经初始化
    }
    
    // 加载同义词映射（用于标准化场景类型）
    XmlRpc::XmlRpcValue synonyms;
    if (nh.getParam("scene_type_synonyms", synonyms)) {
        if (synonyms.getType() == XmlRpc::XmlRpcValue::TypeStruct) {
            for (auto it = synonyms.begin(); it != synonyms.end(); ++it) {
                std::string key = it->first;
                if (it->second.getType() == XmlRpc::XmlRpcValue::TypeString) {
                    std::string value = static_cast<std::string>(it->second);
                    scene_type_synonyms_[key] = value;
                }
            }
            ROS_INFO("SceneSemanticMap: loaded %zu synonym mappings", scene_type_synonyms_.size());
        }
    } else {
        ROS_WARN("SceneSemanticMap: scene_type_synonyms parameter not found, using defaults");
    }
    
    scene_type_mapping_initialized_ = true;
    ROS_INFO("SceneSemanticMap: scene type mapping initialized (using hardcoded semantic index mapping)");
}

std::string SceneSemanticMap::normalizeSceneType(const std::string& raw_type) const
{
    if (raw_type.empty()) {
        return "";
    }
    
    // 1. 转小写
    std::string normalized = raw_type;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
    
    // 2. 去除空格和下划线
    normalized.erase(std::remove_if(normalized.begin(), normalized.end(), 
                    [](char c) { return std::isspace(c) || c == '_'; }), 
                    normalized.end());
    
    // 3. 同义词映射（如果已初始化）
    if (scene_type_mapping_initialized_) {
        auto it = scene_type_synonyms_.find(normalized);
        if (it != scene_type_synonyms_.end()) {
            return it->second;
        }
    }
    
    // 如果没有找到映射，返回标准化后的字符串
    return normalized;
}

// visualize code dont change
















std_msgs::ColorRGBA SceneSemanticMap::getColorForSceneType(const std::string& normalized_scene_type) const
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

std_msgs::ColorRGBA SceneSemanticMap::getColorForSceneIdInternal(int scene_id) const
{
    std_msgs::ColorRGBA color;
    color.a = 1.0;
    
    // 特殊ID使用固定颜色
    if (scene_id == SCENE_UNKNOWN) {
        color.r = 0.5; color.g = 0.5; color.b = 0.5;  // 灰色 - 未知
        return color;
    }
    
    // 查找场景类型名称
    auto it = scene_id_to_type_.find(scene_id);
    if (it != scene_id_to_type_.end()) {
        // 标准化场景类型
        std::string normalized = normalizeSceneType(it->second);
        // 尝试使用场景类型的固定颜色
        std_msgs::ColorRGBA type_color = getColorForSceneType(normalized);
        if (type_color.r >= 0.0) {  // 如果找到了预定义颜色
            return type_color;
        }
    }
    
    // 如果场景类型没有预定义颜色，使用HSV颜色空间根据scene_id生成颜色
    // scene_id 范围是 0-9，直接使用
    float hue = static_cast<float>(scene_id % 10) / 10.0f;  // 0-1范围（对应0-360度，均匀分布10个类别）
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

sensor_msgs::PointCloud2 SceneSemanticMap::getColoredPointCloud(const std::string& frame_id, double height) const
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    
    // 遍历所有栅格cell
    for (unsigned int y = 0; y < height_; ++y) {
        for (unsigned int x = 0; x < width_; ++x) {
            size_t idx = gridToIndex(x, y);
            int scene_id = static_cast<int>(scene_id_grid_.data[idx]);
            
            // 跳过未知区域
            if (scene_id == SCENE_UNKNOWN) {
                continue;
            }
            
            // 获取该cell的世界坐标
            double world_x, world_y;
            gridToWorld(x, y, world_x, world_y);
            
            // 获取颜色
            std_msgs::ColorRGBA color = getColorForSceneIdInternal(scene_id);
            
            // 创建点
            pcl::PointXYZRGB point;
            point.x = world_x;
            point.y = world_y;
            point.z = height;
            point.r = static_cast<uint8_t>(color.r * 255);
            point.g = static_cast<uint8_t>(color.g * 255);
            point.b = static_cast<uint8_t>(color.b * 255);
            
            cloud.points.push_back(point);
        }
    }
    
    // 转换为ROS消息
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(cloud, cloud_msg);
    cloud_msg.header.frame_id = frame_id;
    cloud_msg.header.stamp = ros::Time::now();
    
    return cloud_msg;
}

visualization_msgs::MarkerArray SceneSemanticMap::getLegendMarkerArray(const std::string& frame_id, 
                                                                        double position_x, 
                                                                        double position_y, 
                                                                        double position_z) const
{
    visualization_msgs::MarkerArray marker_array;
    
    // 图例参数
    double cube_size = 0.3;  // 颜色块大小
    double text_size = 0.2;  // 文字大小
    double spacing = 0.4;    // 每行间距
    double start_x = position_x;
    double start_y = position_y;
    double current_y = start_y;
    
    int marker_id = 0;
    
    // 1. 添加特殊ID的图例
    std::vector<std::pair<int, std::string>> special_scenes = {
        {SCENE_UNKNOWN, "Unknown"}
    };
    
    for (const auto& pair : special_scenes) {
        int scene_id = pair.first;
        std::string label = pair.second;
        std_msgs::ColorRGBA color = getColorForSceneIdInternal(scene_id);
        
        // 颜色块（CUBE）
        visualization_msgs::Marker color_marker;
        color_marker.header.frame_id = frame_id;
        color_marker.header.stamp = ros::Time::now();
        color_marker.ns = "scene_legend";
        color_marker.id = marker_id++;
        color_marker.type = visualization_msgs::Marker::CUBE;
        color_marker.action = visualization_msgs::Marker::ADD;
        color_marker.pose.position.x = start_x;
        color_marker.pose.position.y = current_y;
        color_marker.pose.position.z = position_z;
        color_marker.pose.orientation.w = 1.0;
        color_marker.scale.x = cube_size;
        color_marker.scale.y = cube_size;
        color_marker.scale.z = cube_size;
        color_marker.color = color;
        marker_array.markers.push_back(color_marker);
        
        // 文字标签（TEXT_VIEW_FACING）
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = frame_id;
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "scene_legend";
        text_marker.id = marker_id++;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        text_marker.pose.position.x = start_x + cube_size * 0.6;
        text_marker.pose.position.y = current_y;
        text_marker.pose.position.z = position_z;
        text_marker.pose.orientation.w = 1.0;
        text_marker.scale.z = text_size;
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        text_marker.text = label;
        marker_array.markers.push_back(text_marker);
        
        current_y -= spacing;
    }
    
    // 2. 添加常见场景类型的图例（按固定顺序）
    std::vector<std::string> common_scenes = {
        "bathroom", "corridor", "kitchen", "bedroom", "livingroom",
        "dinningroom", "laundry", "storage", "garden", "parkinglot"
    };
    
    for (const std::string& scene_type : common_scenes) {
        // 检查是否已注册（使用统一的标准化函数）
        bool found = false;
        int scene_id = -1;
        std::string normalized_scene = normalizeSceneType(scene_type);
        
        for (const auto& pair : scene_id_to_type_) {
            std::string normalized_existing = normalizeSceneType(pair.second);
            
            if (normalized_existing == normalized_scene) {
                found = true;
                scene_id = pair.first;
                break;
            }
        }
        
        if (!found) continue;  // 如果场景类型未注册，跳过
        
        std_msgs::ColorRGBA color = getColorForSceneIdInternal(scene_id);
        
        // 颜色块
        visualization_msgs::Marker color_marker;
        color_marker.header.frame_id = frame_id;
        color_marker.header.stamp = ros::Time::now();
        color_marker.ns = "scene_legend";
        color_marker.id = marker_id++;
        color_marker.type = visualization_msgs::Marker::CUBE;
        color_marker.action = visualization_msgs::Marker::ADD;
        color_marker.pose.position.x = start_x;
        color_marker.pose.position.y = current_y;
        color_marker.pose.position.z = position_z;
        color_marker.pose.orientation.w = 1.0;
        color_marker.scale.x = cube_size;
        color_marker.scale.y = cube_size;
        color_marker.scale.z = cube_size;
        color_marker.color = color;
        marker_array.markers.push_back(color_marker);
        
        // 文字标签
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = frame_id;
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "scene_legend";
        text_marker.id = marker_id++;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        text_marker.pose.position.x = start_x + cube_size * 0.6;
        text_marker.pose.position.y = current_y;
        text_marker.pose.position.z = position_z;
        text_marker.pose.orientation.w = 1.0;
        text_marker.scale.z = text_size;
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        text_marker.text = scene_type;
        marker_array.markers.push_back(text_marker);
        
        current_y -= spacing;
    }
    
    // 3. 添加其他已注册的场景类型（不在常见列表中的）
    for (const auto& pair : scene_id_to_type_) {
        int scene_id = pair.first;
        std::string scene_type = pair.second;
        
        // 跳过特殊ID和已处理的常见场景
        if (scene_id == SCENE_UNKNOWN) continue;
        
        bool is_common = false;
        std::string normalized_scene = normalizeSceneType(scene_type);
        
        for (const std::string& common : common_scenes) {
            std::string normalized_common = normalizeSceneType(common);
            if (normalized_scene == normalized_common) {
                is_common = true;
                break;
            }
        }
        
        if (is_common) continue;
        
        std_msgs::ColorRGBA color = getColorForSceneIdInternal(scene_id);
        
        // 颜色块
        visualization_msgs::Marker color_marker;
        color_marker.header.frame_id = frame_id;
        color_marker.header.stamp = ros::Time::now();
        color_marker.ns = "scene_legend";
        color_marker.id = marker_id++;
        color_marker.type = visualization_msgs::Marker::CUBE;
        color_marker.action = visualization_msgs::Marker::ADD;
        color_marker.pose.position.x = start_x;
        color_marker.pose.position.y = current_y;
        color_marker.pose.position.z = position_z;
        color_marker.pose.orientation.w = 1.0;
        color_marker.scale.x = cube_size;
        color_marker.scale.y = cube_size;
        color_marker.scale.z = cube_size;
        color_marker.color = color;
        marker_array.markers.push_back(color_marker);
        
        // 文字标签
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = frame_id;
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "scene_legend";
        text_marker.id = marker_id++;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        text_marker.pose.position.x = start_x + cube_size * 0.6;
        text_marker.pose.position.y = current_y;
        text_marker.pose.position.z = position_z;
        text_marker.pose.orientation.w = 1.0;
        text_marker.scale.z = text_size;
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        text_marker.text = scene_type + " (ID:" + std::to_string(scene_id) + ")";
        marker_array.markers.push_back(text_marker);
        
        current_y -= spacing;
    }
    
    return marker_array;
}
