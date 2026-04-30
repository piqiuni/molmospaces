#include "semantic_mapping_pkg/semantic_mapping_node.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <limits>
#include <algorithm>
#include <memory>
#include <ctime>
#include <iomanip>
#include <sstream>

SemanticMappingNode::SemanticMappingNode()
    : object_map_(0.1)  // 0.1m voxel size
    , object_mapper_(&object_map_)
    , scene_map_(0.1, 1000, 1000, -50.0, -50.0)  // 默认参数，后续会从参数服务器读取
    , scene_mapper_(&scene_map_)
    , tf_listener_(tf_buffer_)
    , enable_object_mapping_(true)
    , enable_scene_mapping_(true)
    , max_pointcloud_queue_size_(10)
{
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    // 参数（从全局命名空间读取，因为配置文件通过rosparam加载到全局命名空间）
    nh.param("enable_object_mapping", enable_object_mapping_, true);
    nh.param("enable_scene_mapping", enable_scene_mapping_, true);
    nh.param("voxel_size", voxel_size_, 0.1);
    nh.param("position_threshold", position_threshold_, 0.5);
    nh.param("size_threshold", size_threshold_, 0.3);
    nh.param("time_window", time_window_, 5.0);
    nh.param("confidence_threshold", confidence_threshold_, 0.5);
    nh.param("camera_frame", camera_frame_, std::string("camera"));
    nh.param("world_frame", world_frame_, std::string("map"));
    nh.param("detection_topic", detection_topic_, std::string("/explore_agent/result_info"));
    nh.param("pointcloud_topic", pointcloud_topic_, std::string("/local_scan"));
    nh.param("scene_attribute_topic", scene_attribute_topic_, std::string("/semantic_mapping/scene_attribute"));
    nh.param("occupancy_grid_topic", occupancy_grid_topic_, std::string("/struct_mapping/wall_occ_map"));
    nh.param("input_in_world_frame", input_in_world_frame_, true);  // 默认假设输入是相机坐标系
    
    // Scene mapping参数
    nh.param("scene_map_resolution", scene_map_resolution_, 0.1);
    nh.param("scene_map_width", scene_map_width_, 400);
    nh.param("scene_map_height", scene_map_height_, 400);
    nh.param("scene_map_origin_x", scene_map_origin_x_, -20.0);
    nh.param("scene_map_origin_y", scene_map_origin_y_, -20.0);
    nh.param("scene_max_range", scene_max_range_, 10.0);
    nh.param("scene_min_range", scene_min_range_, 0.1);
    nh.param("scene_occlusion_threshold", scene_occlusion_threshold_, 50);
    
    
    if (enable_object_mapping_) {
        // 设置mapper参数
        object_mapper_.setPositionThreshold(position_threshold_);
        object_mapper_.setSizeThreshold(size_threshold_);
        object_mapper_.setTimeWindow(time_window_);
        object_mapper_.setConfidenceThreshold(confidence_threshold_);
        // 订阅话题
        detection_sub_ = nh.subscribe<std_msgs::String>(
            detection_topic_, 10, 
            &SemanticMappingNode::detectionCallback, this);
        // 发布话题
        object_marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>(
            "/semantic_mapping/object_semantic_map_markers", 1);
        object_semantic_map_pub_ = nh.advertise<std_msgs::String>(
            "/semantic_mapping/obj_map", 1);
    }
    
    if (enable_scene_mapping_) {
        // 初始化scene map（使用参数）
        // 注意：SceneSemanticMap构造函数接受unsigned int，int可以隐式转换
        scene_map_ = SceneSemanticMap(scene_map_resolution_, 
                                      static_cast<unsigned int>(scene_map_width_), 
                                      static_cast<unsigned int>(scene_map_height_), 
                                      scene_map_origin_x_, scene_map_origin_y_);
        // 初始化场景类型映射（从配置文件加载）
        scene_map_.initializeSceneTypeMapping(nh);
        scene_mapper_ = SceneSemanticMapper(&scene_map_);
        
        // 设置scene mapper参数
        scene_mapper_.setMaxRange(scene_max_range_);
        scene_mapper_.setMinRange(scene_min_range_);
        scene_mapper_.setOcclusionThreshold(static_cast<int8_t>(scene_occlusion_threshold_));
        
        // 订阅话题
        scene_attribute_sub_ = nh.subscribe<std_msgs::String>(
            scene_attribute_topic_, 10,
            &SemanticMappingNode::sceneAttributeCallback, this);
        pointcloud_sub_ = nh.subscribe<sensor_msgs::PointCloud2>(
            pointcloud_topic_, 10,
            &SemanticMappingNode::pointCloudCallback, this);
        occupancy_grid_sub_ = nh.subscribe<nav_msgs::OccupancyGrid>(
            occupancy_grid_topic_, 1,
            &SemanticMappingNode::occupancyGridCallback, this);
        
        // 发布话题
        scene_colored_pointcloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>(
            "/semantic_mapping/scene_semantic_map/colored_pointcloud", 1);
        scene_legend_pub_ = nh.advertise<visualization_msgs::MarkerArray>(
            "/semantic_mapping/scene_semantic_map/legend", 1);
        scene_id_grid_pub_ = nh.advertise<nav_msgs::OccupancyGrid>(
            "/semantic_mapping/scene_id_grid", 1, true);
        scene_confidence_grid_pub_ = nh.advertise<nav_msgs::OccupancyGrid>(
            "/semantic_mapping/scene_confidence_grid", 1, true);
    }
    
    // 定时器：定期发布可视化
    timer_ = nh.createTimer(ros::Duration(0.5), 
                           &SemanticMappingNode::publishMarkers, this);
    
    // 对象语义地图发布定时器（与可视化同步频率）
    if (enable_object_mapping_) {
        object_map_pub_timer_ = nh.createTimer(ros::Duration(0.5),
                                              &SemanticMappingNode::publishObjectSemanticMap, this);
    }
    
    // 图例定时器：每10秒更新一次位置（降低算力消耗）
    if (enable_scene_mapping_) {
        legend_timer_ = nh.createTimer(ros::Duration(10.0),
                                      &SemanticMappingNode::publishLegend, this);
        // 场景ID映射发布定时器：每5秒更新一次ROS参数服务器
        scene_id_map_timer_ = nh.createTimer(ros::Duration(5.0),
                                           &SemanticMappingNode::publishSceneIdToTypeMap, this);
    }
    
    ROS_INFO("SemanticMappingNode initialized");
    ROS_INFO("  - Object mapping: %s", enable_object_mapping_ ? "enabled" : "disabled");
    ROS_INFO("  - Scene mapping: %s", enable_scene_mapping_ ? "enabled" : "disabled");
    if (enable_scene_mapping_) {
        ROS_INFO("    Scene map: %.1fx%.1f m, resolution=%.3f m/pixel", 
                 scene_map_width_ * scene_map_resolution_, 
                 scene_map_height_ * scene_map_resolution_,
                 scene_map_resolution_);
        ROS_INFO("    Subscribed to: %s, %s", 
                 scene_attribute_topic_.c_str(), pointcloud_topic_.c_str());
    }
}

void SemanticMappingNode::detectionCallback(const std_msgs::String::ConstPtr& msg)
{
    if (!enable_object_mapping_) {
        return;
    }
    
    // 解析JSON检测结果
    rapidjson::Document doc;
    doc.Parse(msg->data.c_str());
    
    if (doc.HasParseError() || !doc.IsArray()) {
        ROS_WARN_THROTTLE(1.0, "Failed to parse detection JSON");
        return;
    }
    
    // 获取相机位姿（如果TF不可用，跳过本次更新）
    geometry_msgs::Pose camera_pose = getCameraPoseInWorld();
    
    // 检查位姿是否有效（通过检查四元数模长）
    double q_norm = std::sqrt(
        camera_pose.orientation.x * camera_pose.orientation.x +
        camera_pose.orientation.y * camera_pose.orientation.y +
        camera_pose.orientation.z * camera_pose.orientation.z +
        camera_pose.orientation.w * camera_pose.orientation.w
    );
    
    if (q_norm < 0.1) {
        ROS_WARN_THROTTLE(1.0, "Camera pose invalid, skipping detection update");
        return;
    }
    
    // 转换为Detection格式
    std::vector<Detection> detections;
    for (const auto& obj : doc.GetArray()) {
        if (!obj.IsObject()) continue;
        
        Detection detection;
        
        // 解析语义类别
        if (obj.HasMember("class") && obj["class"].IsString()) {
            detection.semantic_class = obj["class"].GetString();
        } else if (obj.HasMember("semantic_class") && obj["semantic_class"].IsString()) {
            detection.semantic_class = obj["semantic_class"].GetString();
        }

        // 解析位置
        geometry_msgs::Point input_pos;
        if (obj.HasMember("position") && obj["position"].IsObject()) {
            const auto& pos = obj["position"];
            input_pos.x = pos.HasMember("x") ? pos["x"].GetDouble() : 0.0;
            input_pos.y = pos.HasMember("y") ? pos["y"].GetDouble() : 0.0;
            input_pos.z = pos.HasMember("z") ? pos["z"].GetDouble() : 0.0;
        }

        if (detection.semantic_class == "Painting") {
            
            ROS_WARN("input_pos: %f, %f, %f", input_pos.x, input_pos.y, input_pos.z);
        }
        
        // 根据输入坐标系类型进行转换
        // ROS_WARN("input_in_world_frame_: %d", input_in_world_frame_);
        if (input_in_world_frame_) {
            // 输入是世界坐标系，转换为相机坐标系
            detection.position = common::transformToCamera(input_pos, camera_pose);
        } else {
            // 输入是相机坐标系，直接使用
            detection.position = input_pos;
        }
        
        // 解析尺寸
        if (obj.HasMember("size") && obj["size"].IsObject()) {
            const auto& size = obj["size"];
            detection.size.x = size.HasMember("x") ? size["x"].GetDouble() : 0.1;
            detection.size.y = size.HasMember("y") ? size["y"].GetDouble() : 0.1;
            detection.size.z = size.HasMember("z") ? size["z"].GetDouble() : 0.1;
        }
        
        
        
        // 解析置信度
        if (obj.HasMember("confidence") && obj["confidence"].IsNumber()) {
            detection.confidence = obj["confidence"].GetDouble();
        }
        
        // 解析实例ID（可选）
        if (obj.HasMember("instance_id") && obj["instance_id"].IsString()) {
            detection.instance_id = obj["instance_id"].GetString();
        }
        
        detection.timestamp = ros::Time::now();
        
        detections.push_back(detection);
    }
    
    // 更新地图
    if (!detections.empty()) {
        object_mapper_.updateDetections(detections, camera_pose);
    }
}

void SemanticMappingNode::sceneAttributeCallback(const std_msgs::String::ConstPtr& msg)
{
    if (!enable_scene_mapping_) {
        return;
    }
    
    // 解析消息（JSON格式：包含场景属性和图片时间戳）
    rapidjson::Document doc;
    doc.Parse(msg->data.c_str());
    
    std::string scene_attribute;
    ros::Time image_timestamp;
    
    if (doc.IsObject() && doc.HasMember("scene_attribute") && doc.HasMember("image_timestamp_sec")) {
        // 新格式：JSON包含时间戳
        scene_attribute = doc["scene_attribute"].GetString();
        // 使用整数读取，避免精度丢失（JSON中现在是整数）
        uint64_t timestamp_sec_uint64 = 0;
        if (doc["image_timestamp_sec"].IsUint64()) {
            timestamp_sec_uint64 = doc["image_timestamp_sec"].GetUint64();
        } else if (doc["image_timestamp_sec"].IsInt64()) {
            timestamp_sec_uint64 = static_cast<uint64_t>(doc["image_timestamp_sec"].GetInt64());
        } else {
            // 向后兼容：如果是浮点数，转换为整数
            timestamp_sec_uint64 = static_cast<uint64_t>(doc["image_timestamp_sec"].GetDouble());
        }
        uint32_t timestamp_sec = static_cast<uint32_t>(timestamp_sec_uint64);
        
        uint64_t timestamp_nsec_uint64 = 0;
        if (doc.HasMember("image_timestamp_nsec")) {
            timestamp_nsec_uint64 = doc["image_timestamp_nsec"].GetUint64();
        }
        uint32_t timestamp_nsec = static_cast<uint32_t>(timestamp_nsec_uint64);
        
        image_timestamp = ros::Time(timestamp_sec, timestamp_nsec);
        
        // 调试输出：检查解析到的时间戳
        ROS_INFO_THROTTLE(5.0, "[SceneMapping] Parsed timestamp from JSON: sec=%u, nsec=%u, total=%.9f, valid=%s", 
                         timestamp_sec, timestamp_nsec, image_timestamp.toSec(),
                         (image_timestamp.toSec() > 1000000000) ? "YES" : "NO");
        
        // 如果时间戳无效，输出原始JSON值用于调试
        if (image_timestamp.toSec() == 0 || image_timestamp.toSec() < 1000000000) {
            ROS_WARN_THROTTLE(2.0, "[SceneMapping] Invalid timestamp in JSON: sec=%u, nsec=%u, raw JSON: %s", 
                            timestamp_sec, timestamp_nsec, msg->data.c_str());
        }
    } else {
        // 旧格式：纯字符串（向后兼容）
        scene_attribute = msg->data;
        image_timestamp = ros::Time(0);  // 使用0表示无效时间戳
        ROS_WARN_THROTTLE(5.0, "[SceneMapping] Received old format message (no timestamp), using current time TF");
    }
    
    if (scene_attribute.empty()) {
        ROS_WARN_THROTTLE(1.0, "Empty scene attribute, skipping");
        return;
    }
    
    static int scene_attr_count = 0;
    scene_attr_count++;
    ros::Time now = ros::Time::now();
    double age = (now - image_timestamp).toSec();
    
    if (scene_attr_count % 10 == 0) {
        std::lock_guard<std::mutex> lock(scene_data_mutex_);
        ROS_INFO_THROTTLE(5.0, "[SceneMapping] Received scene attr: '%s', image_ts=%.3f, age=%.3fs, cloud_queue_size=%zu, total_received=%d", 
                         scene_attribute.c_str(), image_timestamp.toSec(), age, 
                         pointcloud_queue_.size(), scene_attr_count);
    }
    
    // 使用图片时间戳查找TF和点云
    processSceneMappingWithTimestamp(scene_attribute, image_timestamp);
}

void SemanticMappingNode::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    if (!enable_scene_mapping_) {
        return;
    }
    
    // 将点云加入队列（带时间戳）
    std::lock_guard<std::mutex> lock(scene_data_mutex_);
    
    PointCloudData data;
    data.msg = msg;
    data.timestamp = msg->header.stamp;  // 使用点云的时间戳
    pointcloud_queue_.push_back(data);
    
    // 限制队列长度
    if (pointcloud_queue_.size() > max_pointcloud_queue_size_) {
        pointcloud_queue_.erase(pointcloud_queue_.begin());
    }
    
    // 清理过期的点云（超过10秒）
    ros::Time now = ros::Time::now();
    size_t before_size = pointcloud_queue_.size();
    pointcloud_queue_.erase(
        std::remove_if(pointcloud_queue_.begin(), pointcloud_queue_.end(),
            [now](const PointCloudData& data) {
                return (now - data.timestamp).toSec() > 10.0;
            }),
        pointcloud_queue_.end()
    );
    
    static int pointcloud_count = 0;
    pointcloud_count++;
    if (pointcloud_count % 50 == 0) {
        ROS_INFO_THROTTLE(5.0, "[SceneMapping] Received pointcloud: ts=%.3f, queue_size=%zu, total_received=%d", 
                         data.timestamp.toSec(), pointcloud_queue_.size(), pointcloud_count);
    }
}

void SemanticMappingNode::occupancyGridCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
    if (!enable_scene_mapping_) {
        return;
    }
    
    // 更新scene mapper的occupancy grid用于遮挡检测
    scene_mapper_.setOccupancyGrid(*msg);
    
    // 如果scene map还没有初始化，从occupancy grid初始化
    // （这样可以确保scene map和occupancy grid有相同的尺寸和分辨率）
    static bool initialized_from_occ_grid = false;
    if (!initialized_from_occ_grid) {
        scene_map_.initializeFromOccupancyGrid(*msg);
        initialized_from_occ_grid = true;
        ROS_INFO("SceneSemanticMap initialized from OccupancyGrid");
    }
}

void SemanticMappingNode::processSceneMappingWithTimestamp(const std::string& scene_attribute, const ros::Time& image_timestamp)
{
    if (!enable_scene_mapping_) {
        return;
    }
    
    static int total_requests = 0;
    static int tf_failed = 0;
    static int no_pointcloud = 0;
    static int time_mismatch = 0;
    static int success_count = 0;
    total_requests++;
    
    ros::Time now = ros::Time::now();
    double age = (now - image_timestamp).toSec();
    
    // 1. 使用图片时间戳查找对应的相机位姿（TF）
    geometry_msgs::Pose camera_pose;
    bool tf_valid = false;
    
    if (image_timestamp.toSec() > 0) {
        // 有效时间戳，查找对应时间的TF
        camera_pose = getCameraPoseAtTime(image_timestamp);
        
        // 检查位姿是否有效（通过检查位置是否为零点来判断TF是否成功）
        double pos_norm = std::sqrt(
            camera_pose.position.x * camera_pose.position.x +
            camera_pose.position.y * camera_pose.position.y +
            camera_pose.position.z * camera_pose.position.z
        );
        
        double q_norm = std::sqrt(
            camera_pose.orientation.x * camera_pose.orientation.x +
            camera_pose.orientation.y * camera_pose.orientation.y +
            camera_pose.orientation.z * camera_pose.orientation.z +
            camera_pose.orientation.w * camera_pose.orientation.w
        );
        
        // 如果位置和四元数都是默认值（零点+单位四元数），说明TF查找失败
        if (pos_norm < 0.01 && q_norm > 0.9 && q_norm < 1.1) {
        tf_failed++;
        ROS_WARN_THROTTLE(2.0, "[SceneMapping] TF lookup failed: image_ts=%.3f, age=%.3fs, queue_size=%zu, stats: total=%d, tf_fail=%d, no_cloud=%d, time_mismatch=%d, success=%d", 
                            image_timestamp.toSec(), age, pointcloud_queue_.size(), 
                            total_requests, tf_failed, no_pointcloud, time_mismatch, success_count);
            return;
        }
        tf_valid = true;
    } else {
        // 无效时间戳，使用当前时间的TF（向后兼容）
        camera_pose = getCameraPoseInWorld();
        tf_valid = true;
    }
    
    // 2. 查找时间戳最接近的点云
    std::unique_lock<std::mutex> lock(scene_data_mutex_);
    
    if (pointcloud_queue_.empty()) {
        no_pointcloud++;
        ROS_WARN_THROTTLE(2.0, "[SceneMapping] Pointcloud queue empty: image_ts=%.3f, stats: total=%d, tf_fail=%d, no_cloud=%d, time_mismatch=%d, success=%d", 
                        image_timestamp.toSec(), total_requests, tf_failed, no_pointcloud, time_mismatch, success_count);
        return;
    }
    
    // 找到时间戳最接近的点云
    double min_time_diff = std::numeric_limits<double>::max();
    sensor_msgs::PointCloud2::ConstPtr matched_pointcloud = nullptr;
    ros::Time matched_cloud_timestamp;
    
    for (const auto& cloud_data : pointcloud_queue_) {
        double time_diff = std::abs((cloud_data.timestamp - image_timestamp).toSec());
        if (time_diff < min_time_diff) {
            min_time_diff = time_diff;
            matched_pointcloud = cloud_data.msg;
            matched_cloud_timestamp = cloud_data.timestamp;
        }
    }
    
    // 放宽时间匹配窗口到5秒（因为BLIP推理有延迟，且图片和点云可能不是完全同步）
    if (!matched_pointcloud || min_time_diff > 5.0) {
        time_mismatch++;
        ROS_WARN_THROTTLE(2.0, "[SceneMapping] Time mismatch: image_ts=%.3f, cloud_ts=%.3f, diff=%.3fs, queue_size=%zu, stats: total=%d, tf_fail=%d, no_cloud=%d, time_mismatch=%d, success=%d", 
                        image_timestamp.toSec(), 
                        matched_pointcloud ? matched_cloud_timestamp.toSec() : 0.0,
                        min_time_diff, pointcloud_queue_.size(),
                        total_requests, tf_failed, no_pointcloud, time_mismatch, success_count);
        return;
    }
    
    lock.unlock();
    
    // 3. 调用scene mapper更新地图
    scene_mapper_.updateFromPointCloud(
        *matched_pointcloud,
        camera_pose,
        scene_attribute,
        1.0f,  // 置信度
        std::numeric_limits<double>::quiet_NaN(),  // min_yaw_angle
        std::numeric_limits<double>::quiet_NaN()   // max_yaw_angle
    );
    
    success_count++;
    ROS_INFO_THROTTLE(1.0, "[SceneMapping] Success: attr='%s', image_ts=%.3f, cloud_ts=%.3f, diff=%.3fs, stats: total=%d, success=%d (%.1f%%)", 
                     scene_attribute.c_str(), image_timestamp.toSec(), matched_cloud_timestamp.toSec(), min_time_diff,
                     total_requests, success_count, 100.0 * success_count / total_requests);
}

void SemanticMappingNode::publishMarkers(const ros::TimerEvent& event)
{
    if (enable_object_mapping_) {
        visualization_msgs::MarkerArray marker_array = 
            object_map_.toMarkerArray(world_frame_);
        object_marker_pub_.publish(marker_array);
    }
    
    if (enable_scene_mapping_) {
        // 发布scene map的彩色点云
        sensor_msgs::PointCloud2 colored_cloud = scene_map_.getColoredPointCloud(world_frame_, 0.1);
        scene_colored_pointcloud_pub_.publish(colored_cloud);
        
        // 发布scene_id_grid和confidence_grid（用于14通道转换）
        nav_msgs::OccupancyGrid scene_id_grid = scene_map_.getSceneIdGrid(world_frame_);
        nav_msgs::OccupancyGrid confidence_grid = scene_map_.getConfidenceGrid(world_frame_);
        scene_id_grid_pub_.publish(scene_id_grid);
        scene_confidence_grid_pub_.publish(confidence_grid);
    }
}

void SemanticMappingNode::publishLegend(const ros::TimerEvent& event)
{
    if (!enable_scene_mapping_) {
        return;
    }
    
    // 发布图例（动态计算位置：已知区域右上角外一定距离）
    double legend_x, legend_y;
    double min_x, min_y, max_x, max_y;
    const double legend_offset = 2.0;  // 图例距离已知区域边界的距离（米）
    
    if (scene_map_.getKnownRegionBounds(min_x, min_y, max_x, max_y)) {
        // 如果找到已知区域，图例放在右上角外
        legend_x = max_x + legend_offset;
        legend_y = max_y + legend_offset;
    } else {
        // 如果没有已知区域，使用地图原点作为默认位置
        legend_x = scene_map_.getOriginX() + legend_offset;
        legend_y = scene_map_.getOriginY() + legend_offset;
    }
    
    visualization_msgs::MarkerArray legend = scene_map_.getLegendMarkerArray(world_frame_, legend_x, legend_y, 0.5);
    scene_legend_pub_.publish(legend);
}

void SemanticMappingNode::publishSceneIdToTypeMap(const ros::TimerEvent& event)
{
    if (!enable_scene_mapping_) {
        return;
    }
    
    // 获取所有场景ID到类型的映射
    std::map<int, std::string> scene_id_to_type = scene_map_.getAllSceneTypes();
    
    // 转换为ROS参数格式（字符串键）
    std::map<std::string, std::string> param_map;
    int user_defined_count = 0;
    for (const auto& pair : scene_id_to_type) {
        // 跳过预留ID（-1, 0, 1, 2），只发布用户自定义ID（3-127）
        if (pair.first >= 3 && pair.first <= 127) {
            param_map[std::to_string(pair.first)] = pair.second;
            user_defined_count++;
        }
    }
    
    // 发布到ROS参数服务器（全局命名空间，供其他节点使用）
    ros::NodeHandle nh;
    if (!param_map.empty()) {
        nh.setParam("/semantic_mapping/scene_id_to_type_map", param_map);
        ROS_INFO_THROTTLE(30.0, "Published scene_id_to_type_map with %d user-defined entries (total: %zu)", 
                         user_defined_count, scene_id_to_type.size());
        
        // 如果用户自定义ID过多，发出警告
        if (user_defined_count > 50) {
            ROS_WARN_THROTTLE(30.0, "检测到大量scene_id (%d个)，可能存在重复注册。建议检查BLIP返回的场景属性字符串是否一致", 
                            user_defined_count);
        }
    }
}

void SemanticMappingNode::publishObjectSemanticMap(const ros::TimerEvent& event)
{
    if (!enable_object_mapping_) {
        return;
    }
    
    // 获取所有体素（对象）
    std::vector<SemanticVoxel*> voxels = object_map_.getAllVoxels();
    
    if (voxels.empty()) {
        return;
    }
    
    // 转换为JSON格式
    rapidjson::Document doc;
    doc.SetArray();
    rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();
    
    for (const auto* voxel : voxels) {
        if (!voxel || voxel->semantic_class.empty()) {
            continue;
        }
        
        rapidjson::Value obj(rapidjson::kObjectType);
        
        // semantic_name (对应 explore_manager 期待的字段名)
        rapidjson::Value name_value(voxel->semantic_class.c_str(), allocator);
        obj.AddMember("semantic_name", name_value, allocator);
        
        // conf (置信度)
        obj.AddMember("conf", voxel->confidence, allocator);
        
        // coord (坐标数组 [x, y, z])
        rapidjson::Value coord_array(rapidjson::kArrayType);
        coord_array.PushBack(voxel->position.x, allocator);
        coord_array.PushBack(voxel->position.y, allocator);
        coord_array.PushBack(voxel->position.z, allocator);
        obj.AddMember("coord", coord_array, allocator);
        
        // env_status (可选，暂时留空或可以添加额外信息)
        // obj.AddMember("env_status", rapidjson::Value("", allocator), allocator);
        
        doc.PushBack(obj, allocator);
    }
    
    // 转换为字符串
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    
    // 发布
    std_msgs::String msg;
    msg.data = buffer.GetString();
    object_semantic_map_pub_.publish(msg);
    
    ROS_DEBUG_THROTTLE(5.0, "Published object semantic map: %zu objects", voxels.size());
}

geometry_msgs::Pose SemanticMappingNode::getCameraPoseInWorld()
{
    return getCameraPoseAtTime(ros::Time(0));  // 使用最新时间
}

geometry_msgs::Pose SemanticMappingNode::getCameraPoseAtTime(const ros::Time& time)
{
    geometry_msgs::Pose pose;
    
    ros::Time now = ros::Time::now();
    double time_age = (now - time).toSec();  // 计算时间差（在所有分支中都需要）
    
    // 检查时间戳是否有效（不是1970年或0）
    ros::Time lookup_time = time;
    if (time.toSec() == 0 || time.toSec() < 1000000000) {  // 1970年或更早，时间戳无效
        ROS_WARN_THROTTLE(2.0, "[SceneMapping] Invalid image timestamp (%.3f, year 1970), using current time for TF lookup", time.toSec());
        lookup_time = ros::Time(0);  // 使用最新时间
        time_age = 0.0;  // 时间戳无效时，时间差设为0
    } else {
        // 如果时间戳太旧（超过10秒），尝试使用最新时间
        if (time_age > 10.0) {
            // 格式化时间为可读格式（线程安全版本）
            time_t image_time_t = time.sec;
            time_t now_time_t = now.sec;
            char image_time_str[64], now_time_str[64];
            struct tm image_tm, now_tm;
            localtime_r(&image_time_t, &image_tm);
            localtime_r(&now_time_t, &now_tm);
            strftime(image_time_str, sizeof(image_time_str), "%Y-%m-%d %H:%M:%S", &image_tm);
            strftime(now_time_str, sizeof(now_time_str), "%Y-%m-%d %H:%M:%S", &now_tm);
            
            ROS_WARN_THROTTLE(2.0, "[SceneMapping] TF timestamp too old: image_time=%s (%.3f), current_time=%s (%.3f), age=%.3fs, using latest time", 
                             image_time_str, time.toSec(), now_time_str, now.toSec(), time_age);
            lookup_time = ros::Time(0);  // 使用最新时间
        }
    }
    
    try {
        // 增加超时时间到0.5秒，给TF buffer更多时间查找
        geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
            world_frame_, camera_frame_, lookup_time, ros::Duration(1.0));
        
        pose.position.x = transform.transform.translation.x;
        pose.position.y = transform.transform.translation.y;
        pose.position.z = transform.transform.translation.z;
        pose.orientation = transform.transform.rotation;
        
        // 验证四元数是否有效
        double q_norm = std::sqrt(
            pose.orientation.x * pose.orientation.x +
            pose.orientation.y * pose.orientation.y +
            pose.orientation.z * pose.orientation.z +
            pose.orientation.w * pose.orientation.w
        );
        
        if (q_norm < 0.1) {
            ROS_WARN_THROTTLE(1.0, "Invalid quaternion from TF at time %.3f, using default", lookup_time.toSec());
            pose.position.x = pose.position.y = pose.position.z = 0.0;
            pose.orientation.x = 0.0;
            pose.orientation.y = 0.0;
            pose.orientation.z = 0.0;
            pose.orientation.w = 1.0;
        }
    }
    catch (tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(2.0, "[SceneMapping] TF lookup failed at time %.3f (age=%.3fs): %s", 
                         lookup_time.toSec(), time_age, ex.what());
        // 使用默认位姿（单位四元数）
        pose.position.x = pose.position.y = pose.position.z = 0.0;
        pose.orientation.x = 0.0;
        pose.orientation.y = 0.0;
        pose.orientation.z = 0.0;
        pose.orientation.w = 1.0;
    }
    
    return pose;
}

// ========== Main函数 ==========
int main(int argc, char** argv)
{
    ros::init(argc, argv, "semantic_mapping_node");
    
    SemanticMappingNode node;
    
    ros::spin();
    
    return 0;
}
