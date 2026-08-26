#ifndef SEMANTIC_MAPPING_NODE_H_
#define SEMANTIC_MAPPING_NODE_H_

#include "semantic_mapping_pkg/object_semantic_map.h"
#include "semantic_mapping_pkg/object_semantic_mapper.h"
#include "semantic_mapping_pkg/scene_semantic_map.h"
#include "semantic_mapping_pkg/scene_semantic_mapper.h"
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/String.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <string>
#include <vector>
#include <mutex>

/**
 * @class SemanticMappingNode
 * @brief 语义建图ROS节点
 * 
 * 统一管理object和scene语义建图：
 * - 订阅检测结果
 * - 获取相机位姿（通过TF）
 * - 更新object和scene语义地图
 * - 发布可视化markers
 */
class SemanticMappingNode
{
public:
    /**
     * @brief 构造函数
     */
    SemanticMappingNode();
    
    /**
     * @brief 析构函数
     */
    ~SemanticMappingNode() = default;

private:
    /**
     * @brief 检测结果回调函数
     * @param msg JSON格式的检测结果
     */
    void detectionCallback(const std_msgs::String::ConstPtr& msg);
    
    /**
     * @brief 场景属性回调函数（单独处理，用于存储最新值）
     */
    void sceneAttributeCallback(const std_msgs::String::ConstPtr& msg);
    
    /**
     * @brief 点云回调函数（单独处理，用于存储最新值）
     */
    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
    
    /**
     * @brief Occupancy grid回调函数（用于遮挡检测）
     */
    void occupancyGridCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);
    
    /**
     * @brief 发布可视化markers
     */
    void publishMarkers(const ros::TimerEvent& event);
    
    /**
     * @brief 发布图例（单独定时器，降低更新频率）
     */
    void publishLegend(const ros::TimerEvent& event);
    
    /**
     * @brief 发布场景ID到类型的映射到ROS参数服务器
     */
    void publishSceneIdToTypeMap(const ros::TimerEvent& event);
    
    /**
     * @brief 发布对象语义地图（JSON格式）
     */
    void publishObjectSemanticMap(const ros::TimerEvent& event);
    
    /**
     * @brief 通过TF获取相机在世界坐标系中的位姿（当前时间）
     * @return 相机位姿
     */
    geometry_msgs::Pose getCameraPoseInWorld();
    
    /**
     * @brief 通过TF获取指定时间的相机在世界坐标系中的位姿
     * @param time 时间戳
     * @return 相机位姿
     */
    geometry_msgs::Pose getCameraPoseAtTime(const ros::Time& time);
    
    /**
     * @brief 处理场景建图更新（使用图片时间戳查找TF和点云）
     * @param scene_attribute 场景属性字符串
     * @param image_timestamp 图片时间戳
     */
    void processSceneMappingWithTimestamp(const std::string& scene_attribute, const ros::Time& image_timestamp);
    
    /**
     * @brief 处理场景建图更新（旧方法，保留用于向后兼容）
     */
    void processSceneMapping();
    
    // ========== Object语义地图 ==========
    ObjectSemanticMap object_map_;
    ObjectSemanticMapper object_mapper_;
    
    // ========== Scene语义地图 ==========
    SceneSemanticMap scene_map_;
    SceneSemanticMapper scene_mapper_;
    
    // ========== ROS接口 ==========
    ros::Subscriber detection_sub_;
    ros::Subscriber scene_attribute_sub_;
    ros::Subscriber pointcloud_sub_;
    ros::Subscriber occupancy_grid_sub_;
    ros::Publisher object_marker_pub_;
    ros::Publisher object_semantic_map_pub_;  // 发布对象语义地图（JSON格式）
    ros::Publisher scene_colored_pointcloud_pub_;
    ros::Publisher scene_legend_pub_;
    ros::Publisher scene_id_grid_pub_;
    ros::Publisher scene_confidence_grid_pub_;
    ros::Timer timer_;
    ros::Timer legend_timer_;
    ros::Timer scene_id_map_timer_;
    ros::Timer object_map_pub_timer_;  // 对象地图发布定时器
    
    // ========== Scene mapping数据缓存 ==========
    struct PointCloudData {
        sensor_msgs::PointCloud2::ConstPtr msg;
        ros::Time timestamp;  // 点云时间戳
    };
    std::vector<PointCloudData> pointcloud_queue_;  // 点云队列（带时间戳）
    std::mutex scene_data_mutex_;
    size_t max_pointcloud_queue_size_;  // 最大队列长度，默认10
    
    // ========== TF ==========
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // ========== 参数 ==========
    bool enable_object_mapping_;
    bool enable_scene_mapping_;
    double voxel_size_;
    double position_threshold_;
    double size_threshold_;
    double time_window_;
    double confidence_threshold_;
    std::string camera_frame_;
    std::string world_frame_;
    std::string detection_topic_;
    std::string pointcloud_topic_;
    std::string scene_attribute_topic_;
    std::string occupancy_grid_topic_;
    bool input_in_world_frame_;  // 输入坐标是否为世界坐标系
    
    // Scene mapping参数
    double scene_map_resolution_;
    int scene_map_width_;
    int scene_map_height_;
    double scene_map_origin_x_;
    double scene_map_origin_y_;
    double scene_max_range_;
    double scene_min_range_;
    int scene_occlusion_threshold_;
};

#endif // SEMANTIC_MAPPING_NODE_H_

