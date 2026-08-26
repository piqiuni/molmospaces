/**
 * Voronoi Mapping Node
 * 订阅占据栅格地图，计算并发布Voronoi图和距离图
 */

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <set>
#include <map>
#include "voronoi_mapping.h"

class VoronoiMappingNode
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    // 订阅者和发布者
    ros::Subscriber map_sub_;
    ros::Publisher voronoi_map_pub_;
    ros::Publisher distance_map_pub_;
    ros::Publisher voronoi_marker_pub_;
    
    // Voronoi计算对象
    VoronoiMapping voronoi_;
    
    // 参数
    double update_rate_;
    bool visualize_voronoi_;
    bool publish_distance_map_;
    int prune_iterations_;
    int obstacle_distance_threshold_;
    
    // 地图缓存
    nav_msgs::OccupancyGrid last_map_;
    bool map_received_;
    
public:
    VoronoiMappingNode() : private_nh_("~"), map_received_(false)
    {
        // 读取参数
        private_nh_.param("update_rate", update_rate_, 2.0);
        private_nh_.param("visualize_voronoi", visualize_voronoi_, true);
        private_nh_.param("publish_distance_map", publish_distance_map_, true);
        private_nh_.param("prune_iterations", prune_iterations_, 0);
        private_nh_.param("obstacle_distance_threshold", obstacle_distance_threshold_, 1);
        
        // 设置Voronoi对象的参数
        voronoi_.setObstacleDistanceThreshold(obstacle_distance_threshold_);
        
        // 订阅占据栅格地图
        map_sub_ = nh_.subscribe("/struct_mapping/occ_map", 1, 
                                  &VoronoiMappingNode::mapCallback, this);
        
        // 发布Voronoi地图（作为OccupancyGrid，便于在RViz中查看）
        voronoi_map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/struct_mapping/voronoi_map", 1, true);
        
        // 发布距离图
        if (publish_distance_map_) {
            distance_map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/struct_mapping/distance_map", 1, true);
        }
        
        // 发布Voronoi骨架线可视化
        if (visualize_voronoi_) {
            voronoi_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/struct_mapping/voronoi_markers", 1, true);
        }
        
        ROS_INFO("Voronoi Mapping Node initialized");
        ROS_INFO("  - Update rate: %.1f Hz", update_rate_);
        ROS_INFO("  - Visualize voronoi: %s", visualize_voronoi_ ? "true" : "false");
        ROS_INFO("  - Publish distance map: %s", publish_distance_map_ ? "true" : "false");
        ROS_INFO("  - Prune iterations: %d", prune_iterations_);
        ROS_INFO("  - Obstacle distance threshold: %d grid cells", obstacle_distance_threshold_);
        ROS_INFO("Subscribed to: /struct_mapping/occ_map");
        ROS_INFO("Publishing to: /struct_mapping/voronoi_map, /struct_mapping/distance_map, /struct_mapping/voronoi_markers");
    }
    
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
    {
        ROS_INFO_ONCE("Received first occupancy grid map");
        
        int width = msg->info.width;
        int height = msg->info.height;
        
        if (width <= 0 || height <= 0) {
            ROS_WARN("Invalid map size: %dx%d", width, height);
            return;
        }
        
        // 转换OccupancyGrid到bool数组
        // OccupancyGrid: -1=未知, 0=自由, 100=占据
        bool** gridMap = new bool*[width];
        for (int x = 0; x < width; x++) {
            gridMap[x] = new bool[height];
            for (int y = 0; y < height; y++) {
                // 注意坐标转换：OccupancyGrid是行优先，需要转置
                int index = y * width + x;
                // 将占据(>50)和未知(<0)都视为障碍物
                gridMap[x][y] = (msg->data[index] > 50 || msg->data[index] < 0);
            }
        }
        
        // 计算Voronoi图
        ros::Time start = ros::Time::now();
        voronoi_.initializeMap(width, height, gridMap);
        voronoi_.update();
        
        // 裁剪优化
        if (prune_iterations_ > 0) {
            ROS_INFO("Applying %d prune iterations to simplify Voronoi diagram...", prune_iterations_);
            for (int i = 0; i < prune_iterations_; i++) {
                voronoi_.prune();
            }
        } else {
            ROS_DEBUG("Pruning disabled (prune_iterations = 0)");
        }
        
        ros::Duration compute_time = ros::Time::now() - start;
        ROS_INFO_THROTTLE(5.0, "Voronoi computation time: %.3f ms (prune_iterations: %d)", 
                         compute_time.toSec() * 1000.0, prune_iterations_);
        
        // 发布结果
        publishVoronoiMap(msg, width, height);
        
        if (publish_distance_map_) {
            publishDistanceMap(msg, width, height);
        }
        
        if (visualize_voronoi_) {
            publishVoronoiMarkers(msg, width, height);
        }
        
        // 清理内存
        for (int x = 0; x < width; x++) {
            delete[] gridMap[x];
        }
        delete[] gridMap;
        
        // 保存地图
        last_map_ = *msg;
        map_received_ = true;
    }
    
    void publishVoronoiMap(const nav_msgs::OccupancyGrid::ConstPtr& original_map, 
                           int width, int height)
    {
        nav_msgs::OccupancyGrid voronoi_map;
        voronoi_map.header = original_map->header;
        voronoi_map.info = original_map->info;
        voronoi_map.data.resize(width * height);
        
        // 填充数据
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int index = y * width + x;
                
                if (voronoi_.isOccupied(x, y)) {
                    // 障碍物
                    voronoi_map.data[index] = 100;
                } else if (voronoi_.isVoronoi(x, y)) {
                    // Voronoi骨架线
                    voronoi_map.data[index] = 50;
                } else {
                    // 自由空间
                    voronoi_map.data[index] = 0;
                }
            }
        }
        
        voronoi_map_pub_.publish(voronoi_map);
        ROS_DEBUG("Published Voronoi map");
    }
    
    void publishDistanceMap(const nav_msgs::OccupancyGrid::ConstPtr& original_map, 
                           int width, int height)
    {
        nav_msgs::OccupancyGrid distance_map;
        distance_map.header = original_map->header;
        distance_map.info = original_map->info;
        distance_map.data.resize(width * height);
        
        // 找到最大距离用于归一化
        float max_dist = 0.0;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                if (!voronoi_.isOccupied(x, y)) {
                    float dist = voronoi_.getDistance(x, y);
                    if (dist > max_dist && dist != INFINITY) {
                        max_dist = dist;
                    }
                }
            }
        }
        
        // 填充距离数据（归一化到0-100）
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int index = y * width + x;
                
                if (voronoi_.isOccupied(x, y)) {
                    distance_map.data[index] = 100;  // 障碍物
                } else {
                    float dist = voronoi_.getDistance(x, y);
                    if (dist == INFINITY || max_dist == 0.0) {
                        distance_map.data[index] = 0;
                    } else {
                        // 距离越大，值越小（代价越低）
                        int value = 100 - static_cast<int>((dist / max_dist) * 100.0);
                        distance_map.data[index] = std::max(0, std::min(100, value));
                    }
                }
            }
        }
        
        distance_map_pub_.publish(distance_map);
        ROS_DEBUG("Published distance map (max_dist: %.2f)", max_dist);
    }
    
    void publishVoronoiMarkers(const nav_msgs::OccupancyGrid::ConstPtr& original_map, 
                              int width, int height)
    {
        visualization_msgs::MarkerArray marker_array;
        
        double resolution = original_map->info.resolution;
        double origin_x = original_map->info.origin.position.x;
        double origin_y = original_map->info.origin.position.y;
        
        // ========== 1. 创建Voronoi节点（球体点） ==========
        visualization_msgs::Marker nodes_marker;
        nodes_marker.header = original_map->header;
        nodes_marker.ns = "voronoi_nodes";
        nodes_marker.id = 0;
        nodes_marker.type = visualization_msgs::Marker::SPHERE_LIST;
        nodes_marker.action = visualization_msgs::Marker::ADD;
        nodes_marker.pose.orientation.w = 1.0;
        
        // 节点样式：蓝色小球
        nodes_marker.scale.x = resolution * 1.5;  // 球体直径
        nodes_marker.scale.y = resolution * 1.5;
        nodes_marker.scale.z = resolution * 1.5;
        nodes_marker.color.r = 0.0;
        nodes_marker.color.g = 0.5;
        nodes_marker.color.b = 1.0;  // 蓝色
        nodes_marker.color.a = 0.8;
        
        // ========== 2. 创建Voronoi边（线段） ==========
        visualization_msgs::Marker edges_marker;
        edges_marker.header = original_map->header;
        edges_marker.ns = "voronoi_edges";
        edges_marker.id = 1;
        edges_marker.type = visualization_msgs::Marker::LINE_LIST;
        edges_marker.action = visualization_msgs::Marker::ADD;
        edges_marker.pose.orientation.w = 1.0;
        
        // 线条样式：红色线
        edges_marker.scale.x = resolution * 0.5;  // 线宽
        edges_marker.color.r = 1.0;  // 红色
        edges_marker.color.g = 0.0;
        edges_marker.color.b = 0.0;
        edges_marker.color.a = 0.9;
        
        // ========== 3. 收集所有Voronoi点和边 ==========
        std::set<std::pair<int, int>> visited_edges;  // 避免重复边
        
        // 第一遍：先统计每个Voronoi点的邻居数量
        std::map<std::pair<int, int>, int> neighbor_count;
        
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                if (voronoi_.isVoronoi(x, y)) {
                    int count = 0;
                    // 统计4邻域内的Voronoi邻居数量
                    int dir4[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
                    for (int i = 0; i < 4; i++) {
                        int nx = x + dir4[i][0];
                        int ny = y + dir4[i][1];
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            if (voronoi_.isVoronoi(nx, ny)) {
                                count++;
                            }
                        }
                    }
                    neighbor_count[std::make_pair(x, y)] = count;
                }
            }
        }
        
        // 第二遍：根据邻居数量决定是否显示节点，并绘制边
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                if (voronoi_.isVoronoi(x, y)) {
                    auto coord = std::make_pair(x, y);
                    int neighbors = neighbor_count[coord];
                    
                    // 只在端点（邻居=1）或分支点（邻居≥3）显示球体
                    // 普通路径点（邻居=2）不显示
                    if (neighbors != 2) {
                        geometry_msgs::Point node_point;
                        node_point.x = origin_x + (x + 0.5) * resolution;
                        node_point.y = origin_y + (y + 0.5) * resolution;
                        node_point.z = 0.15;
                        nodes_marker.points.push_back(node_point);
                    }
                    
                    // 添加到相邻Voronoi点的边
                    // 策略：优先连接4邻域（上下左右），避免不必要的对角线连接
                    // 先连接4邻域
                    int dir4[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};  // 左右上下
                    for (int i = 0; i < 4; i++) {
                        int nx = x + dir4[i][0];
                        int ny = y + dir4[i][1];
                        
                        // 边界检查
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            if (voronoi_.isVoronoi(nx, ny)) {
                                // 避免重复边：只添加一次
                                int x1 = std::min(x, nx);
                                int y1 = std::min(y, ny);
                                int x2 = std::max(x, nx);
                                int y2 = std::max(y, ny);
                                
                                auto edge = std::make_pair(x1 * 10000 + y1, x2 * 10000 + y2);
                                if (visited_edges.find(edge) == visited_edges.end()) {
                                    visited_edges.insert(edge);
                                    
                                    geometry_msgs::Point p1, p2;
                                    p1.x = origin_x + (x + 0.5) * resolution;
                                    p1.y = origin_y + (y + 0.5) * resolution;
                                    p1.z = 0.15;
                                    
                                    p2.x = origin_x + (nx + 0.5) * resolution;
                                    p2.y = origin_y + (ny + 0.5) * resolution;
                                    p2.z = 0.15;
                                    
                                    edges_marker.points.push_back(p1);
                                    edges_marker.points.push_back(p2);
                                }
                            }
                        }
                    }
                    
                    // 连接对角线，但需要更严格的检查以避免三角形
                    // 策略：只有当对角线连接能形成"矩形"时才连接（两个相邻方向都有Voronoi）
                    // 这样可以避免在T形节点时连接不必要的对角线（如1-5-3这种情况）
                    int diag_dirs[4][2] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};  // 四个对角线方向
                    
                    for (int i = 0; i < 4; i++) {
                        int dx = diag_dirs[i][0];
                        int dy = diag_dirs[i][1];
                        int nx = x + dx;
                        int ny = y + dy;
                        
                        // 边界检查
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            if (voronoi_.isVoronoi(nx, ny)) {
                                // 检查对角线对应的两个相邻方向是否都是Voronoi点
                                // 例如：对于左上对角线(-1,-1)，需要检查左(-1,0)和上(0,-1)都是Voronoi
                                // 这样才能形成矩形而不是三角形
                                int check_dx1, check_dy1, check_dx2, check_dy2;
                                
                                if (dx == -1 && dy == -1) {      // 左上
                                    check_dx1 = -1; check_dy1 = 0;   // 左
                                    check_dx2 = 0;  check_dy2 = -1;  // 上
                                } else if (dx == -1 && dy == 1) { // 左下
                                    check_dx1 = -1; check_dy1 = 0;   // 左
                                    check_dx2 = 0;  check_dy2 = 1;   // 下
                                } else if (dx == 1 && dy == -1) {  // 右上
                                    check_dx1 = 1;  check_dy1 = 0;   // 右
                                    check_dx2 = 0;  check_dy2 = -1;  // 上
                                } else {  // 右下 (1, 1)
                                    check_dx1 = 1;  check_dy1 = 0;   // 右
                                    check_dx2 = 0;  check_dy2 = 1;   // 下
                                }
                                
                                // 检查两个相邻方向是否都是Voronoi点
                                bool adj1_is_voronoi = (x + check_dx1 >= 0 && x + check_dx1 < width &&
                                                         y + check_dy1 >= 0 && y + check_dy1 < height &&
                                                         voronoi_.isVoronoi(x + check_dx1, y + check_dy1));
                                bool adj2_is_voronoi = (x + check_dx2 >= 0 && x + check_dx2 < width &&
                                                         y + check_dy2 >= 0 && y + check_dy2 < height &&
                                                         voronoi_.isVoronoi(x + check_dx2, y + check_dy2));
                                
                                // 只有当两个相邻方向都有Voronoi点时才连接对角线
                                // 这样形成的是矩形（4个点都连），而不是三角形（3个点）
                                if (adj1_is_voronoi && adj2_is_voronoi) {
                                    // 避免重复边
                                    int x1 = std::min(x, nx);
                                    int y1 = std::min(y, ny);
                                    int x2 = std::max(x, nx);
                                    int y2 = std::max(y, ny);
                                    
                                    auto edge = std::make_pair(x1 * 10000 + y1, x2 * 10000 + y2);
                                    if (visited_edges.find(edge) == visited_edges.end()) {
                                        visited_edges.insert(edge);
                                        
                                        geometry_msgs::Point p1, p2;
                                        p1.x = origin_x + (x + 0.5) * resolution;
                                        p1.y = origin_y + (y + 0.5) * resolution;
                                        p1.z = 0.15;
                                        
                                        p2.x = origin_x + (nx + 0.5) * resolution;
                                        p2.y = origin_y + (ny + 0.5) * resolution;
                                        p2.z = 0.15;
                                        
                                        edges_marker.points.push_back(p1);
                                        edges_marker.points.push_back(p2);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // ========== 4. 发布Marker ==========
        if (!nodes_marker.points.empty()) {
            marker_array.markers.push_back(nodes_marker);
            ROS_DEBUG("Publishing %zu Voronoi nodes (blue spheres)", nodes_marker.points.size());
        }
        
        if (!edges_marker.points.empty()) {
            marker_array.markers.push_back(edges_marker);
            ROS_DEBUG("Publishing %zu Voronoi edges (red lines)", edges_marker.points.size() / 2);
        }
        
        if (!marker_array.markers.empty()) {
            voronoi_marker_pub_.publish(marker_array);
        }
    }
    
    void run()
    {
        ros::Rate rate(update_rate_);
        
        while (ros::ok()) {
            ros::spinOnce();
            
            // 如果有地图且需要定期更新，可以在这里重新发布
            // 目前是事件驱动，收到新地图才更新
            
            rate.sleep();
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "voronoi_mapping_node");
    
    VoronoiMappingNode node;
    node.run();
    
    return 0;
}

