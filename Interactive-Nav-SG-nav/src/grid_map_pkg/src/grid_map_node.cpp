#include <iostream>
#include <ros/ros.h>
#include <tf/tf.h>
#include <nav_msgs/Odometry.h>
#include <grid_map_pkg/grid_map.h>
#include <grid_map_pkg/grid_mapper.h>
#include <vector>

/* 全局变量 */
GridMap *g_map;
GridMapper *g_gmapper;
ros::Publisher g_occ_map_puber;
ros::Publisher g_fro_map_puber;
ros::Publisher g_room_map_puber;
ros::NodeHandle* g_nh;  // 添加全局节点句柄
Pose2d g_robot_pose;
double altitude;
std::vector<double> altitude_vec;
std::vector<GridMap *> g_map_vec;
std::vector<GridMapper *> g_mapper_vec;

void odometryCallback(const nav_msgs::OdometryConstPtr &odom);
void laserCallback(const sensor_msgs::PointCloud2Ptr scan);

int map_sizex, map_sizey, map_initx, map_inity;
double map_cell_size;
Pose2d T_r_l;
double P_occ, P_free, P_prior;
int averagefilter = -1;
double relative_height_offset;
double z_threshold;

int main(int argc, char **argv) {
    // 初始化ROS
    ros::init(argc, argv, "GridMapping");
    ros::NodeHandle nh;
    g_nh = &nh;  // 设置全局节点句柄
    
    // 加载参数
    nh.getParam("/mapping/map/sizex", map_sizex);
    nh.getParam("/mapping/map/sizey", map_sizey);
    nh.getParam("/mapping/map/initx", map_initx);
    nh.getParam("/mapping/map/inity", map_inity);
    nh.getParam("/mapping/map/cell_size", map_cell_size);
    
    double x, y, theta;
    nh.getParam("/mapping/robot_laser/x", x);
    nh.getParam("/mapping/robot_laser/y", y);
    nh.getParam("/mapping/robot_laser/theta", theta);
    T_r_l = Pose2d(x, y, theta);
    
    nh.getParam("/mapping/sensor_model/P_occ", P_occ);
    nh.getParam("/mapping/sensor_model/P_free", P_free);
    nh.getParam("/mapping/sensor_model/P_prior", P_prior);
    
    // 加载高度过滤参数
    nh.param<double>("/multi_floor/relative_height_offset", relative_height_offset, 0.9);
    nh.param<double>("/multi_floor/z_threshold", z_threshold, 0.8);
    ROS_INFO("高度过滤参数: height_offset=%.2f米, threshold=±%.2f米", 
             relative_height_offset, z_threshold);
    ROS_INFO("保留高度范围: %.2f ~ %.2f 米", 
             relative_height_offset - z_threshold, 
             relative_height_offset + z_threshold);
    
    /* 地图保存地址 */
    std::string map_image_save_dir, map_config_save_dir;
    nh.getParam("/mapping/map_image_save_dir", map_image_save_dir);
    nh.getParam("/mapping/map_config_save_dir", map_config_save_dir);
    ros::param::get("averagefilter", averagefilter);
    
    // 初始化地图和映射器
    ROS_INFO("初始化GridMap: size=(%d,%d), resolution=%.2f", map_sizex, map_sizey, map_cell_size);
    g_map = new GridMap(map_sizex, map_sizey, map_initx, map_inity, map_cell_size, averagefilter);
    g_gmapper = new GridMapper(g_map, T_r_l, P_occ, P_free, P_prior, nh);
    g_map_vec.push_back(g_map);
    g_mapper_vec.push_back(g_gmapper);
    
    ROS_INFO("地图初始化完成，等待数据...");
    
    // 初始化Topic
    ros::Subscriber g_odom_suber = nh.subscribe<nav_msgs::Odometry>("/odometry", 5, odometryCallback);
    ros::Subscriber laser_suber = nh.subscribe("/registered_scan", 1, laserCallback);
    g_occ_map_puber = nh.advertise<nav_msgs::OccupancyGrid>("grid_mapping/occ_map", 1);
    g_fro_map_puber = nh.advertise<nav_msgs::OccupancyGrid>("grid_mapping/frontier_map", 1);
    g_room_map_puber = nh.advertise<nav_msgs::OccupancyGrid>("grid_mapping/room_map", 1);
    
    ros::spin();
    
    return 0;
}

void odometryCallback(const nav_msgs::OdometryConstPtr &odom) {
    // 获取机器人姿态
    double x = odom->pose.pose.position.x;
    double y = odom->pose.pose.position.y;
    altitude = odom->pose.pose.position.z;
    double theta = tf::getYaw(odom->pose.pose.orientation);
    g_robot_pose = Pose2d(x, y, theta);
    
    bool is_floor_regist = false;
    
    // 检查是否是已注册的楼层
    for (auto &&al : altitude_vec) {
        if (abs(altitude - al) < 2.5) {  // 2.5米容差
            is_floor_regist = true;
            break;
        }
    }
    
    // 如果是新楼层，创建新地图
    if (!is_floor_regist || altitude_vec.size() == 0) {
        ROS_INFO("检测到新楼层，高度: %.2f", altitude);
        altitude_vec.push_back(altitude);
        g_map_vec.push_back(new GridMap(map_sizex, map_sizey, map_initx, map_inity, 
                                        map_cell_size, averagefilter));
        g_mapper_vec.push_back(new GridMapper(g_map_vec.back(), T_r_l, P_occ, P_free, P_prior, *g_nh));
    }
}

void laserCallback(const sensor_msgs::PointCloud2Ptr scan) {
    ROS_INFO("收到点云，点数: %d", scan->width * scan->height);
    // 遍历所有楼层，更新匹配的楼层地图
    for (size_t i = 0; i < altitude_vec.size(); i++) {
        if (abs(altitude - altitude_vec[i]) < 2.5) {
            // 更新地图（使用配置的高度过滤参数）
            g_mapper_vec[i]->updateMap(scan, g_robot_pose, altitude_vec[i], 
                                       z_threshold, relative_height_offset);
            
            // 发布三种地图
            nav_msgs::OccupancyGrid occ_map_floor;
            nav_msgs::OccupancyGrid fro_map_floor;
            nav_msgs::OccupancyGrid room_map_floor;
            
            g_map_vec[i]->toRosOccGridMap("map", occ_map_floor);
            g_map_vec[i]->toRosFroGridMap("map", occ_map_floor, fro_map_floor);
            g_map_vec[i]->toRosRoomGridMap("map", occ_map_floor, fro_map_floor, room_map_floor);
            
            g_occ_map_puber.publish(occ_map_floor);
            g_fro_map_puber.publish(fro_map_floor);
            g_room_map_puber.publish(room_map_floor);
            
            break;  // 只更新一个楼层
        }
    }
}

