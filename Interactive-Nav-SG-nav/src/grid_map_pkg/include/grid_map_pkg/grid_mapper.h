#ifndef GRID_MAPPER_H
#define GRID_MAPPER_H

#include <grid_map_pkg/grid_map.h>
#include <grid_map_pkg/pose2d.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <Eigen/Dense>
#include <ros/ros.h>

class GridMapper {
public:
    GridMapper(GridMap *map, Pose2d &T_r_l, double &P_occ, double &P_free, double &P_prior, ros::NodeHandle& nh);
    
    // 地图更新
    void updateMap(sensor_msgs::PointCloud2Ptr cloud, Pose2d &robot_pose, 
                   double altitude, double z_threshold, double height_offset);
    
private:
    void updateGrid(const Eigen::Vector2d &grid, const double &pmzx);
    double laserInvModel(const double &r, const double &R, const double &cell_size);
    
    GridMap *map_;
    Pose2d T_r_l_;  // 机器人到激光的变换
    double P_occ_, P_free_, P_prior_;
    
    // 点云发布器
    ros::Publisher filtered_cloud_pub_;
};

#endif // GRID_MAPPER_H
