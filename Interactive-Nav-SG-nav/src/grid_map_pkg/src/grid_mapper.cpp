#include <grid_map_pkg/grid_mapper.h>
#include <cmath>

GridMapper::GridMapper(GridMap *map, Pose2d &T_r_l, double &P_occ, double &P_free, double &P_prior, ros::NodeHandle& nh)
    : map_(map), T_r_l_(T_r_l), P_occ_(P_occ), P_free_(P_free), P_prior_(P_prior) {
    // 初始化过滤后点云发布器
    filtered_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("filtered_pointcloud", 1);
}

void GridMapper::updateMap(sensor_msgs::PointCloud2Ptr cloud, Pose2d &robot_pose,
                           double altitude, double z_threshold, double height_offset) {
    static int point_count = 0;
    static double z_sum = 0;
    const double &cell_size = map_->getCellSize();
    
    // 创建过滤后的点云用于发布
    sensor_msgs::PointCloud2 filtered_cloud;
    filtered_cloud.header = cloud->header;
    filtered_cloud.header.frame_id = cloud->header.frame_id;
    filtered_cloud.height = 1;
    filtered_cloud.width = 0;  // 初始为0，后面会动态调整
    filtered_cloud.fields = cloud->fields;
    filtered_cloud.is_bigendian = cloud->is_bigendian;
    filtered_cloud.point_step = cloud->point_step;
    filtered_cloud.row_step = 0;
    filtered_cloud.data.clear();
    
    // 遍历点云中的每个点
    sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud, "z");
    
    int filtered_point_count = 0;
    
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
        // 相对高度过滤：只处理相对高度附近的点
        // 使用配置的height_offset（相机高度）和z_threshold（过滤阈值）
        if (std::abs(*iter_z-height_offset) > z_threshold)
            continue;
        
        // 将过滤后的点添加到发布点云中
        uint8_t* point_data = new uint8_t[cloud->point_step];
        memcpy(point_data, &(*iter_x) - cloud->point_step, cloud->point_step);
        filtered_cloud.data.insert(filtered_cloud.data.end(), point_data, point_data + cloud->point_step);
        delete[] point_data;
        filtered_point_count++;

        // 点云坐标已经是世界坐标，直接使用（不需要旋转）
        Eigen::Vector2d robot_world(robot_pose.x_, robot_pose.y_); // 机器人坐标系原点在世界坐标系下的坐标
        double robot_theta=robot_pose.theta_; // 机器人坐标系相对于世界坐标系的旋转角度
        //创建旋转矩阵
        Eigen::Matrix2d rotation_matrix;
        rotation_matrix << std::cos(robot_theta), -std::sin(robot_theta),
                         std::sin(robot_theta), std::cos(robot_theta);
        //将障碍物坐标转换到世界坐标系
        Eigen::Vector2d p_robot(*iter_x, *iter_y); // 障碍物在机器人坐标系下的坐标
        Eigen::Vector2d p_world = robot_world + rotation_matrix * p_robot;

        double R = (p_world - robot_world).norm();
        const double inc_step = 1.0 * cell_size;
        Eigen::Vector2d last_grid(Eigen::Infinity, Eigen::Infinity);
        
        for (double r = 0; r < R + cell_size; r += inc_step) {
            Eigen::Vector2d p_w_step = robot_world + (r / R) * (p_world - robot_world);
            
            if (p_w_step == last_grid)
                continue;
            
            updateGrid(p_w_step, laserInvModel(r, R, cell_size));
            last_grid = p_w_step;
            
            if (r > 15)
                break;
        }
    }
    
    // 更新过滤后点云的元数据
    filtered_cloud.width = filtered_point_count;
    filtered_cloud.row_step = filtered_cloud.point_step * filtered_cloud.width;
    
    // 发布过滤后的点云
    if (filtered_point_count > 0) {
        filtered_cloud_pub_.publish(filtered_cloud);
        ROS_INFO("发布过滤后点云: %d 个点 (原始: %d 个点)", 
                 filtered_point_count, cloud->width * cloud->height);
    }
}

void GridMapper::updateGrid(const Eigen::Vector2d &grid, const double &pmzx) {
    double log_bel;
    if (!map_->getGridLogBel(grid(0), grid(1), log_bel))
        return;
    
    log_bel += log(pmzx / (1.0 - pmzx));
    map_->setGridLogBel(grid(0), grid(1), log_bel * 1.05);  // 1.05倍增强
}

double GridMapper::laserInvModel(const double &r, const double &R, const double &cell_size) {
    if (r < (R - 0.5 * cell_size))
        return P_free_;
    
    if (r > (R + 0.5 * cell_size))
        return P_prior_;
    
    return P_occ_;
}
