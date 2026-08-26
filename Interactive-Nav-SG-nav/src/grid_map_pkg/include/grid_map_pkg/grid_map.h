#ifndef GRID_MAP_H
#define GRID_MAP_H

#include <Eigen/Dense>
#include <string>
#include <nav_msgs/OccupancyGrid.h>
#include <opencv2/opencv.hpp>

class GridMap {
public:
    GridMap(const int &size_x, const int &size_y, const int &init_x, 
            const int &init_y, const double &cell_size, int &average_filter);
    
    // 坐标转换
    bool getIdx(const double &x, const double &y, Eigen::Vector2i &idx);
    
    // 概率操作
    bool getGridBel(const double &x, const double &y, double &bel);
    bool setGridBel(const double &x, const double &y, const double &bel);
    bool getGridLogBel(const double &x, const double &y, double &log_bel);
    bool setGridLogBel(const double &x, const double &y, const double &log_bel);
    
    // 工具函数
    double getCellSize();
    cv::Mat toCvMat();
    
    // 地图转换
    void toRosOccGridMap(const std::string &frame_id, nav_msgs::OccupancyGrid &occ_grid);
    void toRosFroGridMap(const std::string &frame_id, nav_msgs::OccupancyGrid &current_map, 
                         nav_msgs::OccupancyGrid &fro_grid);
    void toRosRoomGridMap(const std::string &frame_id, nav_msgs::OccupancyGrid &occ_grid,
                          nav_msgs::OccupancyGrid &fro_grid, nav_msgs::OccupancyGrid &room_grid);
    
    // 地图保存
    void saveMap(const std::string &img_dir, const std::string &cfg_dir);
    
private:
    void applyAverageFilter(const nav_msgs::OccupancyGrid &input_map,
                           nav_msgs::OccupancyGrid &filtered_map, int kernel_size);
    
    // 地图数据
    Eigen::MatrixXd bel_data_;
    
    // 显示相关
    Eigen::MatrixXd m_one_;
    Eigen::MatrixXd m_show_;
    
    // 参数
    int size_x_, size_y_;
    int init_x_, init_y_;
    double cell_size_;
    int average_filter_;
};

#endif // GRID_MAP_H

