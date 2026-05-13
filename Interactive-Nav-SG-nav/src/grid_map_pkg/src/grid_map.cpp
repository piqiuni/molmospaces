#include <grid_map_pkg/grid_map.h>
#include <ros/ros.h>
#include <fstream>

GridMap::GridMap(const int &size_x, const int &size_y, const int &init_x, const int &init_y, 
                 const double &cell_size, int &average_filter) 
    : size_x_(size_x), size_y_(size_y), init_x_(init_x), init_y_(init_y), 
      cell_size_(cell_size), average_filter_(average_filter) {
    
    bel_data_.resize(size_x_, size_y_);
    bel_data_.setOnes() *= 0.5;  // 全部设为0.5的概率
    
    // 为opencv图片显示相关
    m_one_.resize(size_x_, size_y_);
    m_one_.setOnes();
    m_show_.resize(size_x_, size_y_);
    m_show_.setOnes() * 0.5;
}

bool GridMap::getIdx(const double &x, const double &y, Eigen::Vector2i &idx) {
    int xidx = cvFloor(x / cell_size_) + init_x_;
    int yidx = cvFloor(y / cell_size_) + init_y_;
    
    if ((xidx < 0) || (yidx < 0) || (xidx >= size_x_) || (yidx >= size_y_))
        return false;
    
    idx << xidx, yidx;
    return true;
}

bool GridMap::getGridBel(const double &x, const double &y, double &bel) {
    Eigen::Vector2i idx;
    if (!getIdx(x, y, idx))
        return false;
    bel = bel_data_(idx(0), idx(1));
    return true;
}

bool GridMap::setGridBel(const double &x, const double &y, const double &bel) {
    Eigen::Vector2i idx;
    if (!getIdx(x, y, idx))
        return false;
    bel_data_(idx(0), idx(1)) = bel;
    return true;
}

bool GridMap::getGridLogBel(const double &x, const double &y, double &log_bel) {
    double bel;
    if (!getGridBel(x, y, bel))
        return false;
    log_bel = log(bel / (1.0 - bel));
    return true;
}

bool GridMap::setGridLogBel(const double &x, const double &y, const double &log_bel) {
    double bel = 1.0 - 1.0 / (1 + exp(log_bel));
    if (!setGridBel(x, y, bel))
        return false;
    return true;
}

double GridMap::getCellSize() {
    return cell_size_;
}

cv::Mat GridMap::toCvMat() {
    m_show_ = m_one_ - bel_data_;
    cv::Mat map(cv::Size(size_x_, size_y_), CV_64FC1, m_show_.data(), cv::Mat::AUTO_STEP);
    cv::flip(map, map, 0);
    return map;
}

void GridMap::toRosOccGridMap(const std::string &frame_id, nav_msgs::OccupancyGrid &occ_grid) {
    occ_grid.header.frame_id = frame_id;
    occ_grid.header.stamp = ros::Time::now();
    
    occ_grid.info.width = size_x_;
    occ_grid.info.height = size_y_;
    occ_grid.info.resolution = cell_size_;
    occ_grid.info.origin.position.x = -init_x_ * cell_size_;
    occ_grid.info.origin.position.y = -init_y_ * cell_size_;
    
    const int N = size_x_ * size_y_;
    occ_grid.data.resize(N);
    
    for (size_t i = 0; i < N; i++) {
        double &value = bel_data_.data()[i];
        if (value == 0.5)
            occ_grid.data[i] = -1;
        else
            occ_grid.data[i] = value * 100;
    }
    
    // 应用平均滤波
    if (average_filter_ > 0) {
        nav_msgs::OccupancyGrid filtered_grid;
        applyAverageFilter(occ_grid, filtered_grid, average_filter_);
        occ_grid = filtered_grid;
    }
}

void GridMap::toRosFroGridMap(const std::string &frame_id, nav_msgs::OccupancyGrid &current_map, 
                              nav_msgs::OccupancyGrid &fro_grid) {
    const size_t map_size = current_map.data.size();
    fro_grid.data.resize(map_size, 0);
    
    fro_grid.header.stamp = ros::Time::now();
    fro_grid.header.frame_id = frame_id;
    fro_grid.info = current_map.info;
    
    const int width = current_map.info.width;
    const int neighbors[8] = {
        -width-1, -width, -width+1,
        -1,                +1,
        width-1,   width,  width+1
    };
    
    for (int y = 1; y < current_map.info.height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            const int index = y * width + x;
            const int8_t current_value = current_map.data[index];
            
            if (current_value == -1 || current_value > 30)
                continue;
            
            for (int i = 0; i < 8; i++) {
                if (current_map.data[index + neighbors[i]] == -1) {
                    fro_grid.data[index] = -2;
                    break;
                }
            }
        }
    }
}

void GridMap::toRosRoomGridMap(const std::string &frame_id, nav_msgs::OccupancyGrid &occ_grid,
                               nav_msgs::OccupancyGrid &fro_grid, nav_msgs::OccupancyGrid &room_grid) {
    room_grid = occ_grid;
    
    cv::Mat grayMap(size_y_, size_x_, CV_8UC1);
    for (int y = 0; y < size_y_; y++) {
        for (int x = 0; x < size_x_; x++) {
            int8_t occ_val = occ_grid.data[y * size_x_ + x];
            int8_t fro_val = fro_grid.data[y * size_x_ + x];
            grayMap.at<uchar>(y, x) = (occ_val > 50 || fro_val == -2) ? 255 : 0;
        }
    }
    
    int blockSize = 32;
    int overlap = 8;
    
    cv::Mat resultMap = cv::Mat::zeros(size_y_, size_x_, CV_8UC1);
    
    for (int blockY = 0; blockY < size_y_; blockY += blockSize - overlap) {
        for (int blockX = 0; blockX < size_x_; blockX += blockSize - overlap) {
            int endY = std::min(blockY + blockSize, size_y_);
            int endX = std::min(blockX + blockSize, size_x_);
            
            if (endX - blockX < 20 || endY - blockY < 20)
                continue;
            
            cv::Rect blockRect(blockX, blockY, endX - blockX, endY - blockY);
            cv::Mat blockImg = grayMap(blockRect);
            
            int nonZeroCount = cv::countNonZero(blockImg);
            if (nonZeroCount < 15)
                continue;
            
            std::vector<cv::Vec4i> lines;
            cv::HoughLinesP(blockImg, lines, 0.1, CV_PI/180, 5, 7, 5);
            
            for (const auto& line : lines) {
                cv::Point pt1(line[0] + blockX, line[1] + blockY);
                cv::Point pt2(line[2] + blockX, line[3] + blockY);
                cv::line(resultMap, pt1, pt2, cv::Scalar(127), 1);
            }
        }
    }
    
    for (int y = 0; y < size_y_; y++) {
        for (int x = 0; x < size_x_; x++) {
            if (resultMap.at<uchar>(y, x) == 127) {
                room_grid.data[y * size_x_ + x] = 100;
            }
        }
    }
}

void GridMap::saveMap(const std::string &img_dir, const std::string &cfg_dir) {
    cv::Mat img = toCvMat();
    img = img * 255;
    cv::imwrite(img_dir, img);
    
    std::ofstream file;
    file.open(cfg_dir);
    file << "map:" << std::endl
         << "  size_x: " << size_x_ << std::endl
         << "  size_y: " << size_y_ << std::endl
         << "  init_x: " << init_x_ << std::endl
         << "  init_y: " << init_y_ << std::endl
         << "  cell_size: " << cell_size_ << std::endl;
}

void GridMap::applyAverageFilter(const nav_msgs::OccupancyGrid &input_map,
                                 nav_msgs::OccupancyGrid &filtered_map, int kernel_size) {
    filtered_map.info = input_map.info;
    filtered_map.header = input_map.header;
    filtered_map.data.resize(input_map.data.size());
    
    const int width = input_map.info.width;
    const int height = input_map.info.height;
    const int half_kernel = kernel_size / 2;
    
    for (int y = half_kernel; y < height - half_kernel; ++y) {
        for (int x = half_kernel; x < width - half_kernel; ++x) {
            int sum = 0;
            int count = 0;
            
            for (int dy = -half_kernel; dy <= half_kernel; ++dy) {
                for (int dx = -half_kernel; dx <= half_kernel; ++dx) {
                    const int nx = x + dx;
                    const int ny = y + dy;
                    const int idx = ny * width + nx;
                    
                    if (input_map.data[idx] != -1) {
                        sum += input_map.data[idx];
                        count++;
                    }
                }
            }
            
            const int output_idx = y * width + x;
            filtered_map.data[output_idx] = count > 0 ? static_cast<int8_t>(sum / count) : -1;
        }
    }
}

