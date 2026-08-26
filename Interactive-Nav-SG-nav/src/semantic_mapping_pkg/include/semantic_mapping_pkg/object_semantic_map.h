#ifndef OBJECT_SEMANTIC_MAP_H_
#define OBJECT_SEMANTIC_MAP_H_

#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <visualization_msgs/MarkerArray.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <Eigen/Dense>

/**
 * @brief 体素索引（用于哈希表key）
 */
struct VoxelIndex {
    int x, y, z;
    
    VoxelIndex(int x = 0, int y = 0, int z = 0) : x(x), y(y), z(z) {}
    
    bool operator==(const VoxelIndex& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

/**
 * @brief VoxelIndex的哈希函数
 */
struct VoxelIndexHash {
    size_t operator()(const VoxelIndex& idx) const {
        // 使用质数组合避免冲突
        return std::hash<int>()(idx.x) * 73856093 ^
               std::hash<int>()(idx.y) * 19349663 ^
               std::hash<int>()(idx.z) * 83492791;
    }
};

/**
 * @brief 语义体素数据结构
 */
struct SemanticVoxel {
    // 物体信息
    uint64_t object_id;              // 唯一物体ID
    geometry_msgs::Point position;   // 3D位置（体素中心）
    geometry_msgs::Vector3 size;     // 物体尺寸
    
    // 语义信息
    std::string semantic_class;      // 语义类别（如"chair", "table"）
    double confidence;               // 置信度 [0, 1]
    
    // 时间信息
    ros::Time first_observed;        // 首次观测时间
    ros::Time last_updated;          // 最后更新时间
    int observation_count;           // 观测次数
    
    // 构造函数
    SemanticVoxel() 
        : object_id(0)
        , confidence(0.0)
        , observation_count(0)
    {
        position.x = position.y = position.z = 0.0;
        size.x = size.y = size.z = 0.0;
        first_observed = ros::Time::now();
        last_updated = ros::Time::now();
    }
};

/**
 * @class ObjectSemanticMap
 * @brief 物体语义地图（3D稀疏栅格）
 * 
 * 使用哈希表存储3D体素，每个体素包含语义物体信息
 */
class ObjectSemanticMap
{
public:
    /**
     * @brief 构造函数
     * @param voxel_size 体素大小（米），默认0.1m
     */
    explicit ObjectSemanticMap(double voxel_size = 0.1)
        : voxel_size_(voxel_size)
    {
        ROS_INFO("ObjectSemanticMap initialized with voxel size: %.3f m", voxel_size_);
    }
    
    ~ObjectSemanticMap() = default;
    
    // ========== 基本操作 ==========
    
    /**
     * @brief 获取体素
     * @param x, y, z 世界坐标（米）
     * @return 体素指针，如果不存在返回nullptr
     */
    SemanticVoxel* getVoxel(double x, double y, double z);
    
    /**
     * @brief 检查体素是否存在
     */
    bool hasVoxel(double x, double y, double z) const;
    
    /**
     * @brief 设置体素
     * @param x, y, z 世界坐标（米）
     * @param voxel 体素数据
     */
    void setVoxel(double x, double y, double z, const SemanticVoxel& voxel);
    
    /**
     * @brief 更新体素（如果存在则更新，不存在则创建）
     * @return 体素指针
     */
    SemanticVoxel* updateVoxel(double x, double y, double z, const SemanticVoxel& voxel);
    
    // ========== 查询操作 ==========
    
    /**
     * @brief 获取指定半径内的所有体素
     * @param x, y, z 中心点坐标
     * @param radius 半径（米）
     * @return 体素指针列表
     */
    std::vector<SemanticVoxel*> getVoxelsInRadius(double x, double y, double z, double radius);
    
    /**
     * @brief 根据语义类别获取所有体素
     * @param semantic_class 语义类别名称
     * @return 体素指针列表
     */
    std::vector<SemanticVoxel*> getVoxelsByClass(const std::string& semantic_class);
    
    /**
     * @brief 根据物体ID获取体素
     * @param object_id 物体ID
     * @return 体素指针，如果不存在返回nullptr
     */
    SemanticVoxel* getVoxelByObjectId(uint64_t object_id);
    
    /**
     * @brief 获取所有体素
     * @return 体素指针列表
     */
    std::vector<SemanticVoxel*> getAllVoxels();
    
    // ========== 统计信息 ==========
    
    /**
     * @brief 获取地图中体素总数
     */
    size_t getVoxelCount() const { return voxel_map_.size(); }
    
    /**
     * @brief 获取体素大小
     */
    double getVoxelSize() const { return voxel_size_; }
    
    /**
     * @brief 清空地图
     */
    void clear();
    
    // ========== 可视化 ==========
    
    /**
     * @brief 转换为MarkerArray用于RViz可视化
     * @param frame_id 坐标系名称
     * @return MarkerArray
     */
    visualization_msgs::MarkerArray toMarkerArray(const std::string& frame_id = "map");

private:
    // ========== 坐标转换 ==========
    
    /**
     * @brief 世界坐标转体素索引
     */
    VoxelIndex worldToVoxel(double x, double y, double z) const;
    
    /**
     * @brief 体素索引转世界坐标（体素中心）
     */
    Eigen::Vector3d voxelToWorld(const VoxelIndex& idx) const;
    
    // ========== 成员变量 ==========
    
    std::unordered_map<VoxelIndex, SemanticVoxel, VoxelIndexHash> voxel_map_;  // 体素地图
    std::unordered_map<uint64_t, VoxelIndex> object_id_to_voxel_;             // 物体ID到体素索引的映射
    
    double voxel_size_;  // 体素大小（米）
    
    // 线程安全（可选，如果需要多线程访问）
    // mutable std::shared_mutex map_mutex_;  // 需要C++14
};

#endif // OBJECT_SEMANTIC_MAP_H_

