#ifndef SCENE_SEMANTIC_MAP_H_
#define SCENE_SEMANTIC_MAP_H_

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <vector>
#include <string>
#include <map>

/**
 * @class SceneSemanticMap
 * @brief 场景语义地图的栅格层表示
 * 
 * 使用两张OccupancyGrid：
 * 1. scene_id_grid_: 存储场景ID（直接作为语义类别索引）
 *    - -1: 未知（未观测）
 *    - 0-9: 预定义语义类别索引，直接对应14通道格式的通道4-13
 *      * 0: livingroom (通道4)
 *      * 1: bedroom (通道5)
 *      * 2: kitchen (通道6)
 *      * 3: bathroom (通道7)
 *      * 4: balcony (通道8)
 *      * 5: storage (通道9)
 *      * 6: door (通道10)
 *      * 7: wall (通道11)
 *      * 8: entrance (通道12)
 *      * 9: outside (通道13)
 * 2. confidence_grid_: 存储置信度 (int8_t: 0-100表示0.0-1.0, -1=未知)
 */
class SceneSemanticMap
{
public:
    /**
     * @brief 构造函数
     * @param resolution 地图分辨率（米/像素），默认0.1m
     * @param width 地图宽度（像素）
     * @param height 地图高度（像素）
     * @param origin_x 地图原点x坐标（米）
     * @param origin_y 地图原点y坐标（米）
     */
    SceneSemanticMap(double resolution = 0.1, 
             unsigned int width = 1000, 
             unsigned int height = 1000,
             double origin_x = -50.0,
             double origin_y = -50.0);
    
    /**
     * @brief 从OccupancyGrid初始化（使用相同的尺寸和分辨率）
     * @param occ_grid 参考的OccupancyGrid
     */
    void initializeFromOccupancyGrid(const nav_msgs::OccupancyGrid& occ_grid);
    
    /**
     * @brief 析构函数
     */
    ~SceneSemanticMap();
    
    // ========== 栅格层操作 ==========
    
    /**
     * @brief 设置栅格cell的场景标签
     * @param x, y 世界坐标（米）
     * @param scene_id 场景ID (-1=未知, 0-9=语义类别索引，直接对应通道4-13)
     * @param scene_type 场景类型（可选，用于记录，不影响语义类别索引）
     * @param confidence 置信度参数（已废弃，BLIP模型不支持confidence）
     * 
     * 置信度更新机制（基于观测一致性）：
     * - 首次观测：设置初始置信度为1（对应0.01）
     * - 相同观测（scene_id相同）：置信度+1（上限100，对应1.0）
     * - 不同观测（scene_id不同）：置信度-1（下限0，对应0.0）
     */
    void setSceneLabel(double x, double y, int scene_id, 
                       const std::string& scene_type = "", 
                       float confidence = 1.0);
    
    /**
     * @brief 获取栅格cell的场景ID
     * @param x, y 世界坐标（米）
     * @return 场景ID，如果坐标超出范围返回-1
     */
    int getSceneId(double x, double y) const;
    
    /**
     * @brief 获取栅格cell的置信度
     * @param x, y 世界坐标（米）
     * @return 置信度 [0.0, 1.0]，如果坐标超出范围或未知返回-1.0
     */
    float getConfidence(double x, double y) const;
    
    /**
     * @brief 获取栅格cell的场景类型
     * @param x, y 世界坐标（米）
     * @return 场景类型字符串，如果不存在返回"unknown"
     */
    std::string getSceneType(double x, double y) const;
    
    /**
     * @brief 批量更新场景标签（从场景分割结果）
     * @param world_x, world_y 世界坐标列表
     * @param scene_ids 对应的场景ID列表
     * @param scene_types 对应的场景类型列表（可选）
     * @param confidences 对应的置信度列表（可选，默认1.0）
     */
    void updateSceneLabels(const std::vector<double>& world_x,
                          const std::vector<double>& world_y,
                          const std::vector<int>& scene_ids,
                          const std::vector<std::string>& scene_types = {},
                          const std::vector<float>& confidences = {});
    
    /**
     * @brief 根据场景属性字符串获取对应的语义类别索引（scene_id）
     * @param scene_attribute 场景属性字符串
     * @return scene_id (0-9语义类别索引，直接对应通道4-13)，如果无法映射返回-1
     */
    int getOrCreateSceneId(const std::string& scene_attribute);
    
    /**
     * @brief 初始化场景类型映射（从ROS参数加载配置）
     * @param nh ROS节点句柄
     */
    void initializeSceneTypeMapping(ros::NodeHandle& nh);
    
    /**
     * @brief 标准化场景类型字符串
     * @param raw_type 原始场景类型字符串
     * @return 标准化后的场景类型字符串
     */
    std::string normalizeSceneType(const std::string& raw_type) const;
    
    // ========== 场景类型映射操作 ==========
    
    /**
     * @brief 注册场景ID到类型的映射
     * @param scene_id 场景ID
     * @param scene_type 场景类型
     */
    void registerSceneType(int scene_id, const std::string& scene_type);
    
    /**
     * @brief 根据场景ID获取场景类型
     * @param scene_id 场景ID
     * @return 场景类型字符串，如果不存在返回"unknown"
     */
    std::string getSceneTypeById(int scene_id) const;
    
    /**
     * @brief 获取所有已注册的场景类型映射
     * @return map<scene_id, scene_type>
     */
    std::map<int, std::string> getAllSceneTypes() const { return scene_id_to_type_; }
    
    // ========== 查询和统计 ==========
    
    /**
     * @brief 获取指定场景ID的所有栅格cell数量
     * @param scene_id 场景ID
     * @return cell数量
     */
    int getCellCountBySceneId(int scene_id) const;
    
    /**
     * @brief 获取指定场景类型的所有栅格cell数量
     * @param scene_type 场景类型
     * @return cell数量
     */
    int getCellCountBySceneType(const std::string& scene_type) const;
    
    /**
     * @brief 获取地图信息
     */
    double getResolution() const { return resolution_; }
    unsigned int getWidth() const { return width_; }
    unsigned int getHeight() const { return height_; }
    double getOriginX() const { return origin_x_; }
    double getOriginY() const { return origin_y_; }
    
    /**
     * @brief 检查坐标是否在地图范围内
     */
    bool isValidCoordinate(double x, double y) const;
    
    /**
     * @brief 获取已知区域的边界框（所有非未知cell的边界）
     * @param min_x 输出：最小x坐标（米）
     * @param min_y 输出：最小y坐标（米）
     * @param max_x 输出：最大x坐标（米）
     * @param max_y 输出：最大y坐标（米）
     * @return 如果找到已知区域返回true，否则返回false
     */
    bool getKnownRegionBounds(double& min_x, double& min_y, double& max_x, double& max_y) const;
    
    /**
     * @brief 清空地图（重置所有cell为未知状态）
     */
    void clear();
    
    // ========== 获取OccupancyGrid ==========
    
    /**
     * @brief 获取场景ID的OccupancyGrid（用于发布）
     * @param frame_id 坐标系名称
     * @return OccupancyGrid，data[i] = scene_id
     */
    nav_msgs::OccupancyGrid getSceneIdGrid(const std::string& frame_id = "map") const;
    
    /**
     * @brief 获取置信度的OccupancyGrid（用于发布）
     * @param frame_id 坐标系名称
     * @return OccupancyGrid，data[i] = confidence * 100 (0-100)
     */
    nav_msgs::OccupancyGrid getConfidenceGrid(const std::string& frame_id = "map") const;
    
    /**
     * @brief 直接获取scene_id_grid_的引用（用于高效更新）
     */
    nav_msgs::OccupancyGrid& getSceneIdGridRef() { return scene_id_grid_; }
    const nav_msgs::OccupancyGrid& getSceneIdGridRef() const { return scene_id_grid_; }
    
    /**
     * @brief 直接获取confidence_grid_的引用（用于高效更新）
     */
    nav_msgs::OccupancyGrid& getConfidenceGridRef() { return confidence_grid_; }
    const nav_msgs::OccupancyGrid& getConfidenceGridRef() const { return confidence_grid_; }
    
    /**
     * @brief 生成彩色点云（用于可视化）
     * @param frame_id 坐标系名称
     * @param height 点云高度（米），默认0.1m
     * @return 彩色点云，每个点根据scene_id着色
     */
    sensor_msgs::PointCloud2 getColoredPointCloud(const std::string& frame_id = "map", double height = 0.1) const;
    
    /**
     * @brief 生成图例MarkerArray（显示每个颜色对应的场景属性）
     * @param frame_id 坐标系名称
     * @param position_x 图例位置x坐标（米），默认地图左下角
     * @param position_y 图例位置y坐标（米），默认地图左下角
     * @param position_z 图例位置z坐标（米），默认0.5m
     * @return MarkerArray，包含颜色块和文字标签
     */
    visualization_msgs::MarkerArray getLegendMarkerArray(const std::string& frame_id = "map", 
                                                          double position_x = 0.0, 
                                                          double position_y = 0.0, 
                                                          double position_z = 0.5) const;

private:
    // ========== 坐标转换 ==========
    
    /**
     * @brief 世界坐标转栅格坐标
     * @return 如果坐标有效返回true，否则返回false
     */
    bool worldToGrid(double x, double y, int& grid_x, int& grid_y) const;
    
    /**
     * @brief 栅格坐标转世界坐标（栅格中心）
     */
    void gridToWorld(int grid_x, int grid_y, double& x, double& y) const;
    
    /**
     * @brief 检查栅格坐标是否有效
     */
    bool isValidGrid(int grid_x, int grid_y) const;
    
    /**
     * @brief 计算栅格索引（一维数组索引）
     */
    size_t gridToIndex(int grid_x, int grid_y) const;
    
    /**
     * @brief 根据场景类型获取颜色（内部辅助方法）
     * @param normalized_scene_type 标准化后的场景类型字符串
     * @return ColorRGBA，如果不在预定义列表中返回 r=-1.0 表示无效
     */
    std_msgs::ColorRGBA getColorForSceneType(const std::string& normalized_scene_type) const;
    
    /**
     * @brief 根据场景ID获取颜色（内部辅助方法）
     * @param scene_id 场景ID
     * @return ColorRGBA
     */
    std_msgs::ColorRGBA getColorForSceneIdInternal(int scene_id) const;
    
    // ========== 成员变量 ==========
    
    // 地图参数
    double resolution_;                    // 分辨率（米/像素）
    unsigned int width_;                    // 地图宽度（像素）
    unsigned int height_;                   // 地图高度（像素）
    double origin_x_;                      // 地图原点x（米）
    double origin_y_;                      // 地图原点y（米）
    
    // 两张OccupancyGrid
    nav_msgs::OccupancyGrid scene_id_grid_;      // 场景ID栅格 (data[i] = scene_id)
    nav_msgs::OccupancyGrid confidence_grid_;    // 置信度栅格 (data[i] = confidence * 100)
    
    // 场景类型映射表
    std::map<int, std::string> scene_id_to_type_;  // scene_id -> scene_type
    
    // 场景类型标准化（从配置文件加载同义词映射）
    std::map<std::string, std::string> scene_type_synonyms_;  // 同义词映射：原始类型 -> 标准类型
    bool scene_type_mapping_initialized_;  // 是否已初始化映射
    
    //confidance const
    static const int8_t CONFIDENCE_UNKNOWN = -1;      // 置信度未知（用于confidence_grid_，符合OccupancyGrid标准）
    
    //scene id const
    static const int8_t SCENE_UNKNOWN = -1;           // 场景ID未知（用于scene_id_grid_，表示未观测）
};

#endif // SCENE_SEMANTIC_MAP_H_

