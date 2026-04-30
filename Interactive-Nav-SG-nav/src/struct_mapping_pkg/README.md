# Struct Mapping Package - 结构建图包

## 概述

`struct_mapping_pkg` 是一个基于 OpenSlam's Gmapping 的 ROS 包，提供激光雷达 SLAM（Simultaneous Localization and Mapping）功能。它可以从激光扫描和位姿数据创建 2D 占用栅格地图（类似建筑平面图）。此外，还包含 Voronoi 图建图功能。

## 架构设计

```
struct_mapping_pkg/
├── include/
│   ├── slam_gmapping.h          # GMapping SLAM 头文件
│   └── voronoi_mapping.h        # Voronoi 建图头文件
├── src/
│   ├── slam_gmapping.cpp        # GMapping SLAM 实现
│   ├── main.cpp                 # SLAM 主程序
│   ├── nodelet.cpp              # Nodelet 实现
│   ├── replay.cpp               # 回放功能
│   ├── voronoi_mapping.cpp      # Voronoi 建图实现
│   └── voronoi_mapping_node.cpp # Voronoi 建图节点
├── launch/
│   └── slam_gmapping.launch     # SLAM 启动文件
└── config/
    └── (配置文件)
```

## 功能特性

### SlamGMapping
- ✅ 基于激光扫描的 SLAM
- ✅ 2D 占用栅格地图构建
- ✅ 机器人位姿估计
- ✅ 地图更新和发布
- ✅ TF 变换发布（map → odom）
- ✅ 支持点云输入（PointCloud2）
- ✅ 高度滤波点云发布

### VoronoiMapping
- ✅ Voronoi 图构建
- ✅ 距离地图计算
- ✅ 障碍物距离阈值设置
- ✅ Voronoi 图剪枝
- ✅ 可视化输出

### 节点类型
- ✅ `slam_gmapping` - 独立 SLAM 节点
- ✅ `slam_gmapping_nodelet` - Nodelet 版本
- ✅ `slam_gmapping_replay` - 回放模式
- ✅ `voronoi_mapping_node` - Voronoi 建图节点

## 使用方法

### 1. 编译

```bash
cd ~/Downloads/Interactive-Nav-SG-nav
catkin_make
source devel/setup.bash
```

### 2. 运行 SLAM 节点

```bash
# 使用 launch 文件启动
roslaunch struct_mapping_pkg slam_gmapping.launch

# 或直接运行节点
rosrun struct_mapping_pkg slam_gmapping
```

### 3. 运行 Voronoi 建图节点

```bash
rosrun struct_mapping_pkg voronoi_mapping_node
```

### 4. 保存地图

```bash
# 保存地图
rosrun map_server map_saver -f my_map map:=/dynamic_map

# 或使用自定义话题
rosrun map_server map_saver -f my_map map:=/struct_mapping/map
```

## ROS 话题

### 订阅
- `/scan` - 激光扫描数据（sensor_msgs/LaserScan）
- `/cloud` - 点云数据（sensor_msgs/PointCloud2）
- `/tf` - TF 变换树

### 发布
- `/map` - 占用栅格地图（nav_msgs/OccupancyGrid）
- `/map_metadata` - 地图元数据（nav_msgs/MapMetaData）
- `/entropy` - 地图熵（std_msgs/Float64）
- `/filtered_cloud` - 高度滤波后的点云（sensor_msgs/PointCloud2）
- `/tf` - TF 变换（map → odom）

## TF 坐标系

- `tf_frame_map` - 地图坐标系（固定）
- `tf_frame_odom` - 里程计坐标系
- `tf_frame_base_link` - 机器人基座坐标系
- `laser_frame` - 激光雷达坐标系

## 参数说明

主要参数可通过 ROS 参数服务器设置：

### GMapping 参数
- `map_update_interval`: 地图更新间隔（秒）
- `maxUrange`: 最大激光范围（米）
- `maxRange`: 最大有效范围（米）
- `sigma`: 激光噪声标准差
- `kernelSize`: 核大小
- `lstep`: 平移步长
- `astep`: 旋转步长
- `iterations`: 迭代次数
- `lsigma`: 激光标准差
- `ogain`: 占用增益
- `lskip`: 跳过的扫描数
- `minimumScore`: 最小分数阈值
- `srr`: 平移误差
- `srt`: 旋转误差
- `str`: 平移旋转误差
- `stt`: 旋转旋转误差
- `linearUpdate`: 线性更新阈值
- `angularUpdate`: 角度更新阈值
- `temporalUpdate`: 时间更新阈值
- `resampleThreshold`: 重采样阈值
- `particles`: 粒子数
- `xmin/xmax/ymin/ymax`: 地图边界（米）
- `delta`: 地图分辨率（米）
- `occ_thresh`: 占用阈值
- `llsamplerange`: 激光采样范围
- `llsamplestep`: 激光采样步长
- `lasamplerange`: 角度采样范围
- `lasamplestep`: 角度采样步长

### Voronoi 参数
- `obstacle_distance_threshold`: 障碍物距离阈值（网格单元）

## 依赖

- ROS Noetic
- OpenSlam Gmapping
- Boost
- TF
- sensor_msgs
- nav_msgs
- geometry_msgs

## 注意事项

1. **激光数据**: 确保激光扫描话题正确配置
2. **TF 变换**: 确保激光雷达坐标系和机器人基座坐标系的 TF 变换可用
3. **地图分辨率**: 根据应用场景调整地图分辨率和边界
4. **粒子数**: 增加粒子数可以提高建图质量，但会消耗更多计算资源
5. **坐标系命名**: 确保与其他包（如 `ai2thor_pkg`）的坐标系命名一致

## 使用场景

### 实时建图
适用于机器人在未知环境中实时构建地图的场景。

### 回放建图
使用 `slam_gmapping_replay` 可以从录制的 bag 文件重建地图。

### Voronoi 图应用
Voronoi 图可用于：
- 路径规划（沿着 Voronoi 边）
- 探索规划（选择 Voronoi 点作为目标）
- 障碍物避让

## 故障排除

### 常见问题

1. **地图不更新**: 检查激光数据是否正常发布
2. **TF 错误**: 确保 TF 变换树完整
3. **建图质量差**: 调整粒子数和更新参数
4. **内存占用高**: 减小地图范围或分辨率

## TODO

- [ ] 添加更多 SLAM 算法支持
- [ ] 优化内存使用
- [ ] 添加地图保存和加载功能
- [ ] 支持多机器人建图
- [ ] 性能优化
