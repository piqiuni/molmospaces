# Semantic Mapping Package - 语义建图包

## 概述

`semantic_mapping_pkg` 是一个用于语义建图的 ROS 包，提供物体级和场景级语义地图构建功能。它使用 BLIP 模型进行场景识别，并基于点云数据构建 3D 语义地图。

## 架构设计

```
semantic_mapping_pkg/
├── include/semantic_mapping_pkg/
│   ├── semantic_mapping_node.h      # 主节点头文件
│   ├── object_semantic_map.h        # 物体语义地图
│   ├── object_semantic_mapper.h     # 物体建图器
│   ├── scene_semantic_map.h         # 场景语义地图
│   └── scene_semantic_mapper.h      # 场景建图器
├── src/
│   ├── semantic_mapping_node.cpp   # 主节点实现
│   ├── object_semantic_map.cpp      # 物体地图实现
│   ├── object_semantic_mapper.cpp   # 物体建图器实现
│   ├── scene_semantic_map.cpp       # 场景地图实现
│   └── scene_semantic_mapper.cpp    # 场景建图器实现
├── script/
│   ├── blip_ros_node_client.py      # BLIP 客户端节点
│   ├── blip_ros_node_mock.py        # BLIP 模拟节点（测试用）
│   └── scene_to_14channel_converter.py  # 场景到14通道转换器
├── launch/
│   ├── semantic_mapping.launch      # 语义建图启动文件
│   ├── blip_node.launch            # BLIP 节点启动文件
│   ├── scene_to_14channel.launch   # 14通道转换启动文件
│   ├── unified_system.launch       # 统一系统启动文件
│   └── rviz.launch                 # RViz 可视化启动文件
├── config/
│   ├── default.yaml                 # 默认配置文件
│   └── scene_type_mapping.yaml     # 场景类型映射配置
└── start_blip.sh                   # BLIP 服务器启动脚本
```

## 功能特性

### SemanticMappingNode
- ✅ 物体级语义建图（Object Semantic Mapping）
- ✅ 场景级语义建图（Scene Semantic Mapping）
- ✅ 基于点云的 3D 地图构建
- ✅ TF 变换监听和位姿获取
- ✅ 可视化标记发布
- ✅ JSON 格式地图导出

### ObjectSemanticMapper
- ✅ 物体检测结果融合
- ✅ 空间相似度计算
- ✅ 物体位置和尺寸估计
- ✅ 物体语义地图更新

### SceneSemanticMapper
- ✅ 场景属性识别（通过 BLIP）
- ✅ 场景语义地图构建
- ✅ 场景 ID 网格图生成
- ✅ 场景置信度网格图生成

### BLIP 集成
- ✅ 客户端模式（通过 HTTP 调用远程 BLIP 服务）
- ✅ 模拟模式（用于测试，不加载模型）
- ✅ 场景属性提取

## 使用方法

### 1. 编译

```bash
cd ~/Downloads/Interactive-Nav-SG-nav
catkin_make
source devel/setup.bash
```

### 2. 启动 BLIP 服务器（如果使用 client 模式）

```bash
cd src/semantic_mapping_pkg
bash start_blip.sh
```

### 3. 配置参数

编辑 `config/default.yaml` 设置：
- ROS 话题名称
- 建图参数（体素大小、阈值等）
- 功能开关（物体建图、场景建图）
- BLIP 服务地址（如果使用 client 模式）

### 4. 运行节点

```bash
# 单独启动语义建图节点
roslaunch semantic_mapping_pkg semantic_mapping.launch

# 或使用统一系统启动（包含其他包）
roslaunch semantic_mapping_pkg unified_system.launch

# 使用模拟模式（不需要 BLIP 服务器）
roslaunch semantic_mapping_pkg semantic_mapping.launch blip_mode:=mock
```

### 5. 场景到14通道转换

```bash
roslaunch semantic_mapping_pkg scene_to_14channel.launch
```

## ROS 话题

### 订阅
- `/grounded_sam/detection_result` - GroundedSAM 检测结果
- `/blip/scene_attribute` - BLIP 场景属性识别结果
- `/camera/depth/points` - 点云数据（PointCloud2）
- `/struct_mapping/occ_map` - 占用栅格图（用于14通道转换）

### 发布
- `/semantic_mapping/object_markers` - 物体可视化标记
- `/semantic_mapping/object_semantic_map` - 物体语义地图（JSON）
- `/semantic_mapping/scene_colored_pointcloud` - 场景彩色点云
- `/semantic_mapping/scene_legend` - 场景图例
- `/semantic_mapping/scene_id_grid` - 场景 ID 网格图
- `/semantic_mapping/scene_confidence_grid` - 场景置信度网格图

## 参数说明

主要参数位于 `config/default.yaml`：

- `enable_object_mapping`: 启用物体建图
- `enable_scene_mapping`: 启用场景建图
- `voxel_size`: 体素大小（米）
- `position_threshold`: 位置阈值（米）
- `size_threshold`: 尺寸阈值（米）
- `blip_mode`: BLIP 模式（`client` 或 `mock`）
- `blip_service_url`: BLIP 服务地址（client 模式）

## 依赖

- ROS Noetic
- PCL (Point Cloud Library)
- Eigen3
- RapidJSON
- yaml-cpp
- OpenCV
- BLIP（用于场景识别）
- TF2

## 注意事项

1. **BLIP 服务器**: 使用 `client` 模式时需要先启动 BLIP 服务器
2. **点云数据**: 确保点云话题正确配置，且包含有效的 TF 变换
3. **内存使用**: 大规模场景建图可能消耗较多内存
4. **TF 坐标系**: 确保相机坐标系和地图坐标系的 TF 变换可用

## 14通道格式转换

`scene_to_14channel_converter.py` 将场景语义地图转换为14通道格式（npz 文件），用于后续的导航算法。

转换参数：
- `occupancy_grid_topic`: 占用栅格图话题
- `scene_id_grid_topic`: 场景 ID 网格图话题
- `scene_confidence_grid_topic`: 场景置信度网格图话题
- `save_rate`: 保存频率（Hz）
- `save_dir`: 保存目录（空字符串使用默认路径）

## TODO

- [ ] 添加物体跟踪功能
- [ ] 优化内存使用（大规模场景）
- [ ] 添加地图保存和加载功能
- [ ] 支持动态物体过滤
- [ ] 性能优化（多线程处理）
