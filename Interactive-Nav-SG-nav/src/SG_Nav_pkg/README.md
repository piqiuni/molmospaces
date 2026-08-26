# SG-Nav Package - 场景图导航包

## 概述

`SG_Nav_pkg` 是一个基于场景图（Scene Graph）的语义导航 ROS 包。它使用场景图来表示环境中的物体和房间关系，并基于此进行导航决策。该包集成了 GroundingDINO、SAM、GLIP 等视觉模型，以及 FMM（Fast Marching Method）路径规划算法。

## 架构设计

```
SG_Nav_pkg/
├── scenegraph.py              # 场景图核心模块
├── ai2SG.py                   # ROS 节点（场景图构建）
├── SG_Nav.py                  # Habitat 导航主程序
└── utils/
    ├── utils_fmm/             # FMM 路径规划工具
    │   ├── fmm_planner.py     # FMM 规划器
    │   ├── mapping.py          # 语义建图
    │   ├── control_helper.py  # 控制辅助函数
    │   └── pose_utils.py       # 位姿工具
    ├── utils_scenegraph/       # 场景图工具
    │   ├── grounded_sam_demo.py  # GroundedSAM 集成
    │   ├── mapping.py          # 空间映射
    │   ├── slam_classes.py     # SLAM 类定义
    │   └── utils.py            # 工具函数
    ├── utils_glip.py          # GLIP 工具
    └── image_process.py       # 图像处理工具
```

## 功能特性

### SceneGraph
- ✅ 场景图构建和维护
- ✅ 物体节点管理（ObjectNode）
- ✅ 房间节点管理（RoomNode）
- ✅ 组节点管理（GroupNode）
- ✅ 边关系管理（Edge）
- ✅ 物体检测融合（基于 GroundingDINO + SAM）
- ✅ 场景识别（基于 Ollama LLM）
- ✅ 空间相似度计算
- ✅ 物体跟踪和更新

### ai2SG (ROS Node)
- ✅ 订阅 AI2-THOR 传感器数据
- ✅ 实时场景图构建
- ✅ 发布场景图可视化标记
- ✅ 与 ROS 系统集成

### SG_Nav (Habitat Navigation)
- ✅ Habitat 环境集成
- ✅ 基于场景图的导航决策
- ✅ FMM 路径规划
- ✅ 目标检测和跟踪
- ✅ 房间级导航
- ✅ 物体级导航

### FMM Planner
- ✅ Fast Marching Method 路径规划
- ✅ 障碍物避让
- ✅ 多目标路径规划

## 使用方法

### 1. 环境要求

需要 `SG_Nav` conda 环境：

```bash
conda activate SG_Nav
```

### 2. 依赖安装

确保已安装以下依赖：
- PyTorch
- GroundingDINO
- Segment Anything (SAM)
- GLIP
- Ollama
- Habitat-Lab
- Open3D
- scikit-fmm

### 3. 模型下载

确保以下模型文件存在：
- `data/models/groundingdino_swint_ogc.pth` - GroundingDINO 模型
- `data/models/sam_vit_h_4b8939.pth` - SAM 模型
- `GLIP/MODEL/glip_large_model.pth` - GLIP 模型
- `data/models/bert-base-uncased/` - BERT 模型

### 4. 运行 ROS 节点（场景图构建）

```bash
conda activate SG_Nav
source /opt/ros/noetic/setup.bash
source ~/Downloads/Interactive-Nav-SG-nav/devel/setup.bash

# 确保 roscore 运行
roscore

# 在另一个终端运行
python src/SG_Nav_pkg/ai2SG.py
```

### 5. 运行 Habitat 导航

```bash
conda activate SG_Nav
cd src/SG_Nav_pkg

# 设置 Habitat 配置文件路径
export CHALLENGE_CONFIG_FILE="configs/challenge_objectnav2021.local.rgbd.yaml"

# 运行导航程序
python SG_Nav.py --visualize
```

## ROS 话题（ai2SG.py）

### 订阅
- `/ai2thor/habitat_obs` - AI2-THOR 观测数据（JSON）
- `/ai2thor/rgb/image_raw` - RGB 图像
- `/ai2thor/depth/image_raw` - 深度图像
- `/grid_mapping/occ_map` - 占用栅格图

### 发布
- `/scenegraph/markers` - 场景图可视化标记
- `/scenegraph/graph` - 场景图数据（JSON）
- `/scenegraph/objects` - 检测到的物体列表

## 配置参数

### SceneGraph 参数
- `sam_variant`: SAM 模型变体（`vit_h`, `vit_l`, `vit_b`）
- `device`: 计算设备（`cuda` 或 `cpu`）
- `obj_min_detections`: 物体最小检测次数
- `obj_min_confidence`: 物体最小置信度
- `bert_base_uncased_path`: BERT 模型路径

### SG_Nav 参数
- `visualize`: 是否可视化
- `split_l/split_r`: 数据集分割范围
- `distance_threshold`: 目标到达距离阈值

## 场景图结构

场景图由以下组件组成：

1. **ObjectNode**: 物体节点
   - 位置、尺寸、类别
   - 检测历史
   - 置信度

2. **RoomNode**: 房间节点
   - 房间类型（kitchen, bedroom 等）
   - 探索级别
   - 包含的物体节点

3. **GroupNode**: 组节点
   - 相关物体的分组
   - 相关性分数
   - 中心位置

4. **Edge**: 边关系
   - 物体-物体关系
   - 物体-房间关系
   - 关系类型和权重

## 依赖

- ROS Noetic
- Python 3.9+
- PyTorch
- GroundingDINO
- Segment Anything (SAM)
- GLIP
- Ollama
- Habitat-Lab
- Open3D
- scikit-fmm
- NumPy
- OpenCV
- Conda 环境：`SG_Nav`

## 注意事项

1. **环境要求**: 必须在 `SG_Nav` conda 环境中运行
2. **GPU**: 推荐使用 GPU 以获得更好的性能
3. **内存**: 场景图构建可能消耗较多内存
4. **模型路径**: 确保所有模型文件路径正确
5. **Ollama**: 确保 Ollama 服务运行（用于场景识别）

## 故障排除

### 常见错误

1. **ModuleNotFoundError**: 检查 conda 环境是否正确激活
2. **模型加载失败**: 检查模型文件路径和格式
3. **CUDA 错误**: 检查 GPU 驱动和 PyTorch CUDA 版本
4. **Ollama 连接失败**: 确保 Ollama 服务运行

## TODO

- [ ] 添加场景图持久化（保存/加载）
- [ ] 优化物体跟踪算法
- [ ] 添加更多关系类型
- [ ] 性能优化（多线程、批处理）
- [ ] 添加单元测试
