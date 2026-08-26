# Explore Package - 零样本目标导航管理器

## 概述

`explore_pkg` 是一个用于零样本目标导航的ROS包，采用分离架构设计：
- **ExploreManager**: 高层管理器，负责状态机、目标检测协调、导航调度
- **ExplorationPlanner**: 探索算法接口，具体算法后续实现

## 架构设计

```
explore_pkg/
├── include/explore_pkg/
│   ├── explore_manager.h          # 管理器头文件
│   └── exploration_planner.h      # 探索算法接口
├── src/
│   ├── explore_manager.cpp        # 管理器实现
│   ├── explore_manager_node.cpp   # 主节点
│   └── exploration_planner.cpp    # 算法接口实现（预留）
└── config/
    ├── explore_manager_params.yaml      # 管理器参数
    └── exploration_planner_params.yaml # 算法参数（预留）
```

## 功能特性

### ExploreManager
- ✅ 状态机管理（IDLE → EXPLORING → APPROACHING → REACHED → FAILED）
- ✅ 与GroundedSAM交互（发送检测请求、接收结果）
- ✅ 目标验证与确认（需要连续多次检测确认）
- ✅ 导航目标调度
- ✅ ROS接口（订阅/发布消息）
- ✅ 可视化支持

### ExplorationPlanner（接口已定义，待实现）
- ⏳ `selectNextGoal()` - 选择下一个探索目标
- ⏳ 支持多种算法类型（frontier, RRT, coverage等）

## 使用方法

### 1. 编译

```bash
cd ~/robot_ws/semantic_ws
catkin_make
source devel/setup.bash
```

### 2. 配置参数

编辑 `config/explore_manager_params.yaml` 设置：
- ROS话题名称
- 阈值参数
- 导航参数
- 功能开关

### 3. 运行节点

```bash
rosrun explore_pkg explore_manager_node
```

### 4. 设置目标

发布目标描述到话题：
```bash
rostopic pub /explore_manager/target_description std_msgs/String "data: 'red chair'"
```

### 5. 查看状态

```bash
# 查看当前状态
rostopic echo /explore_manager/state

# 查看详细状态（JSON格式）
rostopic echo /explore_manager/status
```

## ROS话题

### 订阅
- `/camera/rgb/image_raw` - RGB图像
- `/camera/rgb/camera_info` - 相机信息
- `/explore_agent/result_info` - GroundedSAM检测结果
- `/explore_manager/target_description` - 目标描述
- `/odometry` - 机器人里程计
- `/grid_mapping/occ_map` - 占用栅格图

### 发布
- `/move_base_simple/goal` - 导航目标
- `/explore_manager/state` - 当前状态
- `/explore_manager/status` - 详细状态（JSON）
- `/grounded_sam/detection_request` - 检测请求
- `/explore_manager/detection_markers` - 检测可视化标记
- `/explore_manager/target_marker` - 目标可视化标记

## 实现ExplorationPlanner

要实现具体的探索算法，需要：

1. 创建派生类继承 `ExplorationPlanner`
2. 实现 `selectNextGoal()` 方法
3. 在 `ExploreManager::setupPlanner()` 中实例化

示例：
```cpp
class FrontierExplorer : public ExplorationPlanner {
public:
    ExplorationGoal selectNextGoal(
        const nav_msgs::OccupancyGrid& map,
        const geometry_msgs::Pose& current_pose,
        const std::vector<geometry_msgs::Point>& visited_points) override {
        // 实现frontier探索算法
        ExplorationGoal goal;
        // ... 算法逻辑 ...
        return goal;
    }
};
```

## 参数说明

主要参数位于 `config/explore_manager_params.yaml`：

- `thresholds/target_reach`: 到达目标的距离阈值（米）
- `thresholds/detection_confidence`: 检测置信度阈值
- `thresholds/target_confirmation_count`: 目标确认需要的连续检测次数
- `navigation/max_goal_attempts`: 最大目标尝试次数
- `exploration/algorithm_type`: 探索算法类型（目前未实现）

## 依赖

- ROS (Melodic/Noetic)
- OpenCV
- RapidJSON
- actionlib
- move_base_msgs
- cv_bridge

## TODO

- [ ] 实现FrontierExplorer
- [ ] 实现RRTExplorer
- [ ] 添加服务接口
- [ ] 添加launch文件
- [ ] 单元测试

