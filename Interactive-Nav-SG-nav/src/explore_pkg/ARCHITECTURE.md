# Planner 和 Manager 交互关系说明

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                  ExploreManager                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  状态机管理                                        │  │
│  │  - IDLE → EXPLORING → APPROACHING → REACHED      │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ROS接口层                                        │  │
│  │  - 订阅: 图像、检测结果、地图、里程计                 │  │
│  │  - 发布: 导航目标、状态、可视化                      │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  数据流管理                                        │  │
│  │  - 检测结果处理                                    │  │
│  │  - 目标确认                                        │  │
│  │  - 导航调度                                        │  │
│  └──────────────────────────────────────────────────┘  │
│                          │                               │
│                          │ 调用接口                       │
│                          ▼                               │
│              ┌───────────────────────┐                  │
│              │ ExplorationPlanner    │                  │
│              │ (纯算法接口)           │                  │
│              │                        │                  │
│              │ selectNextGoal()      │                  │
│              │ updateDetections()    │                  │
│              └───────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

## 交互关系详解

### 1. **所有权关系**

**Manager 拥有 Planner**
```cpp
// explore_manager.h
std::unique_ptr<ExplorationPlanner> planner_;
```

- Manager 负责创建、初始化和销毁 Planner
- Planner 的生命周期由 Manager 管理
- Planner 不需要知道 Manager 的存在（单向依赖）

### 2. **初始化流程**

```
Manager::initialize()
    ├─> loadParameters()          // 加载算法类型参数
    ├─> setupSubscribers()        // 设置订阅者
    ├─> setupPublishers()         // 设置发布者
    └─> setupPlanner()            // 创建并初始化Planner
            ├─> 根据algorithm_type_创建对应Planner实例
            └─> planner_->initialize(nh_, private_nh_)
```

**代码位置**: `explore_manager.cpp:158-176`

```cpp
void ExploreManager::setupPlanner()
{
  // 根据配置创建对应类型的planner
  if (exploration_algorithm_type_ == "frontier") {
    planner_ = std::make_unique<FrontierExplorer>();
  } else if (exploration_algorithm_type_ == "rrt") {
    planner_ = std::make_unique<RRTExplorer>();
  }
  
  if (planner_) {
    planner_->initialize(nh_, private_nh_);
  }
}
```

### 3. **数据流向**

#### Manager → Planner（数据输入）

**a) 检测结果更新**
```cpp
// Manager接收到检测结果后，更新Planner
void ExploreManager::processDetections(const std::string& json_data)
{
  // ... 解析检测结果 ...
  
  // 更新planner的检测结果
  if (planner_) {
    planner_->updateDetections(current_detections_);
  }
}
```
**位置**: `explore_manager.cpp:376-378`

**数据内容**:
- `std::vector<DetectedObject>` - 当前帧的所有检测结果
- 包含：语义名称、置信度、3D坐标、环境状态

**b) 探索请求（调用核心接口）**
```cpp
void ExploreManager::executeExploration(const ros::TimerEvent& event)
{
  if (current_state_ != NavigationState::EXPLORING) {
    return;
  }
  
  // 调用planner选择下一个探索目标
  ExplorationGoal goal = planner_->selectNextGoal(
      current_map_,           // 当前占用栅格图
      current_pose_.pose,     // 机器人当前位置
      visited_points_         // 已访问的点列表
  );
  
  if (goal.is_valid) {
    // 发布探索目标
    goal_pub_.publish(createGoalPose(goal.position));
    visited_points_.push_back(goal.position);
  }
}
```
**位置**: `explore_manager.cpp:431-471`

**输入参数**:
- `nav_msgs::OccupancyGrid` - 地图数据
- `geometry_msgs::Pose` - 机器人当前位姿
- `std::vector<geometry_msgs::Point>` - 历史访问点

#### Planner → Manager（结果输出）

**返回值结构**:
```cpp
struct ExplorationGoal
{
  geometry_msgs::Point position;    // 目标位置
  double utility_score;              // 探索价值分数
  std::string reason;                // 选择原因（调试用）
  bool is_valid;                     // 目标是否有效
};
```

**Manager的使用**:
- 如果 `goal.is_valid == true`，发布导航目标
- 记录目标到 `visited_points_`，避免重复访问
- 记录日志和调试信息

### 4. **调用时机**

#### 定时调用
```
exploration_timer_ (1.0 Hz)
    └─> executeExploration()
            └─> planner_->selectNextGoal()
```

#### 事件驱动调用
```
检测结果回调
    └─> processDetections()
            └─> planner_->updateDetections()
```

### 5. **状态依赖**

**Planner只在EXPLORING状态下被调用**:
```cpp
void ExploreManager::executeExploration(const ros::TimerEvent& event)
{
  if (current_state_ != NavigationState::EXPLORING) {
    return;  // 只在探索状态下调用planner
  }
  
  // ... 调用planner ...
}
```

**状态转换**:
- `IDLE` → 等待目标描述，不调用planner
- `EXPLORING` → **调用planner选择探索目标**
- `APPROACHING` → 目标已找到，直接导航，不调用planner
- `REACHED` → 任务完成，不调用planner

### 6. **接口设计原则**

#### Manager 对 Planner 的要求
1. **纯算法接口** - Planner不关心ROS、状态机等高层逻辑
2. **无状态或最小状态** - Planner主要处理输入数据，返回结果
3. **可替换性** - 可以随时切换不同的Planner实现

#### Planner 对 Manager 的假设
1. **数据完整性** - Manager保证传入的地图、位姿等数据有效
2. **调用频率** - Planner会被定时调用（1Hz），需要高效实现
3. **线程安全** - Manager保证调用时的线程安全（通过mutex）

### 7. **错误处理**

```cpp
// Manager检查planner是否存在
if (!planner_) {
  ROS_DEBUG("Planner not implemented yet, skipping exploration");
  return;
}

// Planner返回无效目标时的处理
if (!goal.is_valid) {
  ROS_DEBUG("No valid exploration goal found");
  // Manager可以选择：
  // 1. 等待下次调用
  // 2. 切换探索策略
  // 3. 报告失败
}
```

## 数据流图

```
┌─────────────┐
│ ROS Topics  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐      ┌──────────────┐
│ ExploreManager    │      │  检测结果     │
│                   │◄─────┤  (JSON)      │
│  - 订阅消息        │      └──────────────┘
│  - 解析数据        │
│  - 状态管理        │
└──────┬─────────────┘
       │
       │ updateDetections()
       ▼
┌──────────────────┐
│ ExplorationPlanner│
│                   │
│  - 接收检测结果    │
│  - 接收地图数据    │
│  - 计算探索目标    │
└──────┬────────────┘
       │
       │ selectNextGoal()
       │ (返回ExplorationGoal)
       ▼
┌──────────────────┐      ┌──────────────┐
│ ExploreManager   │─────►│ 导航目标      │
│                   │      │ (PoseStamped)│
│  - 验证结果        │      └──────────────┘
│  - 发布目标        │
│  - 记录历史        │
└──────────────────┘
```

## 实现Planner的步骤

当你要实现具体的探索算法时：

1. **创建派生类**
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
        goal.is_valid = true;
        return goal;
    }
};
```

2. **在Manager中注册**
```cpp
// explore_manager.cpp:setupPlanner()
if (exploration_algorithm_type_ == "frontier") {
    planner_ = std::make_unique<FrontierExplorer>();
}
```

3. **使用检测数据（可选）**
```cpp
// Planner可以通过current_detections_访问检测结果
// 用于基于检测的探索策略
void FrontierExplorer::selectNextGoal(...) {
    // 可以使用this->current_detections_
    // 实现混合探索策略
}
```

## 总结

- **职责分离**: Manager负责协调和状态管理，Planner负责算法计算
- **单向依赖**: Planner不依赖Manager，便于独立测试和替换
- **数据驱动**: Manager通过接口传递数据，Planner返回结果
- **可扩展性**: 易于添加新的探索算法实现

