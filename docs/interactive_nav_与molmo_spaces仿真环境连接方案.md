# Interactive-Nav-SG-nav 与 molmo_spaces 仿真环境连接方案

本文给出一个可落地的“分层适配”方案：保留 `Interactive-Nav-SG-nav` 上层导航能力（探索 + move_base + 语义建图），将底层仿真执行从 AI2-THOR 替换/并行到 `molmo_spaces`。

## 1. 目标与原则

目标：让 `Interactive-Nav-SG-nav` 在不大改上层算法的前提下，能够消费 `molmo_spaces` 的观测并输出控制，再反馈给 `molmo_spaces` 执行动作。

原则：

- 上层不改：`explore_pkg`、`nav_pkg`、`semantic_mapping_pkg` 尽量不改
- 中层可替换：将 `ai2thor_pkg` 抽象为“仿真桥接层”
- 下层解耦：`molmo_spaces` 负责真实仿真状态与传感器渲染
- 契约优先：以 ROS topic/TF 协议作为模块边界

## 2. 现有接口契约（建议冻结）

建议将以下接口定义为“导航标准输入输出”：

输入到导航栈：

- `/odometry` (`nav_msgs/Odometry`)
- `/registered_scan` (`sensor_msgs/PointCloud2`)
- `/explore_agent/result_info` (`std_msgs/String`, JSON)
- `/ai2thor/rgb_image` (`sensor_msgs/Image`)（BLIP/调试使用）

导航输出：

- `/cmd_vel` -> `/cmd_vel_stamped` (`geometry_msgs/TwistStamped`)

可选接口：

- `/ai2thor/depth_image`、`/ai2thor/semantic_image`、`/ai2thor/habitat_obs`
- `/explore_agent/explore_target`

TF 最小链：

- `map -> odom -> base_link -> camera`

说明：目前 `Interactive-Nav-SG-nav` 使用 `tf_frame_map/tf_frame_odom/tf_frame_base_link/tf_frame_camera`。接入时可保留该命名，或统一 remap 到标准 `map/odom/base_link/camera`。

## 3. molmo_spaces 侧适配点

`molmo_spaces` 当前核心环境抽象位于 `molmo_spaces/env/env.py`，具备：

- 仿真步进：`reset()` / `step()`
- 渲染：`render_rgb_frame()` / `render_depth_frame()` / `render_segmentation_frame()`
- 可见性与目标相关工具（可用于检测/语义）

因此，推荐新增一个 ROS 桥接包，例如：

- `molmo_spaces_ros_bridge/`
  - `scripts/molmo_ros_bridge.py`
  - `config/bridge.yaml`
  - `launch/molmo_bridge.launch`

桥接职责：

- 从 `molmo_spaces` 拉取 RGB/Depth/Seg/位姿，发布 ROS 消息
- 从 `/cmd_vel_stamped` 读取速度指令，转换为 `molmo_spaces` 动作/控制
- 维护 TF 广播与坐标系一致性
- 产出兼容 `semantic_mapping_pkg` 的检测 JSON（先 mock，后接真实检测器）

## 4. 推荐架构（分层）

### 4.1 层级

- L0 仿真层：`molmo_spaces`（MuJoCo 场景、机器人、传感器）
- L1 适配层：`molmo_spaces_ros_bridge`（新建）
- L2 导航层：`struct_mapping_pkg` + `semantic_mapping_pkg` + `explore_pkg` + `nav_pkg`

### 4.2 数据流

1. `molmo_spaces_ros_bridge` 发布 `/odometry`、`/registered_scan`、`/ai2thor/rgb_image`。
2. 导航层按原流程建图、选点、规划，发布 `/cmd_vel`。
3. `nav_pkg/relay_node` 转发 `/cmd_vel_stamped`。
4. `molmo_spaces_ros_bridge` 消费 `/cmd_vel_stamped` 并调用 `molmo_spaces` 控制接口。
5. 仿真步进后再次发布观测，形成闭环。

## 5. 最小可运行版本（MVP）实施步骤

## 阶段 A：只替换执行器（1-2 天）

- 新建 `molmo_ros_bridge.py`，对齐 `ai2thor_ros.py` 的发布/订阅 topic 名
- 先发布：
  - `/odometry`
  - `/registered_scan`（可由 depth 回投）
  - `/ai2thor/rgb_image`
- 消费 `/cmd_vel_stamped` 并映射到 base 控制（线速度 + 角速度）
- 保证 `nav.launch` + `explore_manager.launch` 可直接跑通

## 阶段 B：补齐语义输入（1-3 天）

- 暂用 mock 检测器发布 `/explore_agent/result_info`（JSON schema 与现有一致）
- 再接 Grounded-SAM 或 `molmo_spaces` 内置视觉管线
- 验证 `semantic_mapping_pkg` 与 `explore_pkg` 的目标接近流程

## 阶段 C：统一配置与启动（1 天）

- 新增统一 launch：
  - 启 `roscore`
  - 启 `molmo_spaces_ros_bridge`
  - 启 `struct_mapping_pkg`
  - 启 `semantic_mapping_pkg`
  - 启 `explore_pkg`
  - 启 `nav_pkg`
- 将硬编码路径改成 ROS 参数（场景、模型、分辨率、FOV、频率）

## 6. 关键技术细节

## 6.1 坐标与单位

- `molmo_spaces` 位姿通常是世界坐标（米），直接映射到 `odom`。
- 统一右手系与 yaw 定义，避免 `+90/-90` 补偿散落在多个节点。
- `depth` 必须是米单位，`PointCloud2` 与 `camera_info` 内参一致。

## 6.2 控制映射

- 将 `/cmd_vel_stamped.twist.linear.x` 与 `angular.z` 映射为 `molmo_spaces` 基座速度命令。
- 推荐在桥接层加限幅和低通滤波，避免 planner 抖动放大。
- 桥接层应维护固定控制周期（例如 20-30Hz）。

## 6.3 频率预算

- 传感器发布：10Hz 左右可满足当前导航链路。
- 控制处理：20-30Hz。
- 若渲染开销大，优先保证 odom + cmd 回路实时性，图像可降频。

## 6.4 故障与回退

- 当语义不可用时，`explore_pkg` 仍可通用探索（无目标描述）。
- 当点云不可用时，可先验证纯里程计 + move_base 基础可达性（降级模式）。

## 7. 建议的代码改造清单

`Interactive-Nav-SG-nav` 侧（少改）：

- `nav_pkg`、`explore_pkg` 基本不动，仅通过参数调 topic/frame。
- `semantic_mapping_pkg/config/default.yaml` 中输入 topic 按桥接输出对齐。

`molmo_spaces` 侧（新增）：

- 新建 `molmo_spaces_ros_bridge` 包（Python 节点优先，便于快速迭代）。
- 增加“仿真驱动器”类：负责 reset/step/render/control。
- 增加“ROS 编码器”类：`numpy -> Image/PointCloud2/Odometry/String(JSON)`。

## 8. 联调检查表

- `rostopic list` 中关键 topic 均存在。
- `/odometry` 与 TF 链连续更新，无跳变。
- `/cmd_vel` 发出后，`molmo_spaces` 机器人可稳定响应。
- `slam_gmapping` 能生成 `/struct_mapping/occ_map`。
- `explore_manager` 能持续发布有效 goal。
- 语义输入开启时，能从 `EXPLORING` 转 `APPROACHING`。

## 9. 风险与规避

- 依赖冲突风险：建议将桥接层与训练/模型环境分离（独立 conda env）。
- 坐标系不一致风险：先写单元测试验证 4 个标准姿态（前后左右）映射。
- 频率不足风险：先跑 headless + 降分辨率，再逐步提画质。

---

结论：最稳妥路径是“保留现有 ROS 导航栈，替换仿真桥接层”。你可以把 `ai2thor_ros.py` 视作接口参考模板，在 `molmo_spaces_ros_bridge` 中复刻其 topic 契约，再逐步将感知与语义能力迁移到 `molmo_spaces` 原生管线。
