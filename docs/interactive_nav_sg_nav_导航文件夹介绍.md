# Interactive-Nav-SG-nav 导航文件夹介绍

本文面向将 `Interactive-Nav-SG-nav` 接入 `molmo_spaces` 的开发者，聚焦该工程中与导航链路直接相关的目录、依赖和接口。

## 1. 导航相关目录总览

`Interactive-Nav-SG-nav` 是一个 ROS Catkin 工作空间（根目录有 `src/`、`build/`、`devel/`），核心导航链路主要由以下包组成：

- `src/ai2thor_pkg`：AI2-THOR 与 ROS 的桥接层（仿真驱动与传感器发布）
- `src/struct_mapping_pkg`：SLAM/GMapping 结构地图构建
- `src/semantic_mapping_pkg`：语义建图（物体/场景语义）
- `src/explore_pkg`：探索管理器（目标调度、状态机、评分地图、探索目标输出）
- `src/nav_pkg`：move_base 导航栈启动与速度话题中继
- `src/SG_Nav_pkg`：场景图与 SG-Nav 相关逻辑（可选上层能力）

其中，最小导航闭环是：
`ai2thor_pkg -> struct_mapping_pkg/semantic_mapping_pkg -> explore_pkg -> nav_pkg -> ai2thor_pkg`。

## 2. 依赖说明（按层分组）

## 2.1 系统与中间件层

- ROS Noetic 生态：`rospy`、`roscpp`、`tf/tf2`、`nav_msgs`、`sensor_msgs`、`geometry_msgs`、`actionlib`、`move_base_msgs` 等
- Catkin 构建链：`catkin`、`catkin-pkg`
- 典型 ROS 导航组件：`move_base`、`global_planner`、`teb_local_planner`

## 2.2 仿真与数据集层

- `ai2thor==5.0.0`
- `prior` / ProcTHOR 数据加载（支持本地 `train.jsonl.gz/val.jsonl.gz/test.jsonl.gz`）

## 2.3 感知与语义层

- OpenCV、PCL、cv_bridge（图像/点云）
- Grounded-SAM 相关链路（GroundingDINO、SAM、GLIP）
- BLIP 场景属性识别（`semantic_mapping_pkg`）

## 2.4 算法/学习层（SG-Nav 侧）

- PyTorch、torchvision、faiss、pytorch3d
- transformers、timm 等模型依赖

## 2.5 环境隔离建议

仓库 README 采用双环境思路：

- `smartllm`：偏 ROS 桥接/仿真执行
- `SG_Nav`：偏视觉模型/场景图构建

如果要和 `molmo_spaces` 联调，建议再补一个“桥接运行环境”（仅保留桥接必需依赖）以降低冲突面。

## 3. 核心接口（ROS Topic/TF/参数）

## 3.1 `ai2thor_pkg`（仿真桥接）

入口脚本：`src/ai2thor_pkg/script/ai2thor_ros.py`

主要发布：

- `/ai2thor/rgb_image` (`sensor_msgs/Image`)
- `/ai2thor/depth_image` (`sensor_msgs/Image`, 32FC1)
- `/ai2thor/semantic_image` (`sensor_msgs/Image`)
- `/registered_scan` (`sensor_msgs/PointCloud2`)
- `/odometry` (`nav_msgs/Odometry`)
- `/explore_agent/result_info` (`std_msgs/String`, JSON 检测结果)
- `/ai2thor/habitat_obs` (`std_msgs/String`, Habitat 风格观测序列化)
- `/ai2thor/top_view`（可选）

主要订阅：

- `/cmd_vel_stamped` (`geometry_msgs/TwistStamped`)
- `/explore_agent/explore_target` (`geometry_msgs/PointStamped`)

TF 链（代码内帧名）：

- `tf_frame_odom -> tf_frame_base_link -> tf_frame_camera`
- `tf_frame_map` 在下游包中作为地图帧使用

## 3.2 `nav_pkg`（导航执行）

启动文件：`src/nav_pkg/launch/nav.launch`

- 启动 `relay_node`：`/cmd_vel` -> `/cmd_vel_stamped`
- 启动 `move_base`，加载全局/局部代价地图与 TEB 参数

代码：`src/nav_pkg/src/relay_node.cpp`

- 订阅 `/cmd_vel` (`geometry_msgs/Twist`)
- 发布 `/cmd_vel_stamped` (`geometry_msgs/TwistStamped`)

## 3.3 `explore_pkg`（探索调度）

入口：`src/explore_pkg/src/explore_manager.cpp`

订阅（默认参数）：

- 语义图：`/sem_mapping/obj_map`
- 里程计：`/odometry`
- 结构图：`/struct_mapping/occ_map`
- 场景 ID 栅格：`/semantic_mapping/scene_id_grid`
- move_base 状态：`/move_base/status`

发布：

- 导航目标：`/move_base_simple/goal` (`geometry_msgs/PoseStamped`)
- 目标点可视化：`/explore_manager/goal_point`
- 速度控制：`/cmd_vel`
- 状态：`/explore_manager/state`、`/explore_manager/status`
- 探索路径：`/explore_manager/exploration_path`
- 评分地图：`/explore_manager/score_map`

## 3.4 `semantic_mapping_pkg`（语义建图）

默认配置文件：`src/semantic_mapping_pkg/config/default.yaml`

关键输入：

- `detection_topic: /explore_agent/result_info`
- `pointcloud_topic: /registered_scan`
- `occupancy_grid_topic: /struct_mapping/occ_map`
- `blip_image_topic: /ai2thor/rgb_image`

关键输出：

- `/semantic_mapping/scene_attribute`
- `/semantic_mapping/scene_id_grid`
- `/semantic_mapping/scene_confidence_grid`

## 4. 导航数据流（简化）

1. `ai2thor_pkg` 生成视觉/深度/点云/里程计。
2. `struct_mapping_pkg` 形成占据地图 `/struct_mapping/occ_map`。
3. `semantic_mapping_pkg` 融合检测和场景语义，输出语义网格。
4. `explore_pkg` 根据占据图 + 语义图选点并发布 `/move_base_simple/goal`。
5. `move_base` 产出 `/cmd_vel`，`nav_pkg/relay_node` 转成 `/cmd_vel_stamped`。
6. `ai2thor_pkg` 消费 `/cmd_vel_stamped`，驱动 AI2-THOR agent 动作，进入下一循环。

## 5. 对接 molmo_spaces 时最重要的接口面

- 输入面（给导航）：`/odometry`、`/registered_scan`、`/explore_agent/result_info`
- 控制面（接收导航）：`/cmd_vel_stamped`
- 目标面（可选）：`/explore_agent/explore_target`
- 坐标一致性：`tf_frame_*` 与 map/odom/base/camera 的统一定义

若这些接口保持稳定，底层仿真器（AI2-THOR 或 `molmo_spaces`）可以替换为“同一 ROS 契约”下的不同实现。
