# Molmo Spaces 导航开发阶段手册

本手册用于维护 `nav_to_obj` 与 ROS 适配相关的开发流程。内容基于 `test.md` 和近期联调结论整理，按阶段执行可减少环境与资源问题。

## 阶段 0：环境与路径配置

目标：确保资源目录和运行环境正确，避免路径冲突与缺资源。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlspaces

export MLSPACES_CACHE_DIR=/home/user/.cache/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/home/user/ldl/molmospaces/assets

python -c "from molmo_spaces.molmo_spaces_constants import DATA_CACHE_DIR, ASSETS_DIR; print(DATA_CACHE_DIR); print(ASSETS_DIR)"
```

说明：
- `MLSPACES_CACHE_DIR` 和 `MLSPACES_ASSETS_DIR` 不能是同一路径。
- `export` 只对当前终端生效，长期生效需写入 `~/.zshrc`。

## 阶段 1：快速单次管线验证（run_pipeline）

目标：先验证 `nav_to_obj` 主流程可跑通。

```bash
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --scene_dataset ithor --house_inds 1 --samples_per_house 1
```

可视化版本：

```bash
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --scene_dataset ithor --house_inds 1 --samples_per_house 1 --viewer
```

默认环境版本（不显式传 `scene_dataset`）：

```bash
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1
```

## 阶段 2：使用 ProcTHOR 场景进行导航生成

目标：切换到 `procthor-10k` 做真实批量场景测试。

```bash
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --scene_dataset procthor-10k --house_inds 1 --samples_per_house 1
```

批量（每场景 1 次，前 100 个场景）：

```bash
for i in $(seq 1 100); do
  python scripts/datagen/run_pipeline.py \
    --task_type nav_to_obj \
    --policy planner \
    --robot rby1 \
    --scene_dataset procthor-10k \
    --house_inds $i \
    --samples_per_house 1 \
    --seed 2 \
    --run_name_prefix nav100 || true
done
```

## 阶段 3：按需抓取场景资源（避免缺文件）

目标：按测试区间预热资源，减少 `missing scene file`。

```bash
for i in $(seq 0 100); do
  python scripts/datagen/fetch_assets.py scene procthor-10k $i --split train --variant ceiling || true
done
```

## 阶段 4：场景筛选与“大场景”挑选

目标：选择更大、更适合传统导航评测的固定场景集合。

生成面积排序 CSV：

```bash
python scripts/datagen/rank_nav_scenes.py --scene_dataset procthor-10k --data_split train
```

同时导出 top-down 预览图：

```bash
python scripts/datagen/rank_nav_scenes.py --scene_dataset procthor-10k --data_split train --save_maps
```

输出：
- `assets/scene_rankings/procthor-10k_train_ranking.csv`
- `assets/scene_rankings/house_*_map.png`（开启 `--save_maps` 时）

## 阶段 5：ROS 仿真桥接验证

目标：运行 ROS 收发 policy，发布观测并接收动作。

```bash
python scripts/datagen/run_nav_ros_sim.py \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 0 \
  --target_types Apple \
  --samples_per_house 1 \
  --observation_topic /molmo_spaces/observation \
  --action_topic /molmo_spaces/action
```

查看图像观测 topic（当前 policy 已改为发布 `sensor_msgs/Image`）：

```bash
rostopic hz /molmo_spaces/observation
rostopic echo /molmo_spaces/observation/header
```

## 阶段 6：传统导航框架适配建议

目标：固定测试集，做可复现实验对比，而不是每次随机。

建议策略：
- 固定场景：使用 `rank_nav_scenes.py` 结果选一批 `house_inds`。
- 固定随机性：固定 `--seed`，关闭随机化开关。
- 固定目标类型：使用 `--target_types`。
- 固定每场景采样次数：`--samples_per_house 1`。

推荐执行顺序：
1. 先批量 `fetch_assets`（目标 house 区间）。
2. 再跑 `rank_nav_scenes` 选 Top-N 大场景。
3. 最后在固定场景列表上批量跑 `run_pipeline` 或 `run_nav_ros_sim`。

## 阶段 7：RBY1 控制配置速查（ROS bridge 联调）

目标：明确 `RBY1Config` 的控制接口，避免 action 维度或字段不匹配。

配置位置：
- `molmo_spaces/configs/robot_configs.py` -> `class RBY1Config`

关键控制项（当前默认）：
- `use_holo_base: True`
- `init_qpos["base"] = [0.0, 0.0, 0.0]`，语义是 `(x, y, theta/yaw)`
- `command_mode["base"] = "holo_joint_planar_position"`
- `command_mode["arm"] = "joint_position"`
- `command_mode["gripper"] = "joint_position"`
- `command_mode["head"] = None`（头部当前不启用控制）

在 `run_nav_ros_sim.py` 的 `nav_to_obj` 场景下，常用 action：
- 导航主控通常只发 `base`（3 维）和可选 `done`
- `base` 三维格式：`[x, y, yaw]`（`yaw` 单位为弧度）

ROS action payload 推荐格式（`/molmo_spaces/action`）：

```json
{"step": 12, "action": {"base": [2.1, 3.0, 3.08], "done": false}}
```

也支持将 action 直接放在外层（不包 `"action"` 键）：

```json
{"step": 12, "base": [2.1, 3.0, 3.08], "done": false}
```

注意：
- 若未提供 `done`，ROS bridge policy 会默认补为 `false`。
- 若未提供 `base`，policy 会补一个当前位姿的 `base` noop，保证动作结构可被任务消费。

## 阶段 8：Ubuntu 22 + ROS Noetic 编译排障总结（Interactive-Nav-SG-nav）

目标：在 Ubuntu 22.04（jammy）上跑通 `Interactive-Nav-SG-nav` 的 `catkin_make`。

### 8.1 典型问题与根因

- `move_base_msgs` / `openslam_gmapping` 找不到：
  - jammy 环境下不一定有 `ros-noetic-*` 同名二进制包，且部分包需要源码方式。
- `RAPIDJSON_INCLUDE_DIR-NOTFOUND`：
  - `semantic_mapping_pkg` 里引用了未找到的 include 变量。
- `std::shared_mutex` 编译错误：
  - 相关头文件要求 C++17，原工程部分包仍是 C++14。
- `std::ofstream` incomplete type：
  - 源码缺少 `<fstream>` 头文件。

### 8.2 本次已执行的修复

1) 依赖与源码
- 在工作区 `src/` 下加入 `openslam_gmapping` 源码包（用于满足 `struct_mapping_pkg` 依赖）。
- 安装 `rapidjson-dev` 以提供 `rapidjson/document.h`。

2) CMake 与源码修复
- `Interactive-Nav-SG-nav/src/semantic_mapping_pkg/CMakeLists.txt`
  - RapidJSON 检测改为兼容模式，找不到时给出明确错误提示。
  - C++ 标准升级为 C++17。
- `Interactive-Nav-SG-nav/src/grid_map_pkg/CMakeLists.txt`
  - 编译标准从 C++14 升级到 C++17。
- `Interactive-Nav-SG-nav/src/grid_map_pkg/src/grid_map.cpp`
  - 补充 `#include <fstream>`。
- `Interactive-Nav-SG-nav/src/nav_pkg/CMakeLists.txt` / `package.xml`
  - 移除未实际使用的 `move_base_msgs` 硬依赖。
- `Interactive-Nav-SG-nav/src/explore_pkg/CMakeLists.txt` / `package.xml`
  - 移除未实际使用的 `move_base_msgs` 硬依赖。

3) 结果
- `catkin_make` 已通过（工程成功完成编译）。

### 8.3 推荐的复现命令（新环境）

```bash
# 1) 基础环境
source /opt/ros/noetic/setup.zsh

# 2) 进入工作区并准备依赖
cd /home/user/ldl/molmospaces/Interactive-Nav-SG-nav
sudo apt-get update
sudo apt-get install -y rapidjson-dev

# 3) 如无 openslam_gmapping，拉取源码
cd src
git clone https://github.com/ros-perception/openslam_gmapping.git

# 4) 编译
cd ..
catkin_make
```

### 8.4 zsh 小坑记录

- 若执行 `sudo apt remove ros-noetic-*` 出现
  `zsh: no matches found: ros-noetic-*`，
  原因是 zsh 先展开通配符导致命令未传给 apt。
- 可改为：
  - `sudo apt remove 'ros-noetic-*'`
  - 或 `noglob sudo apt remove ros-noetic-*`

## 附：`test.md` 里的常用命令归档

基础生成：

```bash
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1 --scene_dataset procthor-10k
```

抓取场景：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlspaces && for i in $(seq 0 100); do python scripts/datagen/fetch_assets.py scene procthor-10k $i --split train --variant ceiling || true; done
```

场景排序：

```bash
python scripts/datagen/rank_nav_scenes.py --scene_dataset procthor-10k --data_split train --save_maps
```

ROS 导航仿真：

```bash
python scripts/datagen/run_nav_ros_sim.py \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 0 \
  --target_types Apple
```

## 阶段 9：Head Camera 深度点云坐标与 CameraInfo 联调记录

目标：排查并修复 `head_camera` 的 `depth -> depthcloud -> /registered_scan` 显示畸变与坐标轴不一致问题。

### 9.1 现象

- `/registered_scan` 在 RViz 中显示正常（`x` 前、`y` 左、`z` 上）。
- `/molmo_spaces/head_camera/depth` 在 RViz `DepthCloud` 中出现比例异常/方向异常。
- 去掉 `/depth` 对应 `camera_info` 后，DepthCloud 显示恢复，说明问题集中在 `CameraInfo`。

### 9.2 根因

1) 内参来源链路不稳定（曾出现 `fx/fy/cx/cy` 与深度分辨率不匹配）  
- 典型异常日志：`fx` 与 `fy` 差异过大，`cy` 偏离图像中心。  
- 这会导致 RViz 的 `DepthCloud` 回投比例错误（常见为“高度被拉长”）。

2) 光学坐标系与机器人坐标系混用  
- RViz `DepthCloud` 默认按光学系解释：`+Z` 前、`+X` 右、`+Y` 下。  
- `/registered_scan` 使用的是机器人系：`+X` 前、`+Y` 左、`+Z` 上。  
- 两者若无清晰 TF 关系，会出现“朝向看起来不一致”。

### 9.3 本次修复

- 统一 head 相机相关 topic 命名空间：
  - `/molmo_spaces/head_camera/image`
  - `/molmo_spaces/head_camera/depth`
  - `/molmo_spaces/head_camera/camera_info`
  - `/molmo_spaces/head_camera/image/camera_info`
  - `/molmo_spaces/head_camera/depth/camera_info`
- `run_nav_ros_sim.py` 中，`rby1` 运行时动态启用 `head_camera.record_depth=True`（仅本次实验生效）。
- `ros_bridge_policy.py` 中：
  - `depth` 与 `image` 使用一致的 `frame_id`（当前为 `optical_frame_id`）。
  - 为 head 相机强制使用 FOV+分辨率重建内参作为最终回投内参（降低观测内参异常影响）。
  - 增加节流 debug 日志，打印 RAW 与最终使用的 `fx/fy/cx/cy`。
  - 发布 `tf_frame_lidar -> head_camera_optical_frame` 静态 TF，并修正旋转矩阵使光学系/机器人系关系符合预期。

### 9.4 坐标约定（当前）

- `DepthCloud`（光学系）：
  - 前：`+Z`
  - 右：`+X`
  - 下：`+Y`
- `/registered_scan`（机器人系）：
  - 前：`+X`
  - 左：`+Y`
  - 上：`+Z`

### 9.5 快速核查命令

```bash
# 检查关键 topic
rostopic info /molmo_spaces/head_camera/depth
rostopic info /molmo_spaces/head_camera/depth/camera_info
rostopic info /registered_scan

# 检查 TF
rosrun tf tf_echo tf_frame_lidar head_camera_optical_frame
rosrun tf view_frames
```

### 9.6 判定标准

- 若 `depth_shape=1024x576`，则内参中心应接近：`cx=512`、`cy=288`。
- 最终用于回投的 `fx/fy` 应同量级，不应出现 2 倍以上差异。
- RViz 中 `DepthCloud` 与 `/registered_scan` 的方向差异应仅来自坐标系定义，不应出现额外扭曲/拉伸。

## 阶段 10：Bug 记录 - RBY1 head 点云 roll 偏差导致地面倾斜

### 10.1 现象

- 在 `python scripts/datagen/run_nav_ros_sim.py` 使用 `rby1` 时，`/registered_scan` 在 RViz 中出现地面左右高差（例如“左高右低”）。
- 表现为点云整体绕前向轴（`+X`）存在一个小的顺时针旋转偏差。

### 10.2 影响

- 占据地图中地面/障碍边界会出现轻微倾斜，影响 `struct_mapping_pkg` 的高度滤波与局部障碍判断稳定性。
- 若高度滤波阈值较窄，可能放大滤波误杀（部分有效地面点被过滤）。

### 10.3 临时规避方案（已实现）

- 在 `ros_bridge_policy.py` 中增加点云 roll 补偿参数：`pointcloud_roll_correction_deg`。
- 在 `run_nav_ros_sim.py` 中暴露 CLI 参数：

```bash
--pointcloud_roll_correction_deg <deg>
```

- 使用建议：
  - 若观察到“左高右低”，优先尝试负值（如 `-2.0` 到 `-4.0`）。
  - 若出现“左低右高”，说明补偿过量，回调到绝对值更小的负值。

### 10.4 示例命令

```bash
python scripts/datagen/run_nav_ros_sim.py \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 0 \
  --target_types Apple \
  --pointcloud_roll_correction_deg -2.5
```

### 10.5 后续计划

- 优先排查 `head_camera` 外参链路中的 roll 源头（相机外参 / 光学坐标变换 / 点云坐标变换）。
- 在确认真实外参后，回收临时补偿参数，避免长期依赖经验调参。
