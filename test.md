# 交互导航开发测试手册

最后更新：2026-07-06

## 1. 文档定位

本文件用于维护交互导航相关的：

1. 开发测试命令
2. 流程测试方式
3. 常用参数配置
4. 调试入口与输出位置

使用原则：

- 面向项目整体立意、研究目标和文档入口，请看 `readme_pi.md`
- 面向子任务拆解和阶段推进，请看 `TODO.md`
- 面向 Agent 协作约定与维护边界，请看 `AGENTS.md`
- 面向具体命令、运行方式、调试方式和测试流程，请看本文件

---

## 2. 使用说明

本文件按“从轻到重”的方式组织测试入口：

1. 环境与路径检查
2. MolmoSpaces 侧最小运行
3. ROS 联调与导航系统启动
4. 语义交互图与检测测试
5. 地图、探索与手动控制辅助调试

如果只是改动文档、配置或高层设计，不需要跑重型测试。  
如果只是改动某个局部模块，优先使用最小相关命令，而不是整套系统全启动。

---

## 3. 环境与路径

## 3.1 常用环境

- `mlspaces`：MolmoSpaces 与主导航流程
- `yolo_world`：本地 YOLOE / 检测测试相关

## 3.2 常用资源路径

- 资产目录：`/home/user/ldl/molmospaces/assets`
- ROS 输出目录：`/home/user/ldl/molmospaces/assets/datagen/nav_to_obj_ros_sim_v1`
- GT 导出目录：`/home/user/ldl/molmospaces/scripts/InteractiveNav/output`
- RViz 配置：`/home/user/ldl/molmospaces/nav_rviz.rviz`

## 3.3 资产环境变量

```bash
export MLSPACES_CACHE_DIR=/home/user/.cache/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/home/user/ldl/molmospaces/assets
```

检查当前解析到的目录：

```bash
python -c "from molmo_spaces.molmo_spaces_constants import DATA_CACHE_DIR, ASSETS_DIR; print(DATA_CACHE_DIR); print(ASSETS_DIR)"
```

---

## 4. MolmoSpaces 侧最小测试

## 4.1 `nav_to_obj` 数据流最小测试

适用场景：

- 检查 MolmoSpaces 环境是否可运行
- 检查 `nav_to_obj` 主流程是否正常
- 作为交互导航主线的最小基础测试

```bash
conda activate mlspaces
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1
```

指定场景数据集：

```bash
conda activate mlspaces
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1 --scene_dataset procthor-10k
```

## 4.2 场景排序测试

适用场景：

- 初步分析导航场景
- 生成可参考的 scene ranking / map 结果

```bash
conda activate mlspaces
python scripts/datagen/rank_nav_scenes.py --scene_dataset procthor-10k --data_split train --save_maps
```

## 4.3 抓取/预取场景资源

适用场景：

- 提前准备一批场景资源，从cache软链接到本地资产目录
- 避免运行时逐个下载

```bash
export MLSPACES_CACHE_DIR=/home/user/.cache/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/home/user/ldl/molmospaces/assets

source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlspaces && for i in $(seq 0 100); do python scripts/datagen/fetch_assets.py scene procthor-10k $i --split train --variant ceiling || true; done

source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlspaces && for i in $(seq 0 1000); do python scripts/datagen/fetch_assets.py scene procthor-10k $i --split val --variant ceiling || true; done

```

注意：该命令较重，不应在普通文档修改后顺手执行。

---

## 5. 导航仿真与 ROS 联调

## 5.1 启动导航仿真

适用场景：

- 运行 `run_nav_ros_sim.py`
- 联通 MolmoSpaces 与 ROS 导航桥接

```bash
conda activate mlspaces
python scripts/InteractiveNav/run_nav_ros_sim.py \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --target_types Apple \
  --task_horizon 3000
```

## 5.2 带 timing 日志的仿真测试

适用场景：

- 观察运行耗时
- 定位性能瓶颈

```bash
conda activate mlspaces
python scripts/datagen/run_nav_ros_sim.py \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --target_types Apple \
  --timing_log_every_n_frames
```

## 5.3 ROS 导航系统一键启动

适用场景：

- 启动 ROS 导航联调整套流程
- 配合 RViz、语义地图、探索包调试

```bash
conda activate mlspaces
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch
```

## 5.4 语义地图调试启动

适用场景：

- 单独调试 `semantic_mapping_py_pkg`
- 检查 unified graph、navigation hints 与相关 topic

```bash
conda activate mlspaces
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch semantic_mapping_py_pkg semantic_mapping_debug.launch
```

## 5.5 RBY1 初始机械臂姿态验证

适用场景：

- 调试探索过程中机器人手臂碰撞墙体、门框或家具的问题
- 固定机器人位置和外部相机，比较不同初始手臂姿态
- 保存正视图和侧视图，作为后续导航探索配置依据

当前推荐初始姿态为 `shoulder_roll_in_045`，该姿态主要做水平方向贴近身体，不做明显前后或上下抬放：

```text
left_arm_qpos  = 0.28,0.0,-0.45,-0.64,0.39,-0.26,-0.04
right_arm_qpos = 0.28,0.0, 0.45,-0.64,0.39,-0.26,-0.04
```

该姿态已经配置为 `Interactive-Nav-SG-nav/src/nav_pkg/launch/molmospaces_nav_system.launch` 中的默认左右臂初始姿态。启动完整导航系统时默认生效：

```bash
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch start_explore_py:=true
```

如需重新生成固定外部相机快照，可关闭 mapping/nav/explore，只启动仿真并保存 reset 后的 debug snapshot。侧视图示例：

```bash
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch \
  start_mapping:=false \
  start_nav:=false \
  start_explore:=false \
  start_explore_py:=false \
  start_semantic_mapping:=false \
  exploration_only:=true \
  publish_debug_front_camera:=true \
  house_ind:=1 \
  target_types:="" \
  task_horizon:=20 \
  sim_extra_args:="--immediate_noop_after_publish --timing_log_every_n_frames 0 --fixed_robot_xyyaw 6.7,7.8,-0.53 --fixed_debug_camera_pos 5.9,6.9,1.25 --fixed_debug_camera_target 6.7,7.8,0.85 --initial_left_arm_qpos 0.28,0.0,-0.45,-0.64,0.39,-0.26,-0.04 --initial_right_arm_qpos 0.28,0.0,0.45,-0.64,0.39,-0.26,-0.04 --debug_snapshot_path /home/user/ldl/molmospaces/outputs/important_results/rby1_initial_arm_pose_shoulder_roll_in_045/side_view.png"
```

正视图只需要替换固定相机位置：

```bash
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch \
  start_mapping:=false \
  start_nav:=false \
  start_explore:=false \
  start_explore_py:=false \
  start_semantic_mapping:=false \
  exploration_only:=true \
  publish_debug_front_camera:=true \
  house_ind:=1 \
  target_types:="" \
  task_horizon:=20 \
  sim_extra_args:="--immediate_noop_after_publish --timing_log_every_n_frames 0 --fixed_robot_xyyaw 6.7,7.8,-0.53 --fixed_debug_camera_pos 8.15,6.95,1.15 --fixed_debug_camera_target 6.7,7.8,0.85 --initial_left_arm_qpos 0.28,0.0,-0.45,-0.64,0.39,-0.26,-0.04 --initial_right_arm_qpos 0.28,0.0,0.45,-0.64,0.39,-0.26,-0.04 --debug_snapshot_path /home/user/ldl/molmospaces/outputs/important_results/rby1_initial_arm_pose_shoulder_roll_in_045/front_view.png"
```

重要测试结果归档目录：

```text
/home/user/ldl/molmospaces/outputs/important_results/rby1_initial_arm_pose_shoulder_roll_in_045
```

---

## 6. 手动控制与辅助调试

## 6.1 手动控制模式

适用场景：

- 不跑自动策略，手动驱动机器人
- 检查底盘响应、topic 连通和 ROS bridge 是否正常

启动手动模式：

```bash
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch manual_control:=true
```

启动手动控制节点：

```bash
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
rosrun nav_pkg manual_cmd_vel.py
```

控制键：

- `w`：前进
- `s`：后退
- `a`：原地左转
- `d`：原地右转
- `space`：停止
- `+/-`：调速度
- `q`：退出

说明：脚本直接发布 `/cmd_vel_stamped`，由 `RosBridgePolicy` 读取并驱动机器人移动。

## 6.2 机械臂位置调试

适用场景：

- 调试左臂位置或 debug policy 模式

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlspaces && \
python scripts/datagen/run_nav_ros_sim.py \
  --viewer \
  --policy_mode left_arm_debug \
  --left_arm_joint_delta 0.05 \
  --debug_loop_episodes 50 \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 0 \
  --target_types Apple \
  --task_horizon 2000
```

## 6.3 RViz

```bash
rviz -d /home/user/ldl/molmospaces/nav_rviz.rviz
```

---

## 7. 地图、探索与导航子系统

## 7.1 占据地图

```bash
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch struct_mapping_pkg slam_gmapping.launch
```

配置文件：

- `Interactive-Nav-SG-nav/src/struct_mapping_pkg/config/slam_gmapping_params.yaml`

## 7.2 探索策略包

```bash
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch explore_pkg explore_manager.launch
```

配置文件：

- `Interactive-Nav-SG-nav/src/explore_pkg/config/exploration_planner_params.yaml`

## 7.3 导航控制

```bash
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg nav.launch
```

## 7.4 地图/策略统一重置

```bash
source ./Interactive-Nav-SG-nav/devel/setup.zsh
rostopic pub -1 /nav_system/reset std_msgs/Empty "{}"
```

---

## 8. 检测与语义交互图测试

## 8.1 本地检测可视化测试

适用场景：

- 测试 YOLOE prompt-free 检测
- 检查 2D 检测与 3D box 投影结果
- 快速验证 `semantic_mapping_py_pkg` 检测侧是否正常

```bash
conda activate yolo_world
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch semantic_mapping_py_pkg object_detection_visual_test.launch \
  rgb_path:=/home/user/ldl/molmospaces/detection_models/tum_rgbd_scribble_samples/kitchen_22_image.png \
  depth_path:=/home/user/ldl/molmospaces/detection_models/tum_rgbd_scribble_samples/kitchen_22_depth.png \
  backend:=yoloe_pf_box3d \
  provider:=yoloe_local \
  model_path:=/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26x-seg-pf.pt
```

## 8.2 GT 导出测试

适用场景：

- 生成场景 GT 导出文件
- 为后续 `semantic_mapping_gt_replay.py` 提供输入

输出目录：

- `/home/user/ldl/molmospaces/scripts/InteractiveNav/output`

```bash
conda activate mlspaces
python scripts/InteractiveNav/read_scene_room_properties.py \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --variant base
```

## 8.3 GT replay 测试

适用场景：

- 用 GT 结果模拟 detector-like 输入
- 调试 unified graph、room context、navigation hints

```bash
conda activate mlspaces
source ./Interactive-Nav-SG-nav/devel/setup.zsh
python Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/semantic_mapping_gt_replay.py \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/output/procthor-10k_train_1_base_scene_full.json \
  --batch-size 4 \
  --publish-rate 1.0
```

## 8.4 实时 GT 语义交互图

只使用当前 `head_camera` segmentation 中可见对象作为 GT observation：

```bash
conda activate mlspaces
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch \
  start_semantic_mapping:=true \
  semantic_source:=realtime_gt \
  publish_realtime_gt:=true \
  start_explore_py:=true
```

检查实时输入与动态图：

```bash
rostopic echo -n 1 /semantic_mapping/gt_observations
rostopic echo -n 1 /semantic_mapping/unified_graph
rosservice call /semantic_mapping/save_graph
```

实时 GT 默认每 `3` 个仿真 step 采样一次，单次扫描 segmentation 中的 geom ID，最大观测距离默认 `6m`。
三场景完整验收默认每场景录制 `600s`：

```bash
scripts/InteractiveNav/run_semantic_gt_three_scene_test.zsh outputs/semantic_gt_three_scene_10min
```

六联图在线以 `1 FPS` 合成关键帧，最终每张关键帧只写入一次并编码为 `15 FPS`。因此 `600s` 测试约产生 `600` 个关键帧，合成视频约 `40s`，即约 `15` 倍加速播放，同时避免 15 Hz 在线绘图阻塞 ROS 与仿真。

前期短测可覆盖运行时间和场景：

```bash
HOUSE_INDS="4" RECORD_SEC=60 TASK_HORIZON=800 \
  scripts/InteractiveNav/run_semantic_gt_three_scene_test.zsh outputs/semantic_gt_debug_short
```


## 8.5 Room分割测试

python /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/room_segmentation_debug_tool.py live \
  --output-dir /home/user/ldl/molmospaces/scripts/InteractiveNav/output/room_occ_snaps
这个 live 模式会订阅 /struct_mapping/occ_map，默认只有当和上一次保存相比，变化栅格数超过 500 或变化比例超过 0.02 才保存一次。保存格式是 .npz 快照。

离线回放模式：
python /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/room_segmentation_debug_tool.py replay \
  --input /home/user/ldl/molmospaces/scripts/InteractiveNav/output/room_occ_snaps \
  --output-dir /home/user/ldl/molmospaces/scripts/InteractiveNav/output/room_occ_overlays
这个 replay 模式会读取单个 .npz 或整个目录，调用 import 进来的 RoomSegmenter 做分割，然后输出：
每张占据图的 room 叠加图 *_room_overlay.png
一个汇总 summary.json


---

## 9. 常见测试组合

## 9.1 最小基础检查

适用场景：

- 检查环境是否能跑
- 不进入 ROS 联调

建议顺序：

1. `nav_to_obj` 最小测试
2. 场景路径与资产目录检查

## 9.2 ROS 联调检查

适用场景：

- 检查导航系统能否正常启动
- 检查 RViz、map、explore、nav 侧链路

建议顺序：

1. 启动 `molmospaces_nav_system.launch`
2. 打开 RViz
3. 必要时手动控制
4. 必要时统一 reset

## 9.3 模块化语义图检查

适用场景：

- 检查 detector-only / GT replay 到 unified graph 的路径

建议顺序：

1. 启动 `semantic_mapping_debug.launch`
2. 运行 `object_detection_visual_test.launch` 或 GT replay
3. 在 RViz / rostopic 中检查输出

## 9.4 小规模验证集检查

适用场景：

- 检查第一阶段闭环是否真实成立
- 在进入大规模 benchmark 之前做稳定 sanity check

建议验证内容：

1. 选择少量场景，确保存在明确 `door-blocked path`
2. 分别运行：
   - `pure nav`
   - `nav + oracle open`
   - ROS 模块化链路或 interaction-aware 规划链路
3. 记录：
   - 是否需要交互
   - 交互前后连通性是否变化
   - 是否成功到达目标
   - 路径是否明显缩短或从不可达到可达

建议优先观察：

- 门关闭时纯导航是否失败或绕行
- oracle 开门后是否立即恢复可达性
- 图/提示是否给出正确的交互信号
- 交互后是否触发继续导航而不是停在中间状态

---

## 10. 参数说明与备注

## 10.1 常用参数

- `--scene_dataset procthor-10k`：指定场景数据集
- `--data_split train`：指定 split
- `--house_ind 1`：指定场景编号
- `--target_types Apple`：指定导航目标类别
- `--task_horizon 3000`：指定任务最长步数
- `manual_control:=true`：ROS 导航系统进入手动控制模式
- `backend:=yoloe_pf_box3d`：检测后端使用 YOLOE prompt-free + box3d
- `provider:=yoloe_local`：本地 YOLOE provider

## 10.2 当前文档边界

本文件当前主要维护：

- 已确认可用的命令
- 常见调试流程
- 常用路径与参数

本文件当前不负责：

- 项目立意与研究主张
- 子任务拆解与阶段判断
- 历史讨论内容归档

这些内容分别由 `readme_pi.md`、`TODO.md` 与 `AGENTS.md` 维护。
