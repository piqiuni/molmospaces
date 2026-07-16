# 交互导航开发测试手册

最后更新：2026-07-16

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

## 4.4 开关门 GT path获取


测试
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MLSPACES_CACHE_DIR=/tmp/molmo-spaces-cache-proxy MLSPACES_ASSETS_DIR=/tmp/molmo-spaces-assets-proxy /home/user/miniconda3/envs/mlspaces/bin/python scripts/InteractiveNav/explore_molmo_interactions.py door-path-study --house_ind 10 --variant ceiling --target_types fridge --close_doors_on_path 0 --output_json scripts/InteractiveNav/output/door-path-study_procthor-10k_train_10_fridge
```

## 4.5 mixed 数据粗筛与 V3 精采集

适用场景：

- 从现有 `container_rough_catalog_v1` 二次生产 GT path 穿门的 crossing rough 候选
- 将单门关闭阻断容器交互目标、门前几何 path-backoff 提示仍可达记录为 `mixed_required_verified` 子集标签，而不是 rough 输入门槛
- 直接生成带真实 joint/readback、路径、可见性、成功证据和最小性证据的 `interactive_nav_v3`

正式 mixed rough 输入范围固定为 container rough 中全部有 strict pair 的 house 和全部 strict pair；禁止使用 door-only `door-required house` 结果预筛。当前全集为 712 house、2603 pair。rough 的门前点只是导航级几何提示，不是 manipulation-validated 开门位姿。

三类 rough smoke（crossing-only、required、无穿门路径）：

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-mixed-rough
python scripts/InteractiveNav/collect_mixed_rough_catalog.py \
  --house_indices 0,101,103 \
  --workers 3 \
  --output_dir scripts/InteractiveNav/output/mixed_rough_catalog_crossing_smoke_v1
```

全量 all-strict rough 扫描：

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-mixed-rough-all-crossing
python scripts/InteractiveNav/collect_mixed_rough_catalog.py \
  --workers 8 \
  --resume \
  --output_dir scripts/InteractiveNav/output/mixed_rough_catalog_all_crossing_v1
```

全量结果必须满足 `summary.json` 中：

- `selection_scope=all_container_rough_houses_with_strict_pairs`
- `candidate_selection=all_open_gt_path_crosses_interactive_door`
- `mixed_required_role=verified_subset_annotation_not_rough_input_gate`
- `door_required_house_prefilter_used=false`
- `expected_strict_house_count=712`
- `expected_strict_pair_count=2603`
- `pair_coverage_complete=true`

2026-07-15 crossing/required 拆分后的全量实测结果：

- 完成 712/712 个 strict-pair house、解析 2603/2603 个 strict pair，失败 0，`pair_coverage_complete=true`
- crossing rough：1609 pair、覆盖 456 house，pair 命中率 61.81%；真正没有任何穿门 pair 的 house 为 256 个
- required 子集：917 pair、覆盖 262 house；与旧 required-only catalog 的 917 个 `case_id` 完全一致
- crossing-only：692 pair；另有 194 个 house 有 crossing pair、但没有任何 rough required pair
- 非 crossing pair：762 对存在开放路径但不穿 interactive door，232 对所有 source start 均无开放路径
- 共检查 6087 个 source start：5513 条开放路径，其中 2895 条穿门；穿门路径中 1511 条通过 required 验证、1384 条为 crossing-only
- crossing 容器类别：Fridge 821，Dresser 788；最短 crossing 路径中位数 9.61 m、均值 10.40 m、范围 0.48--30.05 m
- required 子集路径统计保持原结果：中位数 11.19 m、均值 12.03 m
- 旧 443-house door-required 集合中只有 369 个 house 含 container crossing，另有 87 个 crossing house 位于旧集合外；旧集合只能作为历史对照

### 4.5.1 crossing-only rough 标注俯视图

用于直接检查 `1609 - 917 = 692` 个 crossing-only pair：左图为全开状态的 rough GT planner path，右图为关闭所穿门后的重新规划路径；同时标出起点、门前几何提示、粗容器目标、容器/目标/所穿门以及全部 scene object AABB。

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-mixed-rough-viz
python scripts/InteractiveNav/visualize_mixed_rough_catalog.py \
  scripts/InteractiveNav/output/mixed_rough_catalog_all_crossing_v1/mixed_rough_catalog.json \
  --candidate_type door_crossing_only \
  --max_samples 24 \
  --workers 4 \
  --output_dir scripts/InteractiveNav/output/mixed_rough_catalog_all_crossing_v1/crossing_only_visualizations
```

默认按 Fridge/Dresser、路径长度和穿门数量分层抽样，并优先使用不同 house。输出包括逐样本双面板 PNG、含完整路径点和 Oxxx 对照的 JSON、`contact_sheet_*.png`、`index.html` 与 `manifest.json`。

需要检查某个粗门前几何提示的机器人第一视角时，可指定单个 `case_id` 并增加 `--render_first_person`；输出会同时给出该门全开/关闭的 `head_camera` 对比图：

```bash
python scripts/InteractiveNav/visualize_mixed_rough_catalog.py \
  --case_id <MIXED_ROUGH_CASE_ID> \
  --max_samples 1 \
  --workers 1 \
  --render_first_person \
  --output_dir scripts/InteractiveNav/output/mixed_rough_first_person
```

2026-07-16 的 24-house 图册重放结果：24/24 成功出图、全开路径长度均与 catalog 一致；其中 house 351/candidate 427 的闭门可达性与 catalog 不一致，已在图和 sidecar 中标记 `CATALOG/REPLAY MISMATCH`，应作为粗地图边界不稳定样本复扫，而不是直接进入 fine 生产。

10 条 V3 fine smoke：

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-mixed-fine
python scripts/InteractiveNav/build_mixed_interaction_benchmark.py \
  --mixed_rough_catalog scripts/InteractiveNav/output/mixed_rough_catalog_crossing_smoke_v1/mixed_rough_catalog.json \
  --source_variants_per_pair 2 \
  --max_samples 10 \
  --max_samples_per_house 8 \
  --px_per_m 50 \
  --max_poses_per_joint 2 \
  --output_dir scripts/InteractiveNav/output/mixed_interaction_v3_smoke10
```

相关单元测试：

```bash
conda activate mlspaces
python -m pytest \
  mlspaces_tests/data_generation/test_mixed_interaction_benchmark.py \
  mlspaces_tests/data_generation/test_container_interaction_benchmark.py \
  mlspaces_tests/data_generation/test_visualize_mixed_interaction_benchmark.py \
  mlspaces_tests/data_generation/test_capture_mixed_gt_storyboard.py \
  -q
```

主要输出：

- rough：`mixed_rough_catalog.json`、`summary.json`、`shards/*/run.log`；支持按成功 house 断点续扫
- fine：`benchmark.json`、`valid.json`、`rejected.json`、`summary.json`
- fine builder 接收 crossing rough 后，会在真实容器交互位姿重新验证 required；crossing-only rough 不会被直接视为正式 mixed episode
- fine episode 必须通过 `validate_mixed_v3_episode`，且 `minimal_plan_verified=true`

## 4.6 mixed V3 标注俯视图

适用场景：

- 重放 mixed V3 中冻结的物体位姿、门/容器关节初态
- 对比 required door 关闭的初态与开门后的 GT 导航状态
- 标出起点、朝向、门前位姿、容器交互位姿、目标、交互对象及全部 scene object AABB
- 输出逐 episode PNG、Oxxx 物体标注 JSON、联系图和 HTML 图册

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-mixed-viz
python scripts/InteractiveNav/visualize_mixed_interaction_benchmark.py \
  scripts/InteractiveNav/output/mixed_interaction_v3_smoke10/benchmark.json \
  --output_dir scripts/InteractiveNav/output/mixed_interaction_v3_smoke10/visualizations \
  --px_per_m 100
```

仅展示部分 episode：

```bash
python scripts/InteractiveNav/visualize_mixed_interaction_benchmark.py \
  scripts/InteractiveNav/output/mixed_interaction_v3_smoke10/benchmark.json \
  --episode_indices 0,8 \
  --max_episodes 2
```

默认会校验：初态到容器交互位姿不可达、门前位姿可达、开门后 GT path 可达，以及重算长度满足 `0.35 m` 绝对误差或 `10%` 相对误差。中断后可显式加 `--reuse_existing` 复用已有 PNG/JSON。

## 4.7 mixed GT 五步 storyboard

用于从 mixed V3 benchmark 自动选择一个“单 required door + 单 Fridge + 单容器交互”的代表场景，并以独立 step 重放以下五个 GT 状态：起点（门关、冰箱关）、门前（门关）、同一门前位姿（门开）、冰箱前（冰箱关）、同一冰箱位姿（冰箱开）。每一步都先重放 episode 冻结初态，再应用目标状态并保存实际关节读回。

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-mixed-story
python scripts/InteractiveNav/capture_mixed_gt_storyboard.py
```

默认相机固定在机器人右肩后上方，并朝机器人前方观察。输出目录包含五张原始 RGB、五份 step sidecar、`story_steps.json`、`storyboard.png`、`pathpoints_topdown.png` 与汇总 `manifest.json`。可固定 episode 或调整镜头：

```bash
python scripts/InteractiveNav/capture_mixed_gt_storyboard.py \
  --episode_index 1 \
  --camera_behind_m 0.72 \
  --camera_right_m 0.34 \
  --camera_height_m 1.72 \
  --camera_lookahead_m 1.75
```

脚本以 `StoryStep` 和逐步 `apply state -> place robot -> place camera -> capture` 循环组织；后续连续视频采集可在相邻 step 之间插值机器人 pathpoint，并将门/冰箱关节状态替换为连续轨迹。



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
  --task_horizon 3000 \
  --policy_dt_ms 100 \
  --ctrl_dt_ms 2 \
  --action_timeout_s 0 \
  --require_fresh_cmd_vel true \
  --require_move_base_active_for_cmd_vel true
```

`action_timeout_s=0` 表示仿真阻塞等待当前观测之后产生的新 ROS 动作；等待期间会周期性重发当前观测，因此 ROS 启动和首个规划阶段不会推进空动作 step。每次 `task.step()` 固定推进 `100ms` 仿真时间，不按墙钟 10Hz sleep。

## 5.2 带 timing 日志的仿真测试

适用场景：

- 观察运行耗时
- 定位性能瓶颈

```bash
conda activate mlspaces
python scripts/InteractiveNav/run_nav_ros_sim.py \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --target_types Apple \
  --timing_log_every_n_frames 20 \
  --sim_timing_log_every_n_steps 20 \
  --step_log_every_n_steps 10
```

`RosBridgePolicy timing` 分解观测发布与动作等待；`SimLoop timing` 分解 policy、MuJoCo physics、sensor polling 和完整循环耗时。ROS 探索默认不渲染未使用的腕部 RGB/depth，相机需求调试时可传 `--include_wrist_cameras true` 恢复。

默认 `ctrl_dt_ms=2` 保持原有低层控制动态。已验证 `--ctrl_dt_ms 10` 可进一步减少 physics 耗时，但会改变控制更新频率，应作为性能实验参数单独验证，不作为探索基线默认值。

## 5.3 ROS 导航系统一键启动

适用场景：

- 启动 ROS 导航联调整套流程
- 配合 RViz、语义地图、探索包调试

```bash
conda activate mlspaces
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch
```

## 5.3.1 多 ROS Master 并行运行

并行调度器为每个 worker 创建独立的 `ROS_MASTER_URI`、`ROS_HOME`、ROS 日志目录、仿真输出目录和进程组。不同 worker 可以继续使用相同的话题名与 TF frame。

先检查分片、端口、GPU 和实际启动命令，不启动 ROS：

```bash
conda activate mlspaces
python scripts/InteractiveNav/run_parallel_ros_episodes.py \
  --house-inds 4 7 10 \
  --num-workers 2 \
  --base-master-port 11411 \
  --output-dir /home/user/ldl/molmospaces/outputs/parallel_ros_dryrun \
  --exploration-only \
  --start-explore-py \
  --task-horizon 50 \
  --dry-run
```

运行两个短场景 worker：

```bash
conda activate mlspaces
python scripts/InteractiveNav/run_parallel_ros_episodes.py \
  --house-inds 4 7 \
  --num-workers 2 \
  --base-master-port 11411 \
  --output-dir /home/user/ldl/molmospaces/outputs/parallel_ros_houses_04_07 \
  --exploration-only \
  --start-explore-py \
  --task-horizon 50 \
  --worker-timeout-s 600 \
  --resource-interval-s 2
```

如果有多个 GPU，可按 worker 指定；GPU 数量少于 worker 时会轮询复用：

```bash
--gpu-ids 0 1
```

主要输出：

- `plan.json`：worker 分片、ROS Master、GPU、命令和 RViz 连接方式
- `worker_NNN/roscore.log`：独立 Master 日志
- `worker_NNN/roslaunch.log`：导航系统和仿真日志
- `worker_NNN/worker.log`：调度器生命周期日志
- `worker_NNN/episodes.jsonl`：house/episode 的 running 与最终状态事件
- `worker_NNN/resources.jsonl`：CPU、RSS、主机内存、swap、GPU 利用率与显存采样
- `worker_NNN/status.json`：worker 退出状态与耗时
- `summary.json`：全部 worker 状态和资源统计

观察指定 worker：

```bash
ROS_MASTER_URI=http://127.0.0.1:11411 \
ROS_HOME=/tmp/molmospaces_ros/worker_000 \
rviz
```

调度器收到 `Ctrl-C` 后会依次向自己创建的 `roslaunch` 和 `roscore` 进程组发送 `SIGINT`、`SIGTERM` 和最终的 `SIGKILL`。不要使用全局 `pkill roscore` 清理，以免影响其他 ROS 任务。

单场景超时与失败重试：

```bash
python scripts/InteractiveNav/run_parallel_ros_episodes.py \
  --house-inds 0 1 2 3 \
  --num-workers 4 \
  --output-dir /home/user/ldl/molmospaces/outputs/parallel_ros_retry_example \
  --data-split val \
  --exploration-only \
  --start-explore-py \
  --task-horizon 850 \
  --scene-timeout-s 900 \
  --max-scene-attempts 2 \
  --max-consecutive-action-timeouts 12
```

- `scene-timeout-s`：单次场景 attempt 的墙钟上限。
- `max-scene-attempts`：场景执行异常或超时后的最大尝试次数。
- `max-consecutive-action-timeouts`：连续没有新 ROS 动作时提前判定本次 attempt 失败。启用场景超时且未显式设置 `action_timeout_s` 时，单次动作等待自动限制为最多5秒。
- 超时会抛出场景执行异常，当前 attempt 不保存为完成轨迹；pipeline 执行清理和 ROS reset 后进入下一次 attempt。正常跑满 horizon 的场景不会额外重试。

并行运行时为每个 worker 记录探索四宫格、第一视角、外部相机、地图轨迹和调试信息：

```bash
python scripts/InteractiveNav/run_parallel_ros_episodes.py \
  --house-inds 0 1 2 3 \
  --num-workers 4 \
  --output-dir /home/user/ldl/molmospaces/outputs/parallel_ros_debug_4w \
  --data-split val \
  --exploration-only \
  --start-explore-py \
  --task-horizon 300 \
  --record-debug
```

每个 worker 的 debug 输出位于：

```text
worker_NNN/debug/
```

主要产物包括：

- `videos/composite_frames/`：第一视角、地图、规划与状态组成的逐帧四宫格PNG
- `videos/first_person.mp4`：包含地图四宫格的H264视频
- `videos/external_camera.mp4`：外部相机视频
- `summary.json`：轨迹、命令、规划、视频和停滞统计
- `final_map_trajectory.png`：最终地图与探索轨迹
- `recorder.log`：独立recorder日志

`--recorder-extra-args` 可继续传递 `record_explore_debug.py` 参数。调度器结束仿真后会先给recorder最多120秒完成异步写盘与视频编码，再关闭ROS Master。

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

开启 semantic mapping 后，global costmap 自动改读 `/semantic_mapping/planning_occ_map`；semantic 节点仍从原始 `/struct_mapping/occ_map` 构建房间与语义信息，避免处理后的地图反馈回自身。门状态达到 `open` 后，semantic 层会在每一帧原始 OCC 上持续清空缓存的闭合门整体 AABB，并同时发布 `/semantic_mapping/door_clear_mask`。为了让 move_base 的 static layer 立即消费门洞变化，还会持续发布小范围 `/semantic_mapping/planning_occ_map_updates`；门关闭后，原门区恢复值会短期重复发布，覆盖 global costmap 的低频更新周期。已确认的交互关节读回状态优先于后续不稳定的视觉 GT 状态，直到下一次交互结果更新。

## 8.5 门状态与语义 OCC 快速测试

定向单测：

```bash
conda activate mlspaces
pytest -q \
  Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/tests/test_portal_state_tracker.py \
  Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/tests/test_semantic_occ_overlay.py \
  Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/tests/test_interaction_graph_store.py
```

7 号场景检查会读取实际 MuJoCo 门关节和 root AABB，分别设置闭合与完全打开状态，并验证单门、双开门的图状态和 OCC 清空结果：

```bash
conda activate mlspaces
python scripts/InteractiveNav/test_semantic_door_occ_house7.py \
  --house-ind 7 \
  --output /tmp/semantic_door_occ_house7.json
```

使用 MuJoCo `xfrc_applied` 力/力矩驱动门关节，而不是直接写关节位置：

```bash
conda activate mlspaces
python scripts/InteractiveNav/test_semantic_door_occ_house7.py \
  --house-ind 7 \
  --interaction-mode force \
  --force-max-physics-substeps 3000 \
  --output /tmp/semantic_door_occ_house7_force.json
```

该检查要求 force backend 将同一 doorway root 下的全部门板打开到语义 open 阈值以上；若任一门板未达到阈值，测试直接失败，不向上层返回可恢复的交互失败。

正式 ROS 联调会把机器人固定在目标门前，初始化时直接关闭全部门，第 `OPEN_STEP` 个仿真 step 直接将目标门铰链设为全开。测试同时保存 raw OCC、semantic planning OCC、door clear mask、move_base global costmap、闭/开门图状态和对比图：

```bash
conda activate mlspaces
OPEN_STEP=100 \
  scripts/InteractiveNav/run_semantic_door_occ_house7_ros_test.zsh \
  outputs/semantic_door_occ_house7_ros
```

测试 House 7 中相邻的双开门（两块门板同时关闭/打开）：

```bash
OPEN_STEP=100 \
TARGET_ROOT=doorway_ada234694d8669f8c477500ae8f01b1a_1_0_4 \
ROBOT_XYYAW=4.20,4.881,0.0 \
  scripts/InteractiveNav/run_semantic_door_occ_house7_ros_test.zsh \
  outputs/semantic_door_occ_house7_double_ros
```

双开门 5 位姿开门、穿门、回头关门测试。这里直接设置机器人 base 位姿，从而同步改变相机、odom 和 TF，而不是只修改渲染相机：

```bash
TARGET_ROOT=doorway_ada234694d8669f8c477500ae8f01b1a_1_0_4 \
ROBOT_XYYAW=4.05,4.881,0.0 \
POSE_SEQUENCE='0:4.05,4.881,0.0,closed,left_far/75:4.35,4.881,0.0,open,left_near/150:4.99,4.881,0.0,open,doorway/225:5.55,4.881,0.0,open,right_forward/300:5.85,4.881,3.1415926536,closed,right_turnback' \
EXPECTED_PHASES=5 \
  scripts/InteractiveNav/run_semantic_door_occ_house7_ros_test.zsh \
  outputs/semantic_door_occ_house7_double_pose5
```

该模式只有在每个位姿的关节状态、语义图状态、机器人位姿、raw/planning OCC 和 global costmap 均满足条件并稳定发布后才截取。输出包括 `occ/phase_*.npz`、每阶段图 JSON、`occ/pose_sequence_summary.json` 和 `occ/pose_sequence_occ_comparison.png`。

验收条件：

- 关门时 raw OCC 与 planning OCC 在门区域一致，clear mask 为空。
- 开门时 clear mask 非空，mask 内 planning OCC 全部为 free。
- raw OCC 仍允许保留静态门板痕迹，证明清空来自 semantic overlay，而不是 GMapping 自行更新。
- move_base global costmap 的门区域收到增量更新，至少产生可通行 free 单元且不再含 lethal 单元。

2026-07-16 的 House 7 正式回归结果：门洞 mask 为 `65` 个栅格；raw OCC 仍有 `16` 个 occupied 和 `6` 个 unknown；planning OCC 的 `65/65` 个栅格均为 free；global costmap 有 `57` 个门洞栅格发生变化，最终为 `53` 个 free、`12` 个 inflation/inscribed、`0` 个 lethal。完整结果见输出目录中的 `occ/summary.json` 与 `occ/door_occ_comparison.png`。

同日双开门 5 位姿严格回归通过：开门后 `115/115` 个 mask 栅格在 planning OCC 中均为 free；左侧近门位姿的 global costmap 为 `107` free、`8` inflation、`0` lethal、`0` unknown；门洞内和右侧位姿均为 `101` free、`14` inflation、`0` lethal、`0` unknown。最终在右侧回头并关门后，clear mask 恢复为 `0`，planning OCC 在参考门区恢复 `31` 个 non-free 栅格，global costmap 恢复 `25` 个 lethal 栅格。

运行中的 ROS 话题检查：

```bash
rostopic echo -n 1 /semantic_mapping/planning_occ_map/info
rostopic echo -n 1 /semantic_mapping/planning_occ_map_updates
rostopic echo -n 1 /semantic_mapping/door_clear_mask/info
rosparam get /move_base/global_costmap/static_layer/map_topic
```

注意：当前 GT 快速版依赖“首次有效门板关节观测发生在交互前、门处于闭合状态”的任务约束；真实场景中的门关节/转轴提取尚未实现。


## 8.6 Room分割测试

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

## 9.5 Python 探索与多场景检查

单场景探索：

```bash
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch \
  start_explore_py:=true \
  exploration_only:=true \
  house_ind:=4
```

同一 simulator 进程顺序加载多个场景：

```bash
roslaunch nav_pkg molmospaces_nav_system.launch \
  start_explore_py:=true \
  exploration_only:=true \
  house_inds:="4,7,10"
```

当前 ROS 探索仿真默认使用 `policy_dt_ms=200`、`ctrl_dt_ms=10`、
`sim_dt_ms=10`，`move_base` controller 为 `5Hz`。场景切换时会取消目标、
清理 costmap，并重置 gmapping 与 `explore_py` 状态。

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
