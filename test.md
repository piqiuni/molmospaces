# 交互导航开发测试手册

最后更新：2026-07-29

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

### 4.5.2 Full 逐步采集与 smoke

`mode: full` 不生成错误动作负轨迹，也不生成 `open_gt_control`。它按指定
executor 执行连续导航和交互，并将每个 step 的第一视角图像、动作、qpos/qvel、
phase、segment、reward、terminal 和 info 写入 `trajectory.h5`；同时为每个相机
写出同一 step 序列的 MP4。force 模式的交互 step 使用
`force effort -> mujoco.mj_step() -> joint readback`，不会直接写 qpos。

统一 full smoke 配置：

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-interactive-full
python scripts/InteractiveNav/collect_interactive_nav.py \
  --config scripts/InteractiveNav/configs/collection/procthor10k_train_full_3domain_smoke.yaml \
  --stage full
```

full 训练资格要求：

- channel：`nav_to_door`、`force_open_door`、`nav_to_target`、`terminal_observation`
- container：`nav_to_container`、`force_open_container`、`terminal_observation`
- mixed：上述两组交互 segment 和 `terminal_observation`
- H5 数组（图像、动作、状态）长度完全对齐，且至少有一个 `force_joint` step 和
  一个 terminal step
- 失败 rollout 仅保留在 `full/runs` 供诊断，不计入训练数据；汇总见
  `full/summary.json`

full 导航使用平滑后的 waypoint 作为 RBY1 holonomic base 的绝对 position target，
由 MuJoCo actuator 逐步跟踪。导航和 force 阶段会保持初始 head、左右臂、夹爪和
torso 的 qpos/qvel/position target；`lock_base_during_force: true` 还会在每个 force
step 前后恢复 base 的 x/y/yaw pose，避免交互把机器人推离操作位或让机械臂下落，
并在 force metadata 中记录锁定组和最大漂移。

full 数据时间基准默认是 `collection_hz=5`，即 `dt=0.2s`；H5 训练 step 与 MP4
视频帧使用同一频率。内部 MuJoCo/controller 可以使用更高频率，但只在每个 0.2s
采样边界保存一次。按人类步行速度 1.4m/s 的 60%（0.84m/s）计算，每个导航
采样间隔对应约 0.168m；当前场景实际导航距离约 4.666m，对应约 28 个导航 step。
冰箱门 2s 连续打开对应 10 个交互 step，目标开度从 0% 递增到100%。任务
`success_threshold`（当前 0.8）只记录成功事件，不提前结束2s交互轨迹。

复用预计算 rough 时可配置：

```yaml
rough:
  container_catalog: /path/to/container/rough_catalog.json
  mixed_catalog: /path/to/mixed/mixed_rough_catalog.json
  generate_if_missing: false
```

统一入口相关测试：

```bash
python -m pytest -q \
  mlspaces_tests/data_generation/test_collect_interactive_nav.py \
  mlspaces_tests/data_generation/test_container_interaction_benchmark.py \
  mlspaces_tests/data_generation/test_mixed_interaction_benchmark.py
```

主要输出位于配置的 `output.root`：

- `scene_manifest.json`：版本化 train house 清单
- `seeds/benchmark.json`：真实 train scene 生成的 NavToObj seed episode
- `raw/{channel,container,mixed}/benchmark.json`：三类 fine 原始 V3 数据
- `balanced/benchmark.json`：最终严格均衡 benchmark
- `balanced/audit.json`、`balanced/structure_report.md`：字段、类别、recipe、house 与占位值审计

full step 级采集使用同一入口；`full.max_episodes` 表示每个
`full.domains` 中各采集的 episode 数。先运行单个 mixed smoke：

```bash
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-interactive-full
python scripts/InteractiveNav/collect_interactive_nav.py \
  --config scripts/InteractiveNav/configs/collection/procthor10k_train_full_smoke.yaml \
  --stage full
```

三类各采一个 smoke 可改用
`scripts/InteractiveNav/configs/collection/procthor10k_train_full_3domain_smoke.yaml`。

full 输出位于 `output.root/full/`：每个 run 包含 `trajectory.h5`、`manifest.json`、
交互结果和日志。只有 `returncode=0`、H5 对齐校验通过且 `success=true` 的 run 才会
进入 `valid_trajectory_count`；导航或操作失败不会被当成 rollout 负轨迹。

### 4.5.3 PointGoal / InstructionGoal 数据生成 demo

从现有 InteractiveNav V3 channel benchmark 采一个 interaction-aware PointGoal：

```bash
conda activate mlspaces
export MUJOCO_GL=egl
python scripts/InteractiveNav/generate_point_goal_v3.py \
  <V3_BENCHMARK_OR_DIR> \
  --episode-index 0 \
  --interaction-aware \
  --px-per-m 50 \
  --output-dir scripts/InteractiveNav/output/point_goal_v3_smoke
```

直接从原始 scene split 采一个普通可达 PointGoal：

```bash
python scripts/InteractiveNav/generate_point_goal_v3.py \
  --source-mode scene_split \
  --house-ind 7 \
  --output-dir scripts/InteractiveNav/output/point_goal_raw_scene_smoke
```

基于 V3 oracle plan 规则生成 hidden/partial/explicit 三条 InstructionGoal：

```bash
python scripts/InteractiveNav/generate_instruction_goal_v3.py \
  <V3_BENCHMARK_OR_DIR> \
  --mode rule \
  --output-dir scripts/InteractiveNav/output/instruction_goal_rule_smoke
```

如果已有 full rollout，可按 segment 抽取首/中/尾关键帧并调用 VLM；`--graph-json`
可选传入 unified graph JSON，生成器只保留 GT path 周围指定半径及必需交互实体：

```bash
python scripts/InteractiveNav/generate_instruction_goal_v3.py \
  <V3_BENCHMARK_OR_DIR> \
  --mode vlm \
  --trajectory-h5 <FULL_RUN>/trajectory.h5 \
  --graph-json <GRAPH_JSON> \
  --graph-radius-m 1.0 \
  --model-mode http \
  --endpoint <OPENAI_COMPATIBLE_ENDPOINT> \
  --model <MODEL_NAME> \
  --output-dir scripts/InteractiveNav/output/instruction_goal_vlm_smoke
```

只验证 full H5 关键帧和 V3 封装、不访问外部 API 时，将上面的
`--model-mode http ...` 替换为 `--model-mode mock`。

不访问外部模型的轻量单元测试与 V3 示例校验：

```bash
conda activate mlspaces
python -m pytest -q mlspaces_tests/data_generation/test_interactive_nav_task_generation.py
python scripts/InteractiveNav/dataset_definition/v3/validate_examples.py
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

## 5.3.2 语义交互导航与并行批测

完整交互导航优先通过统一封装脚本启动，不建议直接调用 `run_nav_ros_sim.py`：

```text
scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh
```

虽然脚本名包含 `house7`，但可通过 `HOUSE_IND` 运行其他场景。脚本会统一启动仿真、实时 GT 感知、动态语义交互图、`explore_py`、语义候选与规则或 MLLM 决策、行为执行器、`move_base`、力交互 backend，以及可选的六联图录制。运行链路为：

```text
run_nav_ros_sim.py
→ realtime GT observation
→ dynamic semantic interaction graph
→ explore_py navigation frontiers
→ semantic navigation/interaction candidates
→ semantic rule / MLLM decision
→ behavior executor
→ move_base navigation / force interaction
→ interaction_result updates graph
```

### 5.3.2.1 METHOD 总览

当前交接基线为分支 `codex/semantic-decision`、提交 `c9fe3c8`。最关键的任务区别是：

- **交互探索没有 object goal**。系统在 `EXPLORE / INTERACT` 候选中决策，完成条件是导航 frontier 与交互 frontier 耗尽，或达到停滞恢复条件。
- **交互目标导航必须额外提供目标来源、目标完成条件和目标优先级**。目标一旦被稳定感知，`NAVIGATE` 目标候选应抢占普通探索；完成条件是到达目标并通过配置的可见性验证。
- 两类任务使用相同 ROS 入口和底层感知、建图、导航、交互执行链路，区别主要来自 `METHOD`、决策配置和运行时目标上下文。

入口脚本当前支持以下全部 METHOD：

| METHOD | 任务与 mission | 决策模块 | 目标来源与完成条件 | 默认配置及用途 |
|---|---|---|---|---|
| `interactive_rule` | 规则交互探索；默认 `semantic_interaction_exploration` | Module 1 `dynamic_rule`；Module 2 `rule_cost`；Module 3 `rule_verified` | 无目标；按交互/导航 frontier 完成 | 不覆盖默认决策配置，适合最小规则链路 smoke test；默认不强制关闭容器 |
| `semantic_interaction_exploration` | 完整规则交互探索 | 三模块均为规则实现 | `target.enabled=false`；导航和交互候选耗尽后完成 | 自动使用 `interactive_exploration.yaml`，并强制关闭容器 |
| `container_exploration` | 与 `semantic_interaction_exploration` 相同的规则交互探索 | 三模块均为规则实现 | 无目标；按 frontier 完成 | 面向批处理脚本保留的别名，自动使用 `interactive_exploration.yaml` |
| `full_mllm_exploration` | 全 MLLM 交互探索 | `dynamic_mllm / mllm_score / mllm_skill_verified` | 无目标；LLM 在具体 `EXPLORE / INTERACT` 候选中选择 | 自动使用 `full_mllm_interactive_exploration.yaml`、`full_mllm_mapping.yaml`，并启用属性推理 |
| `full_mllm_object_goal` | 全 MLLM 交互目标导航 | `dynamic_mllm / mllm_score / mllm_skill_verified` | 默认运行时注入 `random_far_container_object`，只向决策侧公开目标物体类别；目标到达并通过可见性验证后完成 | 自动使用 `full_mllm_object_goal_runtime.yaml`、`full_mllm_mapping.yaml`，并启用属性推理，不需要外部配置覆盖 |
| `full_mllm_object_goal_apple` | House 7 固定 apple 的全 MLLM 交互目标导航 | `dynamic_mllm / mllm_score / mllm_skill_verified` | 固定目标仅为 `apple`，默认禁止运行时目标覆盖，不公开冰箱或交互要求 | 自动使用 `full_mllm_object_goal_apple.yaml` 和 `full_mllm_house7_mapping.yaml`，用于 House 7 隐式容器发现回归 |
| `frontier_only` | 纯导航 frontier 对照 | 不启动 semantic decision | 无目标；以 frontier completion 为准 | 不执行语义候选决策和交互，用于 pure-nav baseline |
| `semantic_interaction_object_goal` | 规则交互目标导航；`semantic_interaction_object_goal` | 默认 `dynamic_rule / rule_cost / rule_verified` | 默认运行时注入 `random_far_container_object`，决策侧只收到目标物体类别；目标到达并通过可见性验证后完成 | 自动使用 `object_goal_runtime.yaml`，强制关闭容器；可通过显式 override 切换为全 MLLM |
| `object_goal_runtime` | 与 `semantic_interaction_object_goal` 相同的运行时目标导航 | 默认三模块规则实现 | 默认 `random_far_container_object`，公开上下文隐藏容器关系 | 面向批处理脚本保留的别名，自动使用 `object_goal_runtime.yaml` |
| `object_goal_rule` | 配置文件固定目标的规则导航 | 三模块均为规则实现 | 默认目标为 fridge/refrigerator；可用 `SEMANTIC_DECISION_OVERRIDE` 替换为 apple 等目标 | 自动使用 `object_goal_fridge.yaml` 和 House 7 交互几何；严格目标完成配置应显式包含 `mission.mode=semantic_interaction_object_goal` |
| `object_goal_model_mock` | 固定 fridge 目标的模型接口测试 | Module 2 使用 `model.mode=mock`，不请求真实模型 | 固定配置目标 | 自动使用 `object_goal_fridge_model_mock.yaml`，用于验证模型协议、选择和回退链路，不用于真实 MLLM 指标 |

补充约定：

- `semantic_interaction_object_goal`、`object_goal_runtime` 和 `full_mllm_object_goal` 会在未设置 `RUNTIME_TARGET_MODE` 时自动改为 `random_far_container_object`；其余 METHOD 默认是 `none`。
- 固定目标配置必须显式设置 `RUNTIME_TARGET_MODE=none`，否则运行时目标可能覆盖配置目标。
- `full_mllm_object_goal_apple` 已在 METHOD 内固定 `RUNTIME_TARGET_MODE=none`，因此 House 7 apple 回归不需要额外设置目标模式或 YAML override。
- `random_far_container_object` 会用完整容器关系构造任务，但默认通过 `public_object_goal_context()` 只向决策/LLM 发布目标物体类别，并把 `require_interaction` 置为 `false`。容器类别、source name 和真实交互要求保存在 `target_selection.json` 的 `private_target_context` 中，用于任务构造和离线评测；只有显式启用 `reveal_container_context` 时才公开。
- 每次真实模型运行的延时、token、错误与 TPS 写入 `<output_dir>/mllm_metrics.jsonl`。
- 多场景脚本 `run_semantic_interaction_exploration_batch.py` 接受 `container_exploration`、`object_goal_runtime`、`full_mllm_exploration` 和 `full_mllm_object_goal`。容器目标队列默认使用 `full_mllm_object_goal`。

### 5.3.2.2 全 LLM 交互探索

该模式不创建 object goal，模块 1、2、3 全部使用 MLLM：

```bash
cd /home/user/ldl/molmospaces-semantic-decision
conda activate mlspaces

ROS_MASTER_URI=http://127.0.0.1:12835 \
SEMANTIC_MODEL_ENV_FILE=$PWD/.env \
METHOD=full_mllm_exploration \
HOUSE_IND=7 \
SCENE_SEED=7 \
USE_FIXED_ROUTE=false \
RUNTIME_TARGET_MODE=none \
TASK_HORIZON=1000 \
INITIAL_DOOR_STATE=closed \
ENABLE_RECORDING=true \
SIM_TIMEOUT_S=1800 \
zsh scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
outputs/house7_full_mllm_exploration
```

自动设置：

- 决策配置：`scripts/InteractiveNav/configs/semantic_decision/full_mllm_interactive_exploration.yaml`
- 映射配置：`scripts/InteractiveNav/configs/semantic_decision/full_mllm_mapping.yaml`
- `ENABLE_ATTRIBUTE_INFERENCE=true`
- Module 1 `dynamic_mllm`、Module 2 `mllm_score`、Module 3 `mllm_skill_verified`

注意：`METHOD=semantic_interaction_exploration` 是规则版本，不是全 MLLM 版本。

### 5.3.2.3 全 LLM 运行时交互目标导航

直接使用内置的 `full_mllm_object_goal`。该 METHOD 会自动选择全 MLLM 决策和映射配置、启用属性推理，并在未指定目标模式时采样远距离容器内物体：

```bash
ROS_MASTER_URI=http://127.0.0.1:12836 \
SEMANTIC_MODEL_ENV_FILE=/home/user/ldl/molmospaces/.env \
METHOD=full_mllm_object_goal \
HOUSE_IND=7 \
SCENE_SEED=7 \
USE_FIXED_ROUTE=false \
TASK_HORIZON=1000 \
INITIAL_DOOR_STATE=closed \
ENABLE_RECORDING=true \
SIM_TIMEOUT_S=1800 \
zsh scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
outputs/house7_full_mllm_object_goal
```

`full_mllm_object_goal_runtime.yaml` 相比全 LLM 探索配置增加或改变：

- `mission.mode=semantic_interaction_object_goal`
- 生成并保留 target `NAVIGATE` 候选
- `target_relevance_weight=3.0`
- top-K 配额为 1 个 `NAVIGATE`、3 个 `INTERACT`、4 个 `EXPLORE`
- 使用目标到达、距离容差和可见性作为完成验证
- 目标稳定出现后允许抢占正在执行的普通探索

### 5.3.2.4 规则模式示例

单场景完整规则交互探索：

```bash
cd /home/user/ldl/molmospaces-semantic-decision
conda activate mlspaces

ROS_MASTER_URI=http://127.0.0.1:11521 \
HOUSE_IND=7 \
SCENE_SEED=7 \
METHOD=semantic_interaction_exploration \
USE_FIXED_ROUTE=false \
TASK_HORIZON=1000 \
ENABLE_RECORDING=true \
INITIAL_DOOR_STATE=closed \
  zsh scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
  outputs/house7_semantic_interaction_exploration
```

单场景规则运行时 Obj-goal：

```bash
ROS_MASTER_URI=http://127.0.0.1:11522 \
HOUSE_IND=7 \
SCENE_SEED=7 \
METHOD=semantic_interaction_object_goal \
USE_FIXED_ROUTE=false \
TASK_HORIZON=1000 \
ENABLE_RECORDING=true \
INITIAL_DOOR_STATE=closed \
  zsh scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
  outputs/house7_object_goal_runtime
```

固定冰箱目标使用 `METHOD=object_goal_rule`；模型接口 mock 使用 `METHOD=object_goal_model_mock`；纯 frontier 对照使用 `METHOD=frontier_only`。常用配置位于：

```text
scripts/InteractiveNav/configs/semantic_decision/interactive_exploration.yaml
scripts/InteractiveNav/configs/semantic_decision/full_mllm_interactive_exploration.yaml
scripts/InteractiveNav/configs/semantic_decision/full_mllm_object_goal_runtime.yaml
scripts/InteractiveNav/configs/semantic_decision/full_mllm_object_goal_apple.yaml
scripts/InteractiveNav/configs/semantic_decision/full_mllm_mapping.yaml
scripts/InteractiveNav/configs/semantic_decision/semantic_controlled_explore.yaml
scripts/InteractiveNav/configs/semantic_decision/semantic_interaction_nav.yaml
scripts/InteractiveNav/configs/semantic_decision/object_goal_runtime.yaml
scripts/InteractiveNav/configs/semantic_decision/object_goal_fridge.yaml
scripts/InteractiveNav/configs/semantic_decision/object_goal_fridge_model_mock.yaml
```

### 5.3.3 冻结 V3 单 episode 可视化评测

冻结 benchmark 的 ROS object-goal 评测使用专用单 episode 入口。它会自行启动独立 ROS master、ROS 算法栈和 recorder；每次都强制输出六联图视频与俯视结果图。不要把多个 episode 放进同一次调用，以免把不同 episode 的 ROS 轨迹混入同一份 recorder 产物。

```bash
ROS_MASTER_URI=http://127.0.0.1:11311 \
MAX_STEPS=1000 \
VIDEO_FPS=5 \
zsh scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_test.zsh \
  outputs/v3_container_episode_1000 1000
```

该入口固定为 `METHOD=full_mllm_object_goal`：`object_goal_v3_full_mllm.yaml` 必须包含
`dynamic_mllm / mllm_score / mllm_skill_verified`。其中 `POLICY=ros_object_goal_rule`
只是受限 GT、ROS 和 opaque interaction 的评测适配器名，不是规则语义方法，不能替换为
其他 policy。需要替换 MLLM 配置时，可传 `SEMANTIC_DECISION_OVERRIDE` 与
`SEMANTIC_MAPPING_OVERRIDE`，但脚本会校验三模块仍为完整 MLLM。

对抽屉目标，MLLM 输出的是高层 `drawer_scan` 宏动作和可见抽屉的归一化区域；V3 对外
仍只发送 opaque `open(object_id)`，由可信评测侧私有执行“低头 → 从上到下逐格打开 →
观察/更新受限感知 → 关闭 → 下一格 → 恢复视角”。扫描期间已验证的开度和目标可见性会
被私有地计入评测，因此关闭抽屉不会被误判为交互或目标失败。空/非法视觉区域会直接
失败，不会退化成扫描全部 simulator drawer；每个 force substep 都锁定机身与上肢，异常时
会先尝试关闭已触及抽屉并恢复视角。

完成后必须检查：

- `debug/videos/overview_6panel.mp4`：ROS 相机、OCC、房间/交互、全局/局部代价图、语义图与拓扑图六联视频。
- `eval/episodes/<episode>/episode_topdown.png`：场景底图、真实探索轨迹、起点、GT target、GT/实际交互及 oracle 路径。
- `eval/episodes/<episode>/episode_result.json`：冻结 V3 的正式评测结果。

该入口默认不重复缓存 head-camera 视频，以避免同一 episode 同时保存两份大视频。若需要保留它作为 force interaction 的补充第一视角，额外传 `RECORD_HEAD_CAMERA=true`。

多场景交互实验使用专用批处理脚本。每个 worker 是独立子进程，拥有独立 `ROS_MASTER_URI`；场景按 round-robin 分片，worker 内部串行运行分到的场景：

```bash
python scripts/InteractiveNav/run_semantic_interaction_exploration_batch.py \
  --output-dir outputs/interaction_explore_houses_0_9 \
  --house-inds 0 1 2 3 4 5 6 7 8 9 \
  --workers 2 \
  --base-master-port 12420 \
  --task-horizon 1000 \
  --method container_exploration \
  --scene-timeout-s 1500
```

当前批处理脚本中的 `container_exploration` 对应完整语义交互探索配置；运行时 Obj-goal 使用：

```bash
python scripts/InteractiveNav/run_semantic_interaction_exploration_batch.py \
  --output-dir outputs/object_goal_houses_0_9 \
  --house-inds 0 1 2 3 4 5 6 7 8 9 \
  --workers 2 \
  --base-master-port 12520 \
  --task-horizon 1000 \
  --method object_goal_runtime \
  --scene-timeout-s 1500
```

容器内物体交互导航队列使用专用脚本。脚本先顺序扫描场景，发现严格位于可交互容器内部的物体后立即入队；扫描满 5 个场景后才启动 5 个独立 ROS worker。公开目标上下文只包含目标物体，不包含其容器身份或 `require_interaction=true`：

```bash
python scripts/InteractiveNav/run_container_goal_queue.py \
  --output-dir outputs/container_object_goal_queue \
  --house-start 0 \
  --house-count 10 \
  --workers 5 \
  --warmup-scenes 5 \
  --task-horizon 1000 \
  --scene-timeout-s 1500 \
  --gt-step-interval 5 \
  --gt-max-distance-m 6.0 \
  --gt-min-visible-pixels 16 \
  --env-file .env \
  --allow-failures
```

每个任务默认启用 `semantic_interaction_object_goal`、模块 1/2/3 的 MLLM 配置、关闭初始门和容器、快速原子交互及六联图录像。批次根目录保存 `scan_results.json`、`queue_state.json`、`aggregate_metrics.json` 和 `summary.csv`；每个任务目录保存目标选择、语义结果、MLLM 指标和 `videos/overview_6panel.mp4`。

例如 10 个场景和 2 个 worker 的分配为：

```text
worker 0: House 0, 2, 4, 6, 8
worker 1: House 1, 3, 5, 7, 9
```

批量实验可通过环境变量控制录制。完整六联图实验使用 `ENABLE_RECORDING=true`；只统计成功率、覆盖率和耗时的快速实验使用：

```bash
ENABLE_RECORDING=false \
python scripts/InteractiveNav/run_semantic_interaction_exploration_batch.py \
  --output-dir outputs/fast_houses_0_9 \
  --house-inds 0 1 2 3 4 5 6 7 8 9 \
  --workers 2 \
  --task-horizon 1000 \
  --method container_exploration
```

批处理常用参数：

- `--resume`：复用已有有效结果，跳过已完成场景。
- `--allow-failures`：允许部分场景失败而不让批处理整体返回非零状态。
- `--dry-run`：只输出分片、端口和命令，不启动仿真。
- `--memory-sample-interval-s`：资源采样间隔，默认 `2s`。
- `--base-master-port`：worker 端口起点，第 `N` 个 worker 使用 `base + N`。

主要输出包括每个 `house_NNNN/` 下的 `batch_task.log`、`memory_by_step.csv`、`semantic_exploration_result.json` 和可选的 `videos/overview_6panel.mp4`，以及批次根目录的 `summary.json`、`summary.csv` 和 `aggregate_metrics.json`。

内存建议：保存六联图时优先使用最多 `2 workers`；关闭录制后可从 `2 workers` 开始逐步增加。不要直接以 CPU 核数决定 worker 数量，应以 simulator、recorder 和主机可用内存的实测峰值为准。

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

7 号场景检查会在 MuJoCo 内部设置真实门关节，但只向 graph 发布最小 GT 五字段；
随后以 executor 语义结果 `object_id + state` 回写闭合/打开状态，并验证门图状态和 OCC 清空结果：

```bash
conda activate mlspaces
python scripts/InteractiveNav/test_semantic_door_occ_house7.py \
  --house-ind 7 \
  --output /tmp/semantic_door_occ_house7.json
```

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


### 8.5.1 House 7 冰箱内物体 Obj-goal 图像采集

House 7 的冰箱对象为 `refrigerator_4d8cd69ca487b76cae801cfb0248a055_1_0_6`，此前 GT 语义图中确认其内部有 `potato`、`apple` 和 `lettuce`。下面以 `apple` 为目标，初始固定在 `house7_force_route_01` 的起点，门和容器关闭，由规则语义决策自主探索、开门、打开冰箱并导航到苹果。

```bash
ROS_MASTER_URI=http://127.0.0.1:12835 \
METHOD=object_goal_rule \
HOUSE_IND=7 \
USE_FIXED_ROUTE=true \
ROUTE_ID=house7_force_route_01 \
TASK_HORIZON=1000 \
INITIAL_DOOR_STATE=closed \
SEMANTIC_DECISION_OVERRIDE=scripts/InteractiveNav/configs/semantic_decision/object_goal_apple.yaml \
GT_STEP_INTERVAL=1 \
GT_MAX_DISTANCE_M=6.0 \
VIDEO_FPS=15 \
VIDEO_PANEL_WIDTH_PX=640 \
EXTERNAL_VIDEO_WIDTH_PX=1024 \
ENABLE_EXTERNAL_VIDEO=true \
VIDEO_FRAME_JOB_QUEUE_SIZE=1024 \
ARTIFACT_WRITE_QUEUE_SIZE=4096 \
VIDEO_HISTORY_SIZE=1024 \
IMAGE_QUEUE_SIZE=16 \
OBSERVATION_QUEUE_SIZE=16 \
EXTRA_IMAGE_QUEUE_SIZE=16 \
RECORDER_SHUTDOWN_GRACE_S=600 \
CLEAN_INTERMEDIATE=false \
SIM_TIMEOUT_S=1800 \
  scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
  outputs/house7_object_goal_apple_route01
```

主要输出：`videos/overview_6panel.mp4`、`debug/semantic_keyframes/`、`debug/graph/` 和 `semantic_exploration_result.json`。

模块 2 使用 LLM 直接选择具体 `NAVIGATE / INTERACT / EXPLORE` subgoal 时，使用独立配置保留规则基线。模型连接参数继续从仓库 `.env` 或 `SEMANTIC_MODEL_*` 环境变量读取：

```bash
ROS_MASTER_URI=http://127.0.0.1:12835 \
METHOD=semantic_interaction_object_goal \
RUNTIME_TARGET_MODE=none \
HOUSE_IND=7 \
USE_FIXED_ROUTE=true \
ROUTE_ID=house7_force_route_01 \
TASK_HORIZON=1000 \
INITIAL_DOOR_STATE=closed \
SEMANTIC_DECISION_OVERRIDE=scripts/InteractiveNav/configs/semantic_decision/object_goal_apple_module2_mllm.yaml \
GT_STEP_INTERVAL=1 \
GT_MAX_DISTANCE_M=6.0 \
VIDEO_FPS=15 \
VIDEO_PANEL_WIDTH_PX=640 \
CLEAN_INTERMEDIATE=false \
SIM_TIMEOUT_S=1800 \
  scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
  outputs/house7_object_goal_apple_module2_mllm_route01
```

该配置固定 `mission.mode=semantic_interaction_object_goal`，并用 `RUNTIME_TARGET_MODE=none` 禁止运行时随机容器目标覆盖，避免向模型泄露目标容器或强制交互信息。仅消融模块 2：模块 1 保持 `dynamic_rule`，模块 3 保持 `rule_verified`。决策 trace 中应满足正常模型路径的 `model_selected_candidate_id == executed_candidate_id`；若模型响应期间候选失效，则会记录 `candidate_validation_reason` 和 `stale_fallback_used`。

House 7 中仅公开 `apple` 类别、模块 1/2/3 全部使用 MLLM 的隐式容器发现测试使用：

```bash
ROS_MASTER_URI=http://127.0.0.1:12836 \
SEMANTIC_MODEL_ENV_FILE=/home/user/ldl/molmospaces/.env \
METHOD=full_mllm_object_goal_apple \
HOUSE_IND=7 \
SCENE_SEED=7 \
USE_FIXED_ROUTE=true \
ROUTE_ID=house7_force_route_01 \
TASK_HORIZON=1000 \
INITIAL_DOOR_STATE=closed \
ENABLE_RECORDING=true \
ENABLE_EXTERNAL_VIDEO=true \
EXTERNAL_VIDEO_OVERLAY=true \
PAPER_FRAME_EXPORTS=true \
EXTRA_IMAGE_QUEUE_SIZE=0 \
GT_STEP_INTERVAL=1 \
GT_MAX_DISTANCE_M=6.0 \
GT_MIN_VISIBLE_PIXELS=16 \
VIDEO_FPS=15 \
VIDEO_PANEL_WIDTH_PX=640 \
EXTERNAL_VIDEO_WIDTH_PX=1024 \
CLEAN_INTERMEDIATE=false \
SIM_TIMEOUT_S=1800 \
  zsh scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
  outputs/house7_full_mllm_apple_hidden_container
```

`METHOD=full_mllm_object_goal_apple` 内部选择的 `full_mllm_object_goal_apple.yaml` 不包含 apple 实例 ID、冰箱 ID、容器类别或强制交互提示。内部选择的 `full_mllm_house7_mapping.yaml` 只在执行侧合并全 MLLM ablation 与 House 7 已校准的冰箱正面交互位姿，不把容器关系写入任务目标或 LLM prompt。运行时目标采样同样默认只向 `/semantic_decision/target` 发布目标物体类别；完整容器关系保存在 `target_selection.json` 的 `private_target_context` 和候选记录中，仅用于任务构造与离线评测。只有队列脚本显式传入 `--reveal-container-context` 时才公开容器上下文。

论文逐帧导出使用 `PAPER_FRAME_EXPORTS=true`。该模式通过
`/molmo_spaces/step_sync` 为每个 sim step 保存一组严格对齐的图片，并区分“论文静帧”和
“带调试信息的视频帧”：

- 无文字外部相机 PNG（`1024x576`）：`debug/videos/external_camera_frames/`
- 带上方诊断文字、用于外部相机视频的 PNG：`debug/videos/external_camera_overlay_frames/`
- 模拟器原始分辨率、无文字外部相机 PNG：`debug/videos/external_camera_raw_frames/`
- 无上方白条、无标题/step、无右下角 step 的图 3：`debug/videos/room_interaction_frames/`
- 无上方白条、无标题/step、无右下角 step 的图 6：`debug/videos/semantic_topology_frames/`
- 保留标题、任务目标、当前 subgoal 和 step 信息的六面板视频：`videos/overview_6panel.mp4`
- 保留上方诊断文字的外部相机视频：`debug/videos/external_camera.mp4`

2026-07-28 成功长测输出：
`outputs/house7_full_mllm_apple_paper_1000_20260728_v2`。任务只公开 `apple` 类别，
冰箱在 step 835 成功打开，apple 在 step 934 满足可见性完成条件，保留 10 个成功后帧并在
step 944 提前结束；`target_goal_success=true`、
`target_object_visible_navigation_success=true`、`overall_success=true`。六面板、外部相机、
图 3、图 6、sim-step 与离线 composite 均保存 944 张/帧，两个写入队列丢帧数均为 0，
离线六面板对齐为 944/944 exact step match。

2026-07-27 回归：冰箱物理正面轴校准为 `+X`，节点图记录固定交互位姿
`[8.254459, 1.053060, 3.141593]`。本次机器人实际交互位姿为
`[8.535121, 1.075141, -2.944551]`，位置误差 `0.282 m`、朝向误差
`0.197 rad`，通过执行端位姿门控并成功从 `closed` 切换到 `open`。后续外部相机默认
采用手动调好的 `CAMERA_REL`：position `[-0.779295308248162, 0.9640243904369644,
1.600000023841858]`、yaw `-34.729731964609215 deg`、pitch
`-16.0519210588521 deg`、FOV `65 deg`。运行脚本中的等价 look-at offset 为
`[0.010510587056107079, 0.4165302897166183, 1.3234916922873792]`。相机水平距离
约 `1.24 m`。原始外部相机 PNG 与视频均为 `1024x576`。节点图右侧决策文字框已移除，
room 横向排列优先依据 portal-room 拓扑连接度，将 House 7 的 livingroom 放在
bedroom 与 kitchen 之间。

2026-07-28 House 7 论文首图相机回归：左后上方位姿会让机器人位于画面右侧，
同时保持冰箱正面和内部物体可见。当前稳定跟随相机配置为
`position_robot=[-1.45, 1.30, 1.90]`、
`lookat_offset=[0.05, 0.40, 1.38]`、FOV `65 deg`，由
`scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh` 默认使用。
House 7 apple 800-step 回归输出为
`outputs/house7_apple_rear_left_compromise_800_20260728_v4`，最终 apple 目标成功，
外部最终 PNG 为 `1024x576`。

#### 手动移动机器人并调外部相机

历史手动可视化工具位于独立实验 worktree：

- `/home/user/ldl/molmospaces-exp-setting/scripts/InteractiveNav/run_manual_interactive_nav_test.py`
- `/home/user/ldl/molmospaces-exp-setting/scripts/InteractiveNav/manual_interactive_nav_camera.py`
- `/home/user/ldl/molmospaces-exp-setting/scripts/InteractiveNav/manual_interactive_nav_policy.py`

启动 House 7 冰箱内 apple 场景：

```bash
cd /home/user/ldl/molmospaces-exp-setting
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate mlspaces
export MPLCONFIGDIR=/tmp/molmospaces-matplotlib
python scripts/InteractiveNav/run_manual_interactive_nav_test.py \
  --catalog scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_train_100/raw/mixed_rough/mixed_rough_catalog.json \
  --case_id 'mixed_h7__refrigerator_4d8cd69ca487b76cae801cfb0248a055_1_0_6__apple_87e4661e0aaedff69f751b5ac78bd93c_1_0_6' \
  --state_preset visualization \
  --camera_mode robot_over_shoulder \
  --camera_position_offset_robot -0.779295308248162 0.9640243904369644 1.600000023841858 \
  --camera_lookat_offset_robot 0.010510587056107079 0.4165302897166183 1.3234916922873792 \
  --camera_fov 65 \
  --camera_translation_step 0.05 \
  --camera_rotation_step_deg 1.0 \
  --capture_dir /tmp/house7_manual_camera_tune
```

控制键：`W/S/A/D` 移动机器人，`I/K/J/L` 平移相机，`;/'` 调 yaw，`./` 调
pitch，`O/P` 打开/关闭交互对象，`V` 截图，`C` 重置相机，`R` 重置机器人，
`Esc` 退出。调好后终端打印的 `CAMERA_REL` 可直接记录；yaw/pitch 转成跟随相机
look-at 时，只需用 `lookat = position + [cos(pitch)cos(yaw), cos(pitch)sin(yaw),
sin(pitch)]`。

注意：跟随相机配置不能同时设置固定 `camera_quaternion`。`RobotMountedCamera` 在
`camera_quaternion` 非空时会忽略 `lookat_offset`；此前正式仿真仍传入
`[0.5, 0.5, -0.5, -0.5]`，实际视线因此被固定为机器人坐标系正前方 `[1, 0, 0]`，
没有使用遥控记录的 yaw/pitch。该覆盖已移除。上述 CAMERA_REL 对应的正确机器人系
forward 为 `[0.7898058953, -0.5474941007, -0.2765083316]`，正式跟随相机现在由
position 与 look-at 两个机器人系 offset 生成，与遥控脚本保持一致。

2026-07-27 耗时回归：三路并行 800-step 测试输出位于
`outputs/house7_perf_parallel3_800_20260727_w{1,2,3}`。三次均完成 800 个 sim step、
800 行 `sim/step_timing.jsonl`、800 张 sim-step PNG 和 800 帧 6-panel 视频；主循环
均值分别为 `1.031`、`1.042`、`1.023 s/step`。合并稳定阶段日志显示，外部相机
同步 ROS 发布平均占 `510 ms/step`，而 sim-step PNG 队列入队仅占 `0.03 ms/step`。

性能修复后，第一视角和外部相机分别使用独立的异步 ROS publisher queue，外部相机
回调不再与 6-panel 渲染共用锁，等待导航命令期间也不再重复发布录像 RGB。最终
100-step 回归位于 `outputs/house7_perf_short_all_async_lossless_100_20260727`：稳定阶段
平均 `0.708 s/step`，外部图像 ROS 发布由 `441 ms` 降至 `1.48 ms`；100 个 sim step
对应 100 张 sim-step PNG、100 张外部原始 PNG 和 100 张 topology panel PNG，且
`artifact_write_dropped_jobs=0`、`video_frame_jobs_dropped=0`。

2026-07-27 apple object-goal 修复：原先实时 GT 消息会携带每个实例的完整像素坐标，
1024x576 下单条 ROS JSON 可达到约 `0.6-1.3 MB`；同时 GT publisher 复用了深度
为 16 的通用 ROS 队列，导致冰箱打开后的新目标观测排在旧消息后。现在 GT 只发布
`bbox_2d + visible_pixels + visible_fraction + box_3d`，发布端和语义映射订阅端均改为
latest-only 队列 1。三路最终测试中，apple 从原始 GT 捕获到节点图 `NEW_NODE` 的
墙钟延迟降为 `7.8-9.1 s`，此前并行长测为 `29-53 s`。

该次回归曾通过精确 apple 与 refrigerator 绑定验证目标状态机。合并后的通用配置仅暴露
apple 类别，不再向决策模块提供目标容器 ID 或强制交互信息；目标一旦稳定进入图中，便由
事件驱动优先级抢占探索。容器内目标复用图关系推导出的父容器及其已验证交互位姿，不再
根据 apple 几何中心生成可能位于冰箱内部或侧面的导航点；机器人已在该位姿时直接进入
可见性验证，不再被无位移看门狗误判。完成后发布 `target_goal_succeeded` 并触发仿真提前退出。

三路最终并行输出：

- `outputs/house7_apple_latest_parallel3_800_20260727_w1`：step 451 成功，保留 10 step 后在 461 结束。
- `outputs/house7_apple_latest_parallel3_800_20260727_w2`：step 339 成功，保留 10 step 后在 349 结束。
- `outputs/house7_apple_latest_parallel3_800_20260727_w3`：step 385 成功，保留 10 step 后在 395 结束。

三路均满足：冰箱交互成功、交互位姿验证通过、apple 当前稳定可见、目标导航成功、
`target_goal_success=true`、`target_container_interaction_success=true`、
`overall_success=true`。`sim_step_frames/manifest.jsonl` 与最终 6-panel 视频帧数一一对应，
且 `video_frame_jobs_dropped=0`、`artifact_write_dropped_jobs=0`、渲染队列覆盖次数为 0。
外部相机原始帧在 `debug/videos/external_camera_raw_frames/`，图 3 在
`debug/videos/room_interaction_frames/`，节点图在
`debug/videos/semantic_topology_frames/`，按 sim step 对齐后的完整 6-panel PNG 在
`videos/offline_composite_frames/`。

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

---

## 11. MLLM 模块化交互导航

三模块消融配置位于：

```text
scripts/InteractiveNav/configs/semantic_decision/ablations/
```

模式定义：

- 模块一：`static_semantic` / `dynamic_rule` / `dynamic_mllm`
- 模块二：`rule_cost` / `mllm_score`
- 模块三：`direct_atomic` / `rule_verified` / `mllm_skill_verified`

House 7 规则基线示例：

```bash
roslaunch nav_pkg semantic_interactive_ablation.launch \
  house_ind:=7 \
  task_horizon:=1000 \
  ablation_config:=$(rospack find semantic_decision_py_pkg)/../../../scripts/InteractiveNav/configs/semantic_decision/ablations/current_rule_baseline.yaml
```

完整 MLLM 示例需要同时启动属性推理：

```bash
roslaunch nav_pkg semantic_interactive_ablation.launch \
  house_ind:=7 \
  task_horizon:=1000 \
  enable_attribute_inference:=true \
  model_name:=<MODEL_NAME> \
  ablation_config:=<REPO>/scripts/InteractiveNav/configs/semantic_decision/ablations/full_mllm.yaml
```

题库绑定 House 场景图像：

```bash
python scripts/InteractiveNav/collect_mllm_benchmark_samples.py \
  --question-bank scripts/InteractiveNav/mllm_benchmark/question_bank.json \
  --source-dir <IMAGE_DIR> \
  --output-dir <BOUND_QUESTION_DIR> \
  --bind attribute_closed_fridge=<CLOSED_FRIDGE_IMAGE> \
  --bind attribute_open_portal=<OPEN_PORTAL_IMAGE> \
  --bind verify_fridge_opened=<OPEN_FRIDGE_IMAGE> \
  --bbox attribute_closed_fridge=<X0,Y0,X1,Y1> \
  --bbox attribute_open_portal=<X0,Y0,X1,Y1> \
  --bbox verify_fridge_opened=<X0,Y0,X1,Y1>
```

运行指定四模型评测：

```bash
PYTHONPATH="Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts" \
python scripts/InteractiveNav/evaluate_mllm_question_bank.py \
  --env-file .env \
  --question-bank <BOUND_QUESTION_DIR>/question_bank_bound.json \
  --models gpt-5.3-codex-spark qwen3.6-flash qwen3.5-35b-a3b deepseek-v4-flash \
  --timeout-s 15 \
  --reasoning-effort low \
  --image-detail low \
  --crop-margin-ratio 0.10 \
  --crop-max-side-px 512 \
  --output <OUTPUT_DIR>/model_comparison.json
```

默认按角色限制输出预算：属性 `384`、Subgoal `128`、技能规划 `192`、视觉验证 `192`。
输出包含整体及各角色的准确率、有效响应率、逐题耗时、token、reasoning token、可见输出 TPS，并同时生成 CSV。
视觉属性和交互反馈只输入目标 `2D bbox` 裁切；交互反馈只使用交互后的单张目标图。

### 模块 3：历史图像视觉操作规划

使用运行时同一提示词与输出 schema，测试门操作方式和多抽屉中心点：

```bash
conda run -n mlspaces python scripts/InteractiveNav/evaluate_module3_visual_planning.py \
  --manifest scripts/InteractiveNav/configs/semantic_decision/module3_historical_eval.json \
  --env-file .env \
  --output outputs/module3_visual_planning_eval.json
```

无可用视觉模型时可用 `--mode mock --model mock-module3` 检查裁图、协议、schema 和评分链路；mock 结果不能作为视觉能力结论。

---

## 12. Native `NavToObj` 远程 A100 大规模测试

本节使用 `scripts/InteractiveNav/nav_to_obj_batch_manager.py` 调度
`run_native_nav_to_obj_eval.zsh`。每个 episode 都由一个独立 ROS master 和一次
原生 JSON evaluator 调用执行；不要把多个 episode 塞进同一个 ROS 进程。

### 12.1 结果口径：严格任务分数与 ROS-adapted 执行器分开标注

当前入口的最终成功与总数来自 MolmoSpaces 的 `NavToObjTask` 和
`native_eval_summary.json`，而不是 ROS topic、`move_base` 状态或日志关键字。执行器本身
是 ROS-adapted：它为映射打开 head-camera depth，并在 runtime replay spec 上应用 RBY1
导航手臂姿态；因此结果应写作“**native NavToObj 官方任务判定 × ROS-adapted 执行器**”，
不能称为未适配 ROS 的原版 policy 复现。

| 结果桶 | 必须满足 | 可否进入严格任务分数表 |
| --- | --- | --- |
| 严格任务 replay | 固定同一 benchmark SHA、同一 evaluator/launcher 哈希、统一的 horizon（下例为 500）、`--filter-missing-scene-objects` 未开启、`NATIVE_NAV_DYNAMIC_HORIZON=false` | 可以；表头仍注明 ROS-adapted 执行器 |
| ROS-adapted 兼容/筛选 | 资源版本不匹配时启用 `--filter-missing-scene-objects`，或采用动态缩短 horizon / 其他非统一测试预算 | 不可以；单独报告 compatibility/screening 结果和覆盖率 |

`--filter-missing-scene-objects` 会删去本地场景中不存在的 benchmark pose/joint 条目，属于
compatibility mode，绝不是 exact replay。`--task-horizon-steps 500` 是本批次所有方法必须
一致记录的公开预算；不可针对 episode 动态缩短后与严格桶混报。

### 12.2 远程机前置检查

先通过远程传输/SSH 工具登录 A100，再在**远程 shell** 中执行下面命令。
本机供连接工具使用的 SSH `.env` 文件仅供连接工具读取：不要 `source`、复制、写入
`--worker-env`，也不要把它传给 `--semantic-model-env-file`。后者只接受独立、非 SSH 凭据的
模型运行时 env 文件；本基线默认不传它。

```bash
cd /path/to/remote/worktree
export REPO="$(pwd -P)"
export MLSPACES_CACHE_DIR=/path/to/remote/molmo-cache
export MLSPACES_ASSETS_DIR=/path/to/remote/molmo-assets
export BENCHMARK_DIR="$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/NavToObjProcthor10kBench_20260112_json_benchmark"
export CONDA_SH=/path/to/remote/miniconda3/etc/profile.d/conda.sh
export CONDA_ENV=mlspaces
export MLSPACES_PYTHON=/path/to/remote/miniconda3/envs/mlspaces/bin/python
export ROS_SETUP="$REPO/Interactive-Nav-SG-nav/devel/setup.bash"
export BASE_PORT=15601

nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
test -x "$MLSPACES_PYTHON"
test -f "$CONDA_SH"
test -f "$ROS_SETUP"
test -x "$REPO/scripts/InteractiveNav/run_native_nav_to_obj_eval.zsh"
test -f "$REPO/scripts/InteractiveNav/nav_to_obj_batch_manager.py"
test -d "$BENCHMARK_DIR"
bash -n "$REPO/scripts/InteractiveNav/run_native_nav_to_obj_eval.zsh"
"$MLSPACES_PYTHON" "$REPO/scripts/InteractiveNav/nav_to_obj_batch_manager.py" --help
for slot in 0 1 2 3 4; do
  ss -ltnH "sport = :$((BASE_PORT + slot))"
done
```

为未来追加的 worker 保持同一环境，初始化时把**非敏感的路径和性能开关**固化为
`--worker-env`。这些键值会写入 `run_config.json`，所以其中不能放 token、密码、API key
或 SSH 配置。建议大规模吞吐阶段关闭录像；待抽检失败/成功样本时再单独重放并开录像。

### 12.3 idx 33/40 smoke：先验证 benchmark、评测代码和远程环境一致

下例只选择全局 benchmark episode 33 与 40，但调度顺序仍由 seed 随机化。先检查
`plan`，再以一个 worker 串行运行；应先人工检查两个 attempt 的 evaluator summary，确认后
再进入全量批次。

```bash
export SMOKE_ROOT="$REPO/outputs/native_nav_to_obj_smoke_33_40_$(date +%Y%m%d_%H%M%S)"

"$MLSPACES_PYTHON" "$REPO/scripts/InteractiveNav/nav_to_obj_batch_manager.py" init \
  --benchmark-dir "$BENCHMARK_DIR" \
  --run-root "$SMOKE_ROOT" \
  --episode-indices 33,40 \
  --seed 20260729 \
  --task-horizon-steps 500 \
  --base-ros-master-port "$BASE_PORT" \
  --ros-hostname 127.0.0.1 \
  --max-attempts-per-episode 1 \
  --episode-timeout-seconds 1800 \
  --worker-env "MLSPACES_CACHE_DIR=$MLSPACES_CACHE_DIR" \
  --worker-env "MLSPACES_ASSETS_DIR=$MLSPACES_ASSETS_DIR" \
  --worker-env "CONDA_SH=$CONDA_SH" \
  --worker-env "CONDA_ENV=$CONDA_ENV" \
  --worker-env "MLSPACES_PYTHON=$MLSPACES_PYTHON" \
  --worker-env "ROS_SETUP=$ROS_SETUP" \
  --worker-env "MUJOCO_GL=egl" \
  --worker-env "PYOPENGL_PLATFORM=egl" \
  --worker-env "ENABLE_RECORDING=false" \
  --worker-env "NATIVE_NAV_RECORD_VIDEOS=false" \
  --worker-env "SKIP_OFFLINE_VIDEO=true" \
  --worker-env "NATIVE_NAV_DYNAMIC_HORIZON=false"

"$MLSPACES_PYTHON" "$REPO/scripts/InteractiveNav/nav_to_obj_batch_manager.py" plan \
  --run-root "$SMOKE_ROOT" --count 2
"$MLSPACES_PYTHON" "$REPO/scripts/InteractiveNav/nav_to_obj_batch_manager.py" run \
  --run-root "$SMOKE_ROOT" --workers 1 --worker-slot-start 0 --worker-id-prefix a100-smoke \
  --cuda-visible-devices-list 0
"$MLSPACES_PYTHON" "$REPO/scripts/InteractiveNav/nav_to_obj_batch_manager.py" status \
  --run-root "$SMOKE_ROOT"
```

`init` 会写入 `run_config.json`、`episode_manifest.json` 和 SQLite ledger，并固定
benchmark SHA、全局 episode 索引、launcher/evaluator 哈希及运行配置。每次 worker 启动前
还会复核 benchmark 与入口代码；任一项漂移时应新建 `RUN_ROOT`，不能续用旧根目录。

smoke 通过的最低条件是：两条记录均有 `batch_result.json`，其中 `return_code=0`，且所引用
的 `native_eval_summary.json` 中 `total_count=1`；`official_success` 可为 0，它表示一次
有效但失败的 benchmark rollout，不等于基础设施失败。另应确认 summary 的
`native_filter_missing_scene_objects=false`，并保留 `run_config.json` 作为结果 provenance。

### 12.4 全量或可中断批次：随机未测 episode、可动态加 worker

不传 `--episode-indices` 即选择完整 benchmark。SQLite 会在 `init` 时以 `--seed` 将全局
episode 索引打乱并保存 `selection_rank`；每个 worker 以 `BEGIN IMMEDIATE` 原子领取下一个
pending episode。因此中断时，后续 worker 会先领取尚未开始的随机序列项，而不是从 0
顺序扫描，也不会重复领取正在运行或已终态的 episode。

```bash
export RUN_ROOT="$REPO/outputs/native_nav_to_obj_a100_strict_$(date +%Y%m%d_%H%M%S)"

"$MLSPACES_PYTHON" "$REPO/scripts/InteractiveNav/nav_to_obj_batch_manager.py" init \
  --benchmark-dir "$BENCHMARK_DIR" \
  --run-root "$RUN_ROOT" \
  --seed 20260729 \
  --task-horizon-steps 500 \
  --base-ros-master-port "$BASE_PORT" \
  --ros-hostname 127.0.0.1 \
  --max-attempts-per-episode 2 \
  --episode-timeout-seconds 1800 \
  --worker-env "MLSPACES_CACHE_DIR=$MLSPACES_CACHE_DIR" \
  --worker-env "MLSPACES_ASSETS_DIR=$MLSPACES_ASSETS_DIR" \
  --worker-env "CONDA_SH=$CONDA_SH" \
  --worker-env "CONDA_ENV=$CONDA_ENV" \
  --worker-env "MLSPACES_PYTHON=$MLSPACES_PYTHON" \
  --worker-env "ROS_SETUP=$ROS_SETUP" \
  --worker-env "MUJOCO_GL=egl" \
  --worker-env "PYOPENGL_PLATFORM=egl" \
  --worker-env "ENABLE_RECORDING=false" \
  --worker-env "NATIVE_NAV_RECORD_VIDEOS=false" \
  --worker-env "SKIP_OFFLINE_VIDEO=true" \
  --worker-env "NATIVE_NAV_DYNAMIC_HORIZON=false"

"$MLSPACES_PYTHON" "$REPO/scripts/InteractiveNav/nav_to_obj_batch_manager.py" plan \
  --run-root "$RUN_ROOT" --count 20
"$MLSPACES_PYTHON" "$REPO/scripts/InteractiveNav/nav_to_obj_batch_manager.py" run \
  --run-root "$RUN_ROOT" --workers 2 --worker-slot-start 0 --worker-id-prefix a100-initial \
  --cuda-visible-devices-list 0,1
```

若当前机器的对象/场景资源与 benchmark 记录的资产版本不完全一致，不要在上述严格桶上临时
打开兼容开关。另建一个清晰命名的 screening 根目录，例如
`native_nav_to_obj_a100_compat_*`，并在 `init` 时额外传入
`--filter-missing-scene-objects`，同时把
`--worker-env "NATIVE_NAV_DYNAMIC_HORIZON=true"` 及其动态 horizon 参数固化。
该根目录的结果仅用于运行稳定性、覆盖率和问题筛查，不能与严格桶累计或比较成功率。

`run` 在当前 shell 中启动一组 worker。需要扩容时，不要停止已有 worker；另开一个远程
shell，使用尚未占用的 slot。单个 worker 和一次追加多个 worker 分别为：

```bash
"$MLSPACES_PYTHON" "$REPO/scripts/InteractiveNav/nav_to_obj_batch_manager.py" worker \
  --run-root "$RUN_ROOT" --worker-id a100-extra-02 --worker-slot 2 \
  --cuda-visible-devices 2

"$MLSPACES_PYTHON" "$REPO/scripts/InteractiveNav/nav_to_obj_batch_manager.py" run \
  --run-root "$RUN_ROOT" --workers 2 --worker-slot-start 3 --worker-id-prefix a100-scale \
  --cuda-visible-devices-list 3,4
```

slot `N` 默认使用 `http://127.0.0.1:$((BASE_PORT + N))`。ledger 对 ROS master slot 也有
独立 lease，重复使用正在活动的 slot 会被拒绝；每次实际尝试又有单独的
`episodes/episode_<idx>/attempt_<n>_<token>/` 目录、`ROS_HOME` 和临时目录。不要手动设为
所有 worker 共用同一 `ROS_MASTER_URI`，也不要从别的进程占用该端口范围。
`--cuda-visible-devices` 是单 worker 运行时覆盖，优先于继承环境和 `--worker-env`，实际
绑定会写入该 attempt 的 `claim.json` 与 `batch_result.json`；`--cuda-visible-devices-list`
必须恰好为本次 `--workers` 数量提供一个逗号分隔的绑定。

### 12.5 监控、恢复与结果分母

```bash
"$MLSPACES_PYTHON" "$REPO/scripts/InteractiveNav/nav_to_obj_batch_manager.py" status \
  --run-root "$RUN_ROOT"
"$MLSPACES_PYTHON" "$REPO/scripts/InteractiveNav/nav_to_obj_batch_manager.py" reclaim \
  --run-root "$RUN_ROOT"
```

`status` 的 `counts`、`pending_never_started`、`running`、`active_worker_slot_count` 和
`official_successes` 是唯一批次总览来源；SQLite 开启 WAL，不能手工编辑其 status。失联
worker 的 episode lease 到期后可由下一 worker 重新领取；`reclaim` 只回收已过期 lease。
`--retry-failed` 是显式动作，只会重排未超过 attempt 上限的 launcher 失败项，绝不会重跑
已完成 episode。

- 尚未全部完成时，报告 `official_successes / completed` 为“已完成样本条件成功率”，同时报告覆盖率 `completed / selected_episode_count`、`failed`、`exhausted` 与 `pending`；不得把它写成完整 benchmark 成功率。
- 只有 `completed == selected_episode_count` 且 `failed=exhausted=running=pending=0` 时，才可报告最终 `official_successes / selected_episode_count`。其中 completed 中的 `official_success=0` 是正式失败样本，必须保留在分母。
- launcher 非零、超时、无有效 summary 的 `failed` 不是已评测的 0 分样本；先修复或明确单列，不得静默从分母剔除，也不得与 compatibility/screening 桶相加。
- 特别地，若 `return_code=0` 但 summary 的 `total_count=0`（常见于缺失 grasp/joint
  资源而 task sampler 跳过 episode），该记录是“不可评估的资产/采样兼容性排除”，不是导航
  失败，也不是基础设施成功。单列其数量和原因；在补齐资源前不要用 `--retry-failed` 重跑。

禁止项：严格桶不要启用 `--filter-missing-scene-objects`、`NATIVE_NAV_DYNAMIC_HORIZON=true`、逐
episode 的不同 horizon 或未记录的 `--command-template`；不要向 `--worker-env` 写入 SSH/模型
凭据；不要混用不同 benchmark、代码哈希、策略配置或 `RUN_ROOT`；不要依据 ROS 日志自行判
定成功，必须读取 evaluator 产生的 `native_eval_summary.json`。
