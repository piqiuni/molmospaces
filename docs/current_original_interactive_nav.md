# 当前原始交互导航运行方式

本文记录当前仓库中“原始交互导航”（实时仿真 + semantic graph + 规则/MLLM 决策）的运行入口，以及冻结 InteractiveNav V3 benchmark 的并行评测方式。

## 1. 原始交互导航单场运行

统一入口是：

```text
scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh
```

脚本名虽然带 `house7`，但通过 `HOUSE_IND` 可以运行其他 house。典型的全 MLLM 探索命令如下（路径按本机约定放在 `/home/ldl`）：

```bash
cd /home/ldl/molmospaces-exp-setting
RUN=/home/ldl/outputs/interactive-nav/manual_house0000_mllm_$(date +%Y%m%d_%H%M%S)
mkdir -p /home/ldl/tmp/interactive_nav_manual /home/ldl/.cache/interactive_nav_manual

TMPDIR=/home/ldl/tmp/interactive_nav_manual \
XDG_CACHE_HOME=/home/ldl/.cache/interactive_nav_manual \
CONDA_ENV=/home/ldl/conda_envs/mlspaces \
PYTHON_BIN=/home/ldl/conda_envs/mlspaces/bin/python \
SEMANTIC_MODEL_ENV_FILE=/home/ldl/molmospaces-exp-setting/.env \
MLLM_DECISION_TIMEOUT_S=30 \
SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S=30 \
ROS_MASTER_URI=http://127.0.0.1:13500 \
METHOD=full_mllm_exploration \
HOUSE_IND=0 SCENE_SEED=0 ROUTE_ID=house_0000 \
USE_FIXED_ROUTE=false ROUTE_NAV_CONFIG='' RUNTIME_TARGET_MODE=none \
TASK_HORIZON=500 SIM_TIMEOUT_S=3600 \
MAPPING_SCAN_SOURCE=organized_depth POINTCLOUD_STRIDE=1 \
INITIAL_DOOR_STATE=closed FORCE_CLOSE_CONTAINERS=true \
ENABLE_RECORDING=true \
bash scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh "$RUN" house_0000
```

运行链路为：

```text
run_nav_ros_sim.py
  -> realtime GT observation
  -> dynamic semantic interaction graph
  -> explore_py frontier navigation
  -> semantic candidates
  -> rule / MLLM decision
  -> behavior executor
  -> move_base / force interaction
  -> interaction_result 回写 graph
```

`METHOD=full_mllm_exploration` 会自动使用：

- `configs/semantic_decision/full_mllm_interactive_exploration.yaml`
- `configs/semantic_decision/full_mllm_mapping.yaml`
- Module 1=`dynamic_mllm`、Module 2=`mllm_score`、Module 3=`mllm_skill_verified`
- `ENABLE_ATTRIBUTE_INFERENCE=true`

注意：`METHOD=semantic_interaction_exploration` 是规则版本，不是全 MLLM 版本。

主要产物：

- `$RUN/semantic_exploration_result.json`：最终任务结果
- `$RUN/force_interaction_events.json`：开门、容器交互和失败原因
- `$RUN/debug/events.jsonl`：逐事件调试轨迹
- `$RUN/sim_step_frames/`：逐 simulator step 原始帧
- `$RUN/videos/overview_6panel.mp4`：离线六联视频
- `$RUN/roslaunch.log`：启动、节点和耗时日志

脚本默认启用 step-ready / capture-ack 同步：bridge 发布观测后等待决策节点 ready，动作完成后 recorder 提交当前 step，recorder ack 后 simulator 才进入下一步。并行运行时，每个实例必须使用不同的 `ROS_MASTER_URI` 端口和独立 `$RUN`。

## 2. 当前历史原始交互导航测试输出

以下是引用任务中已经实际使用过的结果目录；它们是**运行产物，不是 benchmark 输入**：

```text
/home/ldl/outputs/interactive-nav/latestfix_operfront_1000_20260903/
/home/ldl/outputs/interactive-nav/analysisfix_revert_1000_20260902/
/home/ldl/outputs/interactive-nav/portal_consensus_occ_6scene_2000_v6_20260901/
/home/ldl/outputs/interactive-nav/dwa100_h7fix_1000_20260908_final/
```

每个目录通常按 `house_0001`、`house_0003` 等子目录保存结果。不要把这些结果目录传给 `--benchmark`。

## 3. 冻结 InteractiveNav V3 benchmark 的单 episode 运行

单 episode 使用：

```bash
ROS_MASTER_URI=http://127.0.0.1:11311 \
MAX_STEPS=1000 \
VIDEO_FPS=5 \
bash scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_test.zsh \
  /home/ldl/outputs/v3_container_episode_1000 1000
```

该入口固定使用 `METHOD=full_mllm_object_goal`，正式结果位于：

```text
<output>/eval/episodes/<episode>/episode_result.json
<output>/eval/episodes/<episode>/episode_topdown.png
<output>/videos/overview_6panel.mp4
```

## 4. 并行 benchmark 是怎样运行的

批量入口是：

```text
scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_batch.py
```

它为每个 worker 创建独立 ROS master、`ROS_HOME`、日志目录、输出目录和进程组；通过 `--episode-indices` 分配 episode，通过 `--base-master-port` 分配端口。`--no-recording`（等价于 `--fast-eval`）关闭 recorder、逐 step 图片、视频和俯视图，只保留 `episode_result.json` 与资源遥测，适合 benchmark 成功率批测。

示例：

```bash
V3_ROOT=/absolute/path/to/v3_release/benchmark
COMMON=(
  --workers 10
  --base-master-port 12600
  --episode-indices $(seq 0 49)
  --max-steps 2000
  --step-budget-mode fixed
  --scene-timeout-s 7200
  --model-endpoints http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1
  --semantic-model-env-file /home/ldl/molmospaces-exp-setting/.env
  --mujoco-egl-devices 0 1
  --no-recording
)

/home/ldl/conda_envs/mlspaces/bin/python \
  scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_batch.py \
  --benchmark "$V3_ROOT/channel.json" \
  --output-dir /home/ldl/outputs/v3_channel_50 "${COMMON[@]}"
```

`container.json` 和 `mixed.json` 的调用方式相同。三类同时运行时不要都开 10 workers；当前双卡建议总计 10 workers：channel=4（端口 12600）、container=3（12610）、mixed=3（12620）。

批量输出通常包括：

```text
/home/ldl/outputs/v3_channel_50/
/home/ldl/outputs/v3_container_50/
/home/ldl/outputs/v3_mixed_50/
```

每个批次重点检查 `aggregate_metrics.json`、`resource_telemetry.csv` 及各 episode 的 `episode_result.json`。

`episode_result.json` 中的 `success` / `nav_success` 始终按冻结
benchmark 的 selected-instance 严格计分。对于目标描述尚未提供唯一实例约束的场景，
评测器另外记录 `goal_definition_relaxed_success`（以及对应的
`goal_definition_relaxed_instance_id`）；该字段只用于分析“到达同类目标”的情况，
不会改变正式 SR。

## 5. 当前 benchmark 在哪里

本机实际的正式冻结 V3 benchmark 已确认位于：

```text
/home/ldl/molmospaces/scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_val_release_v1_1/benchmark/
```

目录中有四个正式输入文件：

```text
benchmark.json   # 全部正式 episode
channel.json     # channel domain
container.json   # container domain
mixed.json       # mixed domain
```

该目录旁边的 `release_manifest.json` 显示 release id 为 `interactive-nav-v3-procthor10k-val-release-v1.1`。

## 6. Evo 评测俯视图输出

`run_full_mllm_interactive_nav_eval.py` 完成一个场景后，会在该场景目录第一层发布
最终用户产物：

```text
<batch>/<domain>/<episode>/episode_topdown.png
<batch>/<domain>/<episode>/episode_topdown.json
<batch>/<domain>/<episode>/overview_6panel.mp4
```

`attempt_XXX/` 内仍保留原始评测树、日志和产物作为可追溯证据；场景第一层文件是
面向查看与汇总的稳定路径。视频在同一文件系统上优先通过硬链接发布，避免重复占用
大容量存储。正常运行、`--resume` 和最终磁盘恢复都会补齐这些浅层产物。

全部 worker 完成并执行最终磁盘结果对账后，并行评测脚本还会自动生成批次级汇总：

```text
<batch>/topdown_gallery/       # 俯视图 PNG 与 JSON 副本
<batch>/video_gallery/         # 指向场景视频的相对软链接
<batch>/contact_sheet_all.png  # 全部俯视图总览，直接位于批次根目录
<batch>/artifact_gallery_summary.json
```

汇总过程不会移动 `attempt_XXX/` 或场景第一层的原始产物；缺失单场产物会记录在汇总
索引中，但不会阻断其他场景发布。

基础（非 ROS）评测入口
`scripts/InteractiveNav/run_interactive_nav_benchmark_eval.py` 默认在每个
`episode_result.json` 旁串行生成 `episode_topdown.png` 和同名 JSON。图中包含
benchmark 初始位姿、GT 目标/交互点、GT 计划路径、policy 的实际轨迹，以及从
实际轨迹到真实交互位置的橙色交互 approach path；可用
`--no-render-topdown` 关闭后处理。

V3 ROS 标准流程默认加载完整 GT 场景（优先使用场景旁的预计算 `*_map.png`），再将
最终 ROS occupancy 投影到全部 GT 可导航区域。底图中的白色为尚未观测的 GT
可导航区域，绿色为 mapped-free，红色为误标成 occupied 的 GT 可导航区域；标题和
同名 JSON 同时给出 whole-scene observed、mapped-free 与 false-occupied 比例。
标准流程启用 `--require-full-scene`，完整场景缺失时直接报错，避免用局部 ROS 地图
伪造接近 100% 的整体覆盖率。`TOPDOWN_ROS_ONLY=true` 只用于轻量调试；该模式仍可
画出坐标对齐的局部轨迹，但整体覆盖率会明确标为 unavailable。

legacy/raw 入口
`scripts/InteractiveNav/run_full_mllm_interactive_nav_eval.py --runner-mode raw`
通过 `run_house7_semantic_exploration_ros_test.zsh` 生成 `<attempt>/topdown.png`、
`<attempt>/topdown.json`。图层来自 `final_occ_map.yaml`、`trajectory.csv` 和
`force_interaction_events.json`。raw 默认按 `--raw-data-split val` 与冻结 benchmark
对齐；若显式使用 train 场景而 benchmark 是 val，renderer 会抑制不可靠的 GT 并在
metadata 的 `warnings` 中说明，而不会伪造目标位置。固定 route 时可直接显示
route YAML 中的 GT start/approach/goal。renderer 还会核对 raw/benchmark 的
`house_index`；若实际起点与 GT 起点偏差超过
`RAW_TOPDOWN_GT_START_SUPPRESS_M`（默认 2 m），会保留可核验的 GT 目标标记、但
抑制误导性的 GT route。图中的青色线为实际行走路径，橙色线为实际交互路径，
绿色/红色菱形分别表示成功/失败交互。

正式统计如下：

| 文件 / domain | episode 数 |
|---|---:|
| `channel.json` / channel | 1000 |
| `container.json` / container | 976 |
| `mixed.json` / mixed | 992 |
| `benchmark.json` / formal total | **2968** |

候选集原始数量为 3000，正式 release 排除了 32 条，唯一排除原因是 `initial_target_visibility_mismatch`。因此 `1000 + 976 + 992 = 2968`，四个文件的数量相互一致。

manifest 记录的正式 `benchmark/benchmark.json` SHA-256：

```text
91106ca03a73665b1344251cdc276cfdc931bc7066a877c938a4ca1e370efc9e
```

代码中的正式资源标识为：

```text
molmospaces://benchmarks/molmospaces-bench-v2/20240407/procthor-10k/NavToObjDataGenConfig/NavToObjProcthor10kBench_20260112_json_benchmark/benchmark.json
```

这表示资源来源；本地 release 已经落盘。仓库内另外存在的 `mlspaces_tests/data_generation/test_benchmark/benchmark.json` 只是测试夹具，不能替代正式 V3 release。

开始并行正式评测前，建议确认：

```bash
V3_ROOT=/home/ldl/molmospaces/scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_val_release_v1_1/benchmark
test -f "$V3_ROOT/benchmark.json"
test -f "$V3_ROOT/channel.json"
test -f "$V3_ROOT/container.json"
test -f "$V3_ROOT/mixed.json"
```

并行 batch 时，`--benchmark` 应直接指向其中一个 domain 文件，例如：

```bash
--benchmark /home/ldl/molmospaces/scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_val_release_v1_1/benchmark/channel.json
```

## 7. 运行前检查

1. 使用 `/home/ldl/conda_envs/mlspaces/bin/python` 和独立的 `/home/ldl/tmp`、`/home/ldl/.cache`。
2. 每个并行 worker 使用不同 ROS master 端口。
3. 每个 worker 使用独立输出目录，不共享 recorder 或 ROS_HOME。
4. 原始交互回归要检查完成 step、交互成功/失败、no-progress/timeout、视频 exact-step；不能只看脚本退出码。
5. benchmark 批测要记录 benchmark 文件路径、SHA、episode index、worker 数、模型 endpoint 和 step budget，保证结果可复现。
