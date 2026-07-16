# Corridor / Door Interaction Benchmark

本文档总结当前通道交互（door / corridor interaction）实现的能力、数据采集流程、输出格式、已知问题和论文写作时可以使用的结论。当前实现以 MolmoSpaces `nav_to_obj` benchmark episode 为父任务，在不改变原始导航目标语义的前提下，构造门状态变化导致的可达性 / 路径变化场景。

## 1. 任务定位

通道交互的核心不是训练或评测一个更强的开门控制器，而是把门这类通道对象建模为导航空间中的状态变量：

- 门全开状态表示原始 benchmark 的 nominal navigation condition。
- 门全关状态表示反事实通道受阻 condition。
- 如果全开可达、全关不可达或路径显著变化，并且全开最短路径穿过某些 interactive doors，则该 episode 被视为通道交互正例。
- 交互对象由 oracle 给出，执行层可以是 oracle open、已有 open/close policy 或后续交互图 planner。
- 当前 benchmark 更适合支撑“交互改变空间连通性 / 可达性”的故事，而不是“开某一扇门获得更短绕行路径”的故事。

当前使用的 critical door 定义保持简单：

```text
critical door = all-open GT path P_open 穿过的 interactive door root box
```

也就是说，critical door 只由 `P_open` 与门的空间 box 是否相交决定，不引入更复杂的拓扑因果分析。

## 2. 相关代码文件

当前通道交互主要涉及以下 Python 文件：

- `scripts/InteractiveNav/build_door_interaction_benchmark.py`
  - 当前完整 benchmark 构建入口。
  - 支持 critical door 预览与正式 build。
  - 负责读取原始 MolmoSpaces benchmark、采样导航目标、比较全开 / 全关路径、生成多种门状态样本、写出索引与 sample JSON。
- `scripts/InteractiveNav/explore_molmo_interactions.py`
  - 底层核心函数库。
  - 提供环境加载、门状态开关、Procthor occupancy map 构建、路径搜索、door/object box 采集、俯视图绘制等能力。
  - 关键函数包括 `open_all_doors`、`close_all_doors`、`set_door_root_state`、`build_live_procthor_map`、`compute_path_from_map`、`collect_interactive_door_root_object_records`、`save_door_path_figure`。
- `scripts/InteractiveNav/benchmark_door_state_scan.py`
  - 早期门状态扫描与路径变化分析工具。
  - 当前 builder 复用了其中的 `path_changed_strict`、`path_distance_stats`、`traversed_interactive_doors_on_path`。
- `scripts/InteractiveNav/benchmark_longest_nav_paths.py`
  - 原本用于扫描 benchmark 中较长导航路径。
  - 当前 builder 复用了 `set_episode_robot_pose`、`sample_nav_goal_for_episode`、`target_category` 等原始 benchmark episode 解析与目标采样逻辑。
- `scripts/InteractiveNav/read_scene_room_properties.py`
  - 场景 / 房间属性阅读辅助脚本，不是当前正式采集主链路。

## 3. 已实现能力

### 3.1 原始 benchmark 扫描

当前脚本可以直接扫描原始 MolmoSpaces `nav_to_obj` benchmark：

- 默认 benchmark 路径：
  `assets/benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/NavToObjProcthor10kBench_20260112_json_benchmark`
- 原始 episode 提供 house、robot pose、目标对象名 / candidates、成功阈值等信息。
- 原始 episode 不直接提供固定 `goal_xy`，因此当前实现会在目标对象附近采样一个 reachable navigation goal。
- 采样得到的 `nav_goal` 会写入每个 scan result，后续同一个 counterfactual door-state case 都使用同一个 goal。

### 3.2 全开 / 全关路径比较

对每个 episode，当前流程会：

1. 打开所有 interactive doors。
2. 设置 benchmark 中记录的 robot 初始位姿。
3. 在目标物体附近采样 `nav_goal`。
4. 基于全开 occupancy map 计算 `P_open`。
5. 关闭所有 interactive doors。
6. 重新计算到同一 `nav_goal` 的 `P_all_closed`。
7. 用严格路径变化规则判断是否存在交互影响。

严格路径变化由 `path_changed_strict` 判断，当前默认阈值为：

- `path_mean_distance_threshold_m = 0.35`
- `path_max_distance_threshold_m = 0.75`
- `path_length_delta_threshold_m = 0.5`

其中 `path_length_delta_threshold_m` 从较小阈值调到 `0.5`，主要是为了过滤由于目标点采样、栅格化或路径搜索细节造成的小尺度伪变化。

### 3.3 critical door 检测

当前实现会在 `P_open` 上采样稠密点，并判断这些点是否落入 interactive door root AABB 的膨胀区域内：

- `door_on_path_padding_m = 0.2`
- `path_region_sample_step_m = 0.05`

只有同时满足以下条件时，episode 才被视为可用于通道交互构建：

- `open_path` 存在。
- `all_closed_path_changed_strict = true`，即全关门后路径缺失或显著变化。
- `critical_door_names` 非空，即全开路径确实穿过至少一扇 interactive door。

这一步可以过滤类似“全关路径有一点变化，但 P_open 实际没有穿过门”的误报。

### 3.4 多种门状态样本构建

对每个正例 episode，当前会构建以下 case：

- `all_closed`
  - 关闭所有 interactive doors。
  - `required_open_doors = critical_door_names`。
  - 用于表达最强反事实：所有通道门都关闭后，原始导航任务是否仍可完成。
- `single_path_door_closed`
  - 只关闭一扇 critical door。
  - `required_open_doors = [door_name]`。
  - 如果该状态与 `all_closed` 重复，则跳过。
- `distractor_doors_closed`
  - 只关闭 non-critical interactive doors。
  - `required_open_doors = []`。
  - 用于构造不应触发交互的 distractor door state。
- `mixed_critical_and_distractor_closed`
  - 关闭一扇 critical door，同时关闭若干 non-critical doors。
  - `required_open_doors = [critical_door]`。
  - 用于构造有干扰门存在时仍需要识别关键交互门的场景。

当前门选择不再随机，而是按 door 到 robot 初始位置的距离排序：

- critical doors 近的优先。
- distractor doors 从 non-critical door pool 中取最近的 k 个。
- 多个 distractor / mixed sample 通过递增 k 生成，k 被限制在 `distractor_k_min` 到 `distractor_k_max` 之间。
- 每个 case 仍会写入 `sampling_seed` 与采样配置，便于追踪。
- 已实现 closed-door-state 去重，因此最终样本数可能小于参数请求数。

### 3.5 俯视图可视化

当前支持为 sample 保存 `path.png`：

- 绘制当前门状态下的 initial path。
- 叠加 all-open GT path 作为对照。
- 绘制 interactive / non-interactive doorway/object box。
- 高亮 `required_open_doors`。

配置：

- `--save_plots`：保存图片。
- `--plot_positive_only`：只保存正交互相关图，跳过 distractor-only 图。
- 不加 `--plot_positive_only` 时会保存所有 case 的图。

## 4. 数据采集流程

### 4.1 新建采集

新建采集使用：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLCONFIGDIR=/tmp/matplotlib-cache \
MLSPACES_CACHE_DIR=/tmp/molmo-spaces-cache-proxy \
MLSPACES_ASSETS_DIR=/tmp/molmo-spaces-assets-proxy \
python -u \
scripts/InteractiveNav/build_door_interaction_benchmark.py \
  --mode build \
  --input_mode original \
  --start_idx 0 \
  --max_episodes 2000 \
  --output_dir scripts/InteractiveNav/output/door_interaction_benchmark_v1_all2000_len05_d2m2_allplots \
  --resume \
  --door_on_path_padding_m 0.2 \
  --path_region_sample_step_m 0.05 \
  --path_mean_distance_threshold_m 0.35 \
  --path_max_distance_threshold_m 0.75 \
  --path_length_delta_threshold_m 0.5 \
  --num_distractor_samples_per_episode 2 \
  --num_mixed_samples_per_critical_door 2 \
  --distractor_k_min 1 \
  --distractor_k_max 5 \
  --sampling_seed 20260708 \
  --save_plots
```

这里没有设置 `--plot_positive_only`，因此会保存所有 sample 图片。

### 4.2 增量采集

当前代码提供 `--resume` 和 `--input_mode existing`：

- `--resume` 会优先读取已有 episode 的 `scan_result.json`，避免重复处理。
- `--input_mode existing` 会尝试读取已有 `scan_index.jsonl`，用于跳过已经判断过的 episode。
- 如果某个 episode 已经被判定为 `all_closed_path_unchanged_strict`，增量采集时可以直接跳过，无需重新进行仿真和路径计算。

需要注意：`benchmark.json` 与最终 `summary.json` 只在脚本正常结束后统一写出；运行过程中应主要查看 per-episode `scan_result.json`、`samples/` 和日志。

### 4.3 采集参数含义

- `num_distractor_samples_per_episode`
  - 每个正例 episode 构建多少个只关闭 non-critical doors 的 distractor case。
- `num_mixed_samples_per_critical_door`
  - 每个 critical door 构建多少个 mixed case。
- `distractor_k_min`
  - distractor case / mixed case 中最少关闭多少个 distractor doors。
- `distractor_k_max`
  - distractor case / mixed case 中最多关闭多少个 distractor doors。
- `sampling_seed`
  - 写入 sample 的稳定随机种子配置。当前门选择主要是 nearest-first，不再依赖随机抽样，但 seed 仍作为复现实验配置保存。

## 5. 输出数据格式

输出目录结构大致为：

```text
output_dir/
  episodes/
    ep_0003_house_101/
      scan_result.json
  samples/
    ep_0003_house_101_all_closed_all/
      sample.json
      path.png
    ep_0003_house_101_single_path_door_closed_<door>/
      sample.json
      path.png
  scan_index.csv
  scan_index.jsonl
  benchmark.json
  summary.json
  failures.json
```

### 5.1 `scan_result.json`

每个原始 episode 对应一个 `scan_result.json`，记录该 episode 是否存在通道交互：

- 原始 episode index、house index、target object / category。
- robot 初始位置与采样得到的 `nav_goal`。
- `open_path_found`、`open_path_length_m`。
- `all_closed_path_found`、`all_closed_path_length_m`。
- `all_closed_path_changed_strict` 与路径差异统计。
- `critical_door_names`、`noncritical_interactive_door_names`。
- `has_interaction` 与 `skip_reason`。
- 已构建 case 数量与 case id。
- 采样配置、阈值配置与耗时。

常见 `skip_reason`：

- `open_path_missing`
- `all_closed_path_unchanged_strict`
- `no_critical_door_on_open_path`

### 5.2 `sample.json`

每个构建出的交互样本对应一个 `sample.json`。它保留原始 MolmoSpaces episode，并额外增加：

```json
{
  "interactive_nav": {
    "schema_version": "door_interaction_nav_v1",
    "benchmark_type": "door_interaction_nav",
    "case_id": "...",
    "case_type": "...",
    "parent_benchmark_episode_index": 3,
    "door_state": {
      "closed_doors": ["..."],
      "open_doors": ["..."]
    },
    "oracle": {
      "required_open_doors": ["..."],
      "distractor_closed_doors": ["..."],
      "expected_static_path_found": false,
      "expected_after_oracle_path_found": true
    },
    "paths": {
      "all_open_path_length_m": 8.1,
      "initial_state_path_found": false,
      "oracle_restored_path_found": true
    },
    "diagnostics": {
      "critical_door_definition": "interactive door root boxes traversed by P_open"
    },
    "sampling": {...},
    "plot_path": "..."
  }
}
```

其中 `oracle.required_open_doors` 是当前 case 中为了恢复 all-open 可达性应打开的门，不表示模型预测结果。

### 5.3 `benchmark.json`

`benchmark.json` 是所有 `sample.json` 的列表，用于后续自定义 evaluator 或 policy runner 读取。

注意：这不是 MolmoSpaces 官方 schema 中已经内置的 benchmark 类型，而是在原始 episode 上扩展了 `interactive_nav` 字段。

## 6. 当前采集观察

截至当前全量采集运行的中间观察，D=2、M=2、保存所有图片时，数据呈现出以下趋势：

- 正例中绝大多数是 `all_closed` 后无路径，而不是仅路径长度轻微增加。
- `closed_path_missing` 占主要部分，说明当前通道交互数据更像 reachability / connectivity benchmark。
- 少数 episode 会出现 all-closed 仍有路径但路径显著变化。
- `critical_door_count` 以 1 扇门为主，少量 episode 有 2 到 4 扇 critical doors。
- 失败样本中有一类集中出现在 `trashcan_*` 目标对象，错误形态是 `No valid nav target candidates`。

这支持一个比较清晰的论文表述：

```text
In ProcTHOR-style indoor navigation scenes, door interaction rarely creates a
shorter optional detour under partial door states. Instead, the dominant effect
of door state changes is reachability: closing the doors on the nominal route
disconnects the robot from the object goal. Therefore, the first benchmark stage
focuses on interaction-aware reachability and critical passage identification.
```

中文表达可以写成：

```text
我们发现，在当前 MolmoSpaces / ProcTHOR 导航场景中，门交互主要不是表现为“打开某扇门获得更短路径”，而是表现为“关键通道状态改变导致目标区域可达性变化”。因此，本文第一阶段将通道交互建模为导航图上的连通性状态变量，并评测方法是否能够识别路径上的关键交互门、区分无关干扰门，以及在 oracle/open policy 恢复门状态后完成导航。
```

## 7. 已知问题与风险

### 7.1 原始 benchmark 没有固定 point goal

原始 `nav_to_obj` benchmark 是 object-goal navigation，不是固定 point-goal navigation。它提供目标对象及候选对象名，但不直接提供唯一 `goal_xy`。

当前实现会用 `NavGoalSampler` 在目标对象附近采样可达导航点，并将采样结果保存到 `scan_result.json` 和 sample 中。后续同一 episode 的不同门状态会复用该 `nav_goal`，但从方法论上仍应说明这是 object-goal 到 fixed navigation goal 的派生过程。

### 7.2 NavGoalSampler 的采样会造成表面上的路径差异

如果重复运行时重新采样目标点，路径可能不同。这不是路径搜索随机，而是目标点变了。

当前缓解方式：

- 对 episode index 设置稳定 seed。
- 同一个 episode 的所有 counterfactual case 复用同一个 `nav_goal`。
- 使用 `path_length_delta_threshold_m = 0.5` 过滤小尺度路径差异。
- 要求 positive case 必须存在 `critical_door_names`。

### 7.3 `trashcan_*` 目标仍存在候选匹配失败

当前全量采集中仍观察到部分 `trashcan_*` 目标报错：

```text
No valid nav target candidates for trashcan_...
```

这说明目标对象名 / alias / scene object candidate 匹配仍有边界问题。之前已加入 `trashcan_*` 与 `ashcan_*` 方向的匹配修复，但仍建议后续专门检查失败 episode 的 object metadata。

### 7.4 当前正例偏向可达性失败

目前数据中，门全关后大部分正例是路径不存在，而不是存在长绕行路径。这对论文故事是好事，但也限制了 benchmark 类型：

- 适合评测 interaction-aware reachability。
- 适合评测 critical passage detection。
- 适合评测干扰门下的 oracle interaction 选择。
- 不适合强行讲“交互代价与导航路径长度 trade-off”。

### 7.5 partial door state 的作用应谨慎表述

当前没有充分证据支持“关闭部分门后仍有长路径，打开某门可显著缩短路径”这种故事。

partial door state 更适合作为 benchmark 里的泛化 / 干扰配置：

- `single_path_door_closed`：只关路径上的关键门，测试是否能识别最小必要交互。
- `distractor_doors_closed`：只关无关门，测试是否不会误触发交互。
- `mixed_critical_and_distractor_closed`：关键门与干扰门同时关闭，测试是否能在复杂状态中选择正确交互对象。

### 7.6 door-on-path 判定依赖几何近似

critical door 判定使用 path 点与 door root AABB 的重叠关系：

- 优点是简单、可解释、容易人工检查。
- 风险是复杂门几何、门框、滑门、root/child body box 不一致时可能漏判或误判。
- 当前通过人工检查前 30 个 episode 的 preview 图验证过基本可用，但全量仍建议抽样复查。

### 7.7 门 root / child body 语义仍依赖 MolmoSpaces runtime

门的开关需要从 scene object、articulation root、hinge child body 等层级中归并。当前 `set_door_root_state` 与 `collect_interactive_door_root_object_records` 已处理 root grouping，但仍可能受 scene asset 结构影响。

### 7.8 运行耗时和存储开销较大

每个 episode 需要多次重建 occupancy map 与计算路径。D=2、M=2 且保存所有图片时，全量 2000 episode 需要较长时间，并会生成大量 PNG。

如果只是筛选正例，可以关闭图片或使用 `--plot_positive_only`；如果要人工审查，则保存全部图片更方便但体积更大。

### 7.9 最终聚合文件只在正常结束后写出

运行中如果中断：

- 已完成 episode 的 `scan_result.json` 与 sample 目录通常已经存在。
- `benchmark.json`、`summary.json`、`scan_index.csv/jsonl` 可能尚未更新或不完整。
- 继续运行时应使用 `--resume`。

## 8. 论文写作可用结论

当前通道交互 benchmark 可以支撑以下几个点：

- 门是导航图中影响连通性的状态变量，不应只作为静态语义标签。
- 在 object-goal navigation 中，目标本身可能仍然可见 / 可描述，但门状态会改变机器人是否能够到达目标区域。
- 只知道目标类别和静态地图不足以完成任务，agent 还需要识别路径上的交互对象及其状态。
- critical door 与 distractor door 的区分是交互导航中的关键能力。
- 当前 benchmark 的第一阶段应聚焦于 reachability 和 critical passage identification，而不是路径效率 trade-off。

可以考虑的实验分组：

- `StaticNav`
  - 不执行交互，只在当前门状态下导航。
- `AllOpenOracle`
  - 上界设置，所有门均打开或环境恢复到 all-open。
- `CriticalDoorOracle`
  - 只打开 `required_open_doors`，测试最小必要交互是否足够恢复可达性。
- `DistractorRobustness`
  - 在 distractor doors 关闭时，方法不应错误打开无关门。
- `MixedDoorRobustness`
  - 在 critical + distractor 同时存在时，方法应选择 critical door 而不是被干扰门吸引。

## 9. 后续建议

短期建议：

- 全量采集完成后，用最终 `summary.json` 替换本文档中的中间统计描述。
- 对 `trashcan_*` 失败样本做对象候选匹配修复或记录为排除规则。
- 随机抽查每类 case 的 `path.png`，尤其是 multi-critical-door episode。
- 为 `benchmark.json` 编写一个轻量 evaluator / loader，明确如何读取 `interactive_nav` 字段。

中期建议：

- 把 `interactive_nav` 扩展字段整理成正式 schema。
- 将通道交互与容器交互统一成 `interaction_state` / `oracle_interaction` / `diagnostics` 三层结构。
- 在论文中明确区分 corridor interaction 与 container interaction：
  - corridor interaction 改变可达性 / 连通性。
  - container interaction 改变目标可见性 / 可取性。
