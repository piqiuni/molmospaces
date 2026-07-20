# Interactive Navigation Dataset Definition v3

本目录定义统一的交互导航 benchmark JSON 格式，覆盖：

- 通道交互：开门后恢复空间可达性。
- 容器交互：打开容器后使目标满足 NavToObj 可见性条件。
- 混合交互：同一 episode 中依次执行通道和容器交互。
- Instruction 导航：保持 `language.task_description` 作为 policy-facing Instruction。

当前 door、container 和 mixed fine builder 均已输出 `interactive_nav_v3`。统一采集入口为
`scripts/InteractiveNav/collect_interactive_nav.py`，它负责从版本化 scene pool 生成 seed
episode、调用三类现有 builder、按有效 episode 配额冻结均衡数据，并执行统一结构审计。

source 支持两种模式：`kind: scene_split` 从 train/val scene pool 重新采样目标和
机器人起点，可配置目标类别、目标实例、起点—目标直线距离范围与远近偏好；
`kind: nav_benchmark` 直接保留原始 NavToObj benchmark 的目标和机器人起点。
channel 可直接使用 benchmark episode，container 和 mixed 仍必须先生成对应 rough
catalog，再执行 fine 交互验证。

可通过 `rough.container_catalog` / `rough.mixed_catalog` 复用已有 rough；设置
`rough.generate_if_missing: false` 时，缺失 rough 会立即报错，不会绕过 rough 阶段
直接生成 container/mixed fine 数据。

当前正式三类为 `channel`、`container`、`mixed`。统一生产配置不生成
`open_gt_control`，不主动构造错误动作 rollout；mixed 只接受真实
`mixed_required_verified` 门到容器因果链。

ProcTHOR-10K train 的 100 条均衡测试配置：

```bash
python scripts/InteractiveNav/collect_interactive_nav.py \
  --config scripts/InteractiveNav/configs/collection/procthor10k_train_100.yaml \
  --stage all
```

100 不能被 3 整除，因此默认目标为 `channel=34`、`container=33`、`mixed=33`。
示例配置按当前本机已安装资产使用 `train_0..train_99`；统一入口支持任意 train
house 范围，但扩大范围前必须先安装对应 ProcTHOR scene 资产。为兼顾稀有 mixed
链的容量与场景多样性，默认每屋上限为 channel=2、container=2、mixed=3。

2026-07-20 的 100 条轻量采集验证结果为：100/100 通过 V3 校验、三类
34/33/33、100 个唯一 case ID、39 个唯一 house、0 个占位值。三类分别覆盖
18/25/13 个 house；完整报告见配置输出目录下的
`balanced/structure_report.md`。

统一入口支持两种采集模式：

- `mode: light`：只保存 V3 metadata、oracle plan 和交互状态，适合大规模
  benchmark/评估数据。
- `mode: full`：调用指定 policy/executor，逐 step 保存相机图像、动作类型与向量、
  qpos/qvel、phase、reward、terminal/truncated 和 readback。输出为
  `interactive_nav_full_rollout_v1` H5；失败或未完成 rollout 只作为诊断文件，
  不计入训练有效样本。

full 模式目前对 channel、container、mixed 均有统一调度入口；默认
`policy.channel.executor=force`、`policy.container.executor=force`，直接复用现有
力控制器。executor 仍通过接口注册，可替换为已有 policy 或后续策略。生产配置仍
不生成 `open_gt_control`，也不构造错误动作负轨迹。

full rollout 的 segment 顺序按领域固定：channel 为
`initial → nav_to_door → force_open_door → nav_to_target → terminal_observation`，
container 为 `initial → nav_to_container → force_open_container →
terminal_observation`，mixed 则依次包含两组导航/force segment。每个 segment 的
每一步都同步写入 `steps/images/*`、`steps/actions/{type,vector,json}`、
`steps/states/{json,qpos,qvel}`、`phase`、`reward`、`terminal` 和 `info_json`。
force step 的 action type 为 `force_joint`，包含 effort、target value 和 joint
readback；terminal observation 使用 `observe` action type。相机帧按同一 step
索引写入 MP4，因而可以直接重建完整第一视角视频。

`full/summary.json` 只将成功且 segment、terminal、force step 均完整的 rollout 标记
为 `training_eligible=true`。导航碰撞、force 未达到目标或后续导航未完成的 rollout
仍保留完整诊断 H5/视频，但不会混入模型训练集。

## 示例真实性

`examples/` 下四个 episode 是为说明和校验 v3 字段而手工构造的 synthetic examples，不是由现有 benchmark builder 或 MuJoCo 仿真生成。示例中的 house、对象 ID、joint、pose、路径长度、可见像素和验证结果仅用于展示结构，未经过真实门/容器几何、碰撞、路径或 head-camera 可见性判断，不应作为 benchmark 样本或实验结果使用。

真实历史 episode 已原样归档在相邻的 `v1/benchmark.json` 和 `v2/benchmark.json` 中。

## 文件

```text
scripts/InteractiveNav/dataset_definition/v3/
  README.md
  interactive_nav_episode.schema.json
  validate_examples.py
  migration_v1_door_to_v3.md
  migration_v2_container_to_v3.md
  TODO.md
  examples/
    channel_episode.json
    container_episode.json
    mixed_episode.json
    no_interaction_episode.json
```

`interactive_nav_episode.schema.json` 是 JSON Schema Draft 2020-12 定义。示例只保留说明 v3 所需的关键 EpisodeSpec 字段，允许 MolmoSpaces EpisodeSpec 的其他相机、机器人和 provenance 字段继续存在。

在 `mlspaces` 环境中校验 Schema、示例和跨字段引用：

```bash
python scripts/InteractiveNav/dataset_definition/v3/validate_examples.py
```

除 JSON Schema 外，该命令还检查 selection mode 和目标实例一致、距离阈值一致、interaction requirement、articulation replay、typed prerequisite 无环、plan-level required interaction IDs、`oracle_plans[0]` 等于 canonical `oracle_plan`，以及 oracle step 引用的 object/joint 与 interaction 定义一致。

## 设计原则

### Policy 可见字段

policy 可以接收：

- `language.task_description`
- 相机观测
- 机器人 proprioception
- evaluator 明确允许的在线感知字段

`language.task_description` 继续由 `JsonEvalTaskSampler` 绑定到 `task.get_task_description()`，因此兼容当前 learned policy 和 Instruction 输入接口。

### Privileged GT 字段

以下字段属于 benchmark GT，不应直接进入 policy observation：

- `task.pickup_obj_name`
- `task.pickup_obj_candidates`
- `interactive_nav.target.selected_instance`
- `interactive_nav.interactions`
- `interactive_nav.oracle_plan`
- `interactive_nav.oracle_plans`
- `interactive_nav.generation_validation`
- `scene_modifications.articulation_states`

开源 benchmark 可以包含这些 GT。数据文件公开与 policy 是否获得 GT 是两个独立问题；evaluator/task sampler 需要 GT 才能恢复场景和判断成功。

## 顶层结构

```json
{
  "house_index": 1,
  "scene_dataset": "procthor-10k",
  "data_split": "val",
  "robot": {},
  "cameras": [],
  "scene_modifications": {},
  "task": {},
  "language": {},
  "interactive_nav": {}
}
```

其中 `robot`、`cameras` 和其他标准字段继续遵循 MolmoSpaces `EpisodeSpec`。v3 主要固定 `task`、`language`、`scene_modifications.articulation_states` 和 `interactive_nav` 的交互导航扩展。

## Instruction

`language.task_description` 是实际提供给 policy 的 Instruction。

固定 Instruction 类型：

```text
object_goal
route_instruction
interaction_instruction
route_interaction_instruction
```

固定交互披露级别：

```text
hidden
partial
explicit
```

示例：

```json
{
  "language": {
    "task_description": "Find the apple.",
    "instruction_type": "object_goal",
    "locale": "en",
    "interaction_disclosure": "hidden",
    "referral_expressions": {
      "object_name": "apple"
    },
    "referral_expressions_priority": {}
  }
}
```

v3 只定义 grounding 字段，不强制 `specific_instance` 的语言描述在场景中唯一。目标唯一性可以在后续数据生成规则或 quality gate 中增加，不需要再次改变 JSON 主结构。

## 目标选择模式

固定 selection mode：

```text
specific_instance
any_candidate
```

`task.selection_mode` 是运行时 evaluator 的权威配置。

### specific_instance

```json
{
  "selection_mode": "specific_instance",
  "pickup_obj_name": "apple_selected_instance",
  "pickup_obj_candidates": ["apple_selected_instance"]
}
```

只有指定实例满足距离和 head-camera 可见性时成功，其他同类物体不成功。当前 container、channel 和 mixed v3 示例均采用该模式。

### any_candidate

```json
{
  "selection_mode": "any_candidate",
  "pickup_obj_name": "apple_0",
  "pickup_obj_candidates": ["apple_0", "apple_1", "apple_2"]
}
```

任意 Instruction-consistent candidate 满足 NavToObj 条件即可成功。`pickup_obj_name` 在该模式下是兼容当前 task config 的代表实例，不将其解释为唯一成功目标。

## NavToObj 成功条件

v3 保持当前 `NavToObjTask` 的成功语义：

```text
planar_distance(robot_base, selected_target) < succ_pos_threshold
AND
head_camera visibility_fraction(selected_target) > 0
```

`task.succ_pos_threshold` 是运行时权威阈值。`interactive_nav.success_criteria` 是结构化镜像，用于生成验证和跨 evaluator 解释，不替代 `NavToObjTask.judge_success()`。

对于 `specific_instance`，selected target 是 `pickup_obj_candidates` 中唯一实例。对于 `any_candidate`，只要任一候选满足条件即可。

抽屉中的 `joint_object_consistent_motion` 只用于证明目标属于 moving compartment，不能替代最终 head-camera 可见性成功条件。

## Interaction Requirement

每个 episode 明确标记交互必要性：

```text
required
beneficial
unnecessary
unknown
```

- `required`：至少存在一个必要 interaction；不执行有效交互计划时不能完成任务。
- `beneficial`：不执行该 interaction 仍可完成任务，但实测替代路径更长；数据必须保存开门前后路径长度、差值、比例和通过的阈值。该值用于少量路径效率对比，不能伪装成 `required`。
- `unnecessary`：初始状态下无需任何 interaction 即可完成任务，`interactions=[]`，oracle 只包含导航、观察等非交互步骤。
- `unknown`：格式已经生成，但尚未完成交互必要性验证。

`interaction_domains` 表示该 episode 所考察的交互域，不表示 oracle 一定包含 interaction。因此 interaction-unnecessary door 对照样本仍可以使用：

```json
{
  "interaction_domains": ["channel"],
  "interaction_requirement": "unnecessary",
  "interactions": []
}
```

## Interaction 类型

v3 首版只允许当前主线需要的固定列表：

```text
channel_hinged_door
channel_sliding_door
container_hinged_door
container_sliding_drawer
```

暂不把灯、开关、窗帘和可移动障碍物写入 v3。新增 interaction type 时应更新 schema version 或兼容版本，而不是在数据中使用自由字符串。

固定 interaction effect type：

```text
restore_reachability
reduce_navigation_cost
enable_interaction
reveal_target_object
```

`restore_reachability` 表示关门时无路径；`reduce_navigation_cost` 表示关门时仍有路径，但开门后通过实测阈值验证路径代价下降。一个 interaction 也可以同时产生多个 effect。

固定 prerequisite type：

```text
mechanical
reachability
visibility
```

- `mechanical`：例如必须先打开冰箱外门，才能拉出内部抽屉。
- `reachability`：例如必须先打开通道门，机器人才能到达冰箱。
- `visibility`：前置 interaction 只负责建立后续观察或揭示条件。

对于 `mixed + beneficial`，通道门不是容器交互的 `reachability` prerequisite；它是 oracle 中的代价优化动作。容器交互仍可保持 `required`，因为目标在容器关闭时仍不可见。

若只有“联合关闭多扇门”才产生捷径收益，rough catalog 使用 `mixed_door_set_shortcut_verified` 单独保留；不要降格成单门 `mixed_shortcut_verified`。当前精细生成器只物化已逐门验证的单门 beneficial 样本，多门 beneficial oracle 需要后续显式生成多个 channel interaction。

每个 interaction 保存：

- 稳定的 `interaction_id`
- 固定 `type`
- 场景对象和 joint name/index
- 初始和目标 articulation state
- 一个或多个 `effect_types`
- typed `prerequisites`

`joint_name` 是回放主键，`joint_index` 用于调试和一致性检查。

## GT Oracle 动作

固定 oracle step type：

```text
navigate
set_view
open_joint
observe_target
```

这些是 GT 高层动作，不是低层 torque/action trajectory。`open_joint.interaction_id` 必须引用 `interactions` 中的定义。

固定 reason code：

```text
approach_channel_interaction
restore_reachability
traverse_open_channel
approach_container_interaction
improve_target_visibility
prerequisite_for_interaction
reveal_target_object
satisfy_nav_to_obj_success
verify_target_visible
```

一个 mixed oracle 可以包含多个 `navigate` 和多个 `open_joint`：

```text
navigate to door
open door
navigate through channel to container
set view
open outer container door
open inner drawer
observe target
```

每个 oracle plan 保存：

- 稳定的 `plan_id`。
- 该计划实际打开的 `required_interaction_ids`，按首次执行顺序排列。
- typed `steps`。

`oracle_plan` 保存 canonical GT plan；`oracle_plans` 保存所有独立有效 GT plan，并要求 `oracle_plans[0]` 与 `oracle_plan` 一致。interaction-unnecessary plan 的 `required_interaction_ids=[]`，但仍至少包含 `navigate` 和 `observe_target` 等步骤。

## Initial State

权威 articulation 初始值保存在：

```text
scene_modifications.articulation_states
```

`interactive_nav.initial_state` 保存面向分析的状态标签和参与本任务的 interaction state。回放时仍以 joint name 对应的 `scene_modifications.articulation_states` 为准。

## Generation Validation

`generation_validation` 只用于数据生成、质量审计和 oracle 调试，不是导航输入。

主要字段：

- `navigation_validation`：初始路径、交互后路径、GT path 长度和目标 pose。
- `interaction_validations`：每个 interaction 的执行、碰撞、joint 和状态转移证据。
- `oracle_prefixes`：每个 GT plan prefix 后的可达性、距离、可见性和 task success 状态。
- `compartment_evidence`：slide joint 与物体一致运动证据。
- `success_evidence`：数据生成时对 NavToObj 距离加可见性条件的实测结果。
- `minimal_plan_verified`：删除任一必要 interaction 后任务是否重新不可完成。

`success_evidence.status` 固定为：

```text
passed
failed
not_executed
```

如果只验证 GT path 而没有执行到终态，必须使用 `not_executed`，不能写成 task success。

每个 oracle prefix 至少保存：

```text
completed_step_count
robot_reachable_to_next_goal
target_distance_passed
target_visibility_fraction
task_success
```

建议同时保存 `plan_id`、`target_visible_pixels` 和 `opened_interaction_ids`。这些字段用于验证 mixed 计划中“通道恢复可达性、容器 interaction 揭示目标、最终同时满足距离和可见性”的逐步因果链。

## Evaluation Metrics

v3 的 `generation_validation` 用于数据生成质量审计，不等同于 policy 运行结果。正式评测应由独立 scorer 读取 benchmark GT 和 policy rollout 输出后计算。

主评测指标固定为：

| 指标 | 计算对象 | 说明 |
|------|----------|------|
| `SR` | episode | 是否满足 `NavToObj` 的最终距离和 head-camera 可见性成功条件 |
| `SPL` | episode | 成功加权路径效率，参考路径取允许必要交互后的可行计划路径 |
| `Interaction Success Rate` | required episode | 关键交互效果是否完成 |
| `Shortcut Benefit / Regret` | beneficial episode | 相对开门 oracle 的路径代价收益或策略额外代价 |
| `Interaction Precision` | interaction event | 执行过的交互中，有多少是有效交互 |
| `Total Cost` | episode | `path_length + λ * interaction_count`，后续可扩展不同交互类型权重 |

`reachability`、`visibility` 和 `enablement` 不作为主表中的独立指标；它们是 `Interaction Success Rate` 的判定语义：

- `channel`：关键通道交互是否恢复目标区域可达性。
- `container`：关键容器交互是否让目标满足可见性条件。
- `mixed-required`：必要交互链是否按 prerequisite / oracle 语义完成，并最终服务目标可达和可见。
- `mixed-beneficial`：目标仍可通过绕行完成；重点报告是否利用捷径、SPL、Total Cost 与相对 oracle regret。
- `no-interaction`：`Interaction Success Rate` 记为 `N/A`，但任何多余交互都会影响 `Interaction Precision` 和 `Total Cost`。

推荐报告 split：

```text
all
channel
container
mixed
no-interaction
```

主表应优先报告这些 split 的 5 个主指标；更细的 reachability gain、visibility gain、prerequisite violation、distractor interaction 等只作为 debug / appendix 诊断项。

## Planned Evaluation Scripts

建议脚本分为四类，避免把数据集质量和方法性能混在一起：

1. `build_mixed_interaction_benchmark.py`
   - 合并或生成 `channel`、`container`、`mixed`、`no-interaction` v3 episode。
   - 输出统一 `benchmark.json`，其中 `oracle_plan` 和 `interactions` 只作为 privileged GT。

2. `evaluate_interactive_nav_dataset.py`
   - 只检查 benchmark 数据质量。
   - 检查 schema、占位字段、split 分布、`interaction_requirement`、`success_evidence`、oracle/prerequisite 一致性。

3. `score_interactive_nav_run.py`
   - 读取 v3 benchmark 和 policy rollout。
   - 输出 5 个主指标及 per-split 结果。
   - 需要从 rollout 中恢复实际 path length、成功状态、交互事件、交互成功/失败和多余交互。

4. `summarize_interactive_nav_results.py`
   - 汇总多个方法的 scorer 输出。
   - 生成论文表格用 CSV / Markdown。

## 门必要性

v3 当前门样本先使用 `specific_instance`，不在首版生成规则中处理门的多目标 object-goal 必要性。

对指定目标，通道 interaction 必要性定义为：

```text
初始 articulation state 下，不存在满足 NavToObj 距离加可见性条件的可达终态；
执行最小 interaction plan 后，存在这样的终态。
```

`any_candidate` 的候选集合级必要性字段已经由 schema 支持，但具体生成算法和 quality gate 后续再定义。

## Mixed 表达

当前 v3 可在一个 episode 中统一表达 channel 和 container interaction：

```text
open channel door
-> navigate through channel
-> open outer container door
-> open inner drawer
-> set view
-> observe target
```

跨域关系由 typed prerequisites、oracle step 顺序、plan-level required interaction IDs 和 oracle prefix validation 共同表达。双开门和多个 joint 组成一个原子联合动作的 grouping 语义尚未进入当前 Schema，记录在 `TODO.md`。

## 兼容与迁移

- v1 door 数据需要将 `required_open_doors` 迁移为 typed `interactions` 和 typed `oracle_plan.steps`。
- v2 container 数据需要增加 `task.selection_mode`、`language.instruction_type`、统一 `interactions` 和结构化 `success_criteria`。
- v3 benchmark 可以在同一 JSON episode 列表中包含 channel、container 和 mixed 样本，因为三类 episode 使用相同 schema。
- evaluator 必须从 policy observation 中移除所有 privileged GT 字段。
