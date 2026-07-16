# Container Interaction v2 到 Interactive Navigation v3 迁移指南

本文用于指导修改旧容器采集代码，使 `interactive_nav_v2` builder 直接生成统一的 `interactive_nav_v3` episode。

适用代码与数据：

```text
scripts/InteractiveNav/build_container_interaction_benchmark.py
scripts/InteractiveNav/collect_container_fine_parallel.py
scripts/InteractiveNav/collect_container_interaction_benchmark_serial.py
molmo_spaces/evaluation/benchmark_schema.py
scripts/InteractiveNav/dataset_definition/v2/benchmark.json
```

目标定义：

```text
scripts/InteractiveNav/dataset_definition/v3/interactive_nav_episode.schema.json
scripts/InteractiveNav/dataset_definition/v3/examples/container_episode.json
scripts/InteractiveNav/dataset_definition/v3/examples/mixed_episode.json
```

## 1. 迁移范围

container v2 已经具备大部分 v3 所需信息：

- 严格目标物体和 GT 容器。
- controlling joint 和 joint sequence。
- 真实 `scene_modifications.articulation_states`。
- `navigate`、`set_view`、`open_joint`、`observe_target` 高层步骤。
- multi-oracle。
- 路径、可见性、物体绑定和交互 pose 验证。

主要迁移工作是统一字段语义，而不是重新设计容器采集流程：

- 补充 task 和 language 固定字段。
- 重构 target。
- 把 joint sequence 显式转换为 `interactions`。
- 给 oracle step 增加 `interaction_id` 并统一 reason code。
- 将 visibility threshold 对齐 NavToObj 的 `> 0`。
- 将 generation validation 分成固定栏目。
- 增加结构化 success criteria 和 NavToObj 终态证据。

## 2. 顶层 Episode 保留与补全

继续保留 v2 已生成的：

```text
source
house_index
scene_dataset
data_split
seed
robot
img_resolution
cameras
scene_modifications
task_relevant_objects
```

`scene_modifications.articulation_states` 已经是正确的回放主线，不要移除，也不要只保留 `interactive_nav.initial_state.articulation_states`。

## 3. Task 迁移

当前 v2 已绑定单一目标：

```text
pickup_obj_name = object_record.name
pickup_obj_candidates = [object_record.name]
```

补充：

```python
episode["task"]["selection_mode"] = "specific_instance"
```

并检查：

```python
assert episode["task"]["pickup_obj_candidates"] == [
    episode["task"]["pickup_obj_name"]
]
```

当前阶段不改为 `any_candidate`，也不扩展同类目标集合。

## 4. Language 迁移

v2 当前格式：

```json
{
  "task_description": "find the apple.",
  "referral_expressions": {"object_name": "apple"},
  "referral_expressions_priority": {}
}
```

补充固定字段：

```python
episode["language"].update(
    {
        "instruction_type": "object_goal",
        "locale": "en",
        "interaction_disclosure": "hidden",
    }
)
```

容器、joint 和 oracle 不进入 policy-facing Instruction。目标描述是否唯一留给后续 quality gate。

## 5. Target 字段重构

v2 到 v3 的主要映射：

| v2 | v3 |
|---|---|
| `target.object_name` | `target.selected_instance` |
| `target.object_category` | `target.category` |
| `target.container_name` | `target.container_name` |
| `target.container_category` | `target.container_category` |
| 无 | `target.selection_mode=specific_instance` |
| 无 | `target.instruction_consistent_candidates=[object_name]` |
| 无 | `target.grounding` |
| AABB 字段 | 保留为 target 附加字段 |
| controlling joint 字段 | 移到 `interactions`，可保留 legacy 副本作诊断 |

推荐生成：

```python
target = {
    "selection_mode": "specific_instance",
    "category": old_target["object_category"],
    "selected_instance": old_target["object_name"],
    "instruction_consistent_candidates": [old_target["object_name"]],
    "container_name": old_target["container_name"],
    "container_category": old_target["container_category"],
    "grounding": {
        "unique": None,
        "matching_instance_count": None,
        "description": episode["language"]["referral_expressions"].get(
            "object_name"
        ),
        "attributes": {},
    },
    "object_aabb_center": old_target.get("object_aabb_center"),
    "object_aabb_size": old_target.get("object_aabb_size"),
    "container_aabb_center": old_target.get("container_aabb_center"),
    "container_aabb_size": old_target.get("container_aabb_size"),
}
```

## 6. Success Criteria

所有 v3 episode 明确镜像 NavToObj 成功条件：

```python
success_criteria = {
    "type": "nav_to_obj",
    "target_selection": "specific_instance",
    "distance": {
        "metric": "planar_robot_base_to_object",
        "threshold_m": float(episode["task"]["succ_pos_threshold"]),
        "comparison": "strictly_less",
    },
    "visibility": {
        "camera_name": "head_camera",
        "metric": "visibility_fraction",
        "threshold": 0.0,
        "comparison": "strictly_greater",
    },
    "combination": "all",
}
```

注意：v2 builder 的默认 `visibility_threshold` 曾使用 `1e-4`。v3 按当前 NavToObj 判据使用严格 `visibility_fraction > 0`，`observe_target.visibility_threshold` 必须写 `0.0`。

数据生成时可以额外记录更严格的质量阈值，但它不能替代 benchmark 成功条件。例如可保存：

```json
{
  "generation_quality_visibility_threshold": 0.0001
}
```

## 7. 从 Joint Sequence 构建 Interactions

v2 的每个 oracle candidate 已保存：

```text
controlling_joint_index
controlling_joint_name
joint_sequence
joint_type
```

对所有 `oracle_plans` 中出现的 `(container_name, joint_name)` 去重，生成统一 interaction 表。

### 7.1 Interaction ID

推荐稳定格式：

```python
interaction_id = f"container::{container_slug}::{joint_index}"
```

同一 joint 在多个 oracle plan 中必须使用同一个 interaction ID。

### 7.2 Interaction type

根据真实 joint type 映射：

```text
hinge -> container_hinged_door
slide -> container_sliding_drawer
```

不要只根据容器类别判断。例如 Fridge 同时可能包含 hinge 外门和 slide 内部抽屉。

### 7.3 Interaction effects

根据 joint 在计划中的作用映射：

```text
joint_sequence 最后一个 joint -> effect_types=[reveal_target_object]
前置 joint -> effect_types=[enable_interaction]
```

如果同一个 joint 在不同 oracle plan 中产生多个作用，`effect_types` 保存这些 effect 的去重并集。例如同一外门既能使目标部分可见，又能使内部抽屉可操作：

```json
{
  "effect_types": ["enable_interaction", "reveal_target_object"]
}
```

plan 中该次动作的具体作用仍由 `open_joint.reason` 表达，不需要为同一 joint 创建两个 interaction。

### 7.4 Prerequisite IDs

对于 joint sequence：

```text
[joint_3, joint_1]
```

生成 typed prerequisites：

```text
joint_3.prerequisites = []
joint_1.prerequisites = [
  {interaction_id: joint_3.interaction_id, type: mechanical}
]
```

容器内部外门与抽屉通常使用 `mechanical`。若后续 mixed builder 证明必须先打开通道门才能到达容器，则容器 interaction 可以增加 type 为 `reachability` 的跨域 prerequisite。若 sequence 更长，每个 interaction 保存打开它之前所需的全部必要 interaction，顺序仍由 oracle steps 表达。

### 7.5 Initial and target state

从 `scene_modifications.articulation_states` 按 `joint_name` 查询真实初始 fraction：

```python
{
    "initial_state": {
        "joint_fraction": initial_fraction,
        "semantic_state": "closed" if initial_fraction == 0.0 else "open",
    },
    "target_state": {
        "joint_fraction": 1.0,
        "semantic_state": "open",
    },
}
```

容器 benchmark 正常应为 closed `0.0`。若实际 articulation state 不为 0，不要强制篡改 interaction 镜像；应先修复采集初始状态。

### 7.6 完整示例

```python
interaction = {
    "interaction_id": interaction_id,
    "type": "container_sliding_drawer",
    "object_name": container_name,
    "object_category": container_category,
    "joint_name": joint_name,
    "joint_index": joint_index,
    "effect_types": ["reveal_target_object"],
    "prerequisites": [
        {
            "interaction_id": outer_door_interaction_id,
            "type": "mechanical",
        }
    ],
    "initial_state": {
        "joint_fraction": 0.0,
        "semantic_state": "closed",
    },
    "target_state": {
        "joint_fraction": 1.0,
        "semantic_state": "open",
    },
}
```

## 8. Oracle Plan 迁移

v2 step 类型已经与 v3 基本一致，但需要补充引用和修改固定 reason。

### 8.1 Reason 映射

| v2 reason | v3 reason |
|---|---|
| `approach_target_container` | `approach_container_interaction` |
| `improve_target_visibility` | 不变 |
| `prerequisite_for_target_compartment` | `prerequisite_for_interaction` |
| `reveal_target_object` | 不变 |
| `verify_target_visible` | 不变 |

### 8.2 Navigate

保留 v2 的 goal point、yaw 和容差，增加计划中第一个 interaction 的 ID：

```python
{
    **old_navigate_step,
    "interaction_id": first_interaction_id,
    "reason": "approach_container_interaction",
}
```

### 8.3 Set view

以下字段可直接保留：

```text
view_profile
head_qpos
torso_qpos
reason=improve_target_visibility
```

head/torso qpos 必须来自初始化后读取的基准姿态加配置偏移，不能使用与实际机器人不一致的硬编码默认初始值。

### 8.4 Open joint

增加 `interaction_id` 并重写 prerequisite reason：

```python
new_step = {
    **old_step,
    "interaction_id": interaction_id_by_joint_name[old_step["joint_name"]],
    "reason": (
        "prerequisite_for_interaction"
        if old_step["reason"] == "prerequisite_for_target_compartment"
        else old_step["reason"]
    ),
}
```

保留 v2 控制方式：

```text
hinge -> direct
controlling slide -> force
```

内部 slide drawer 的 prerequisite hinge 仍使用 direct；不要因为最终目标在抽屉中而把所有 joint 都改成 force。

### 8.5 Observe target

v3 需要增加 camera name，并把阈值改为 `0.0`：

```python
{
    "type": "observe_target",
    "object_name": target_object_name,
    "camera_name": "head_camera",
    "visibility_threshold": 0.0,
    "reason": "verify_target_visible",
}
```

### 8.6 Multi-oracle

继续保留：

```python
for plan_index, plan in enumerate(migrated_oracle_plans):
    plan["plan_id"] = f"oracle_{plan_index}"
    plan["required_interaction_ids"] = ordered_unique(
        step["interaction_id"]
        for step in plan["steps"]
        if step["type"] == "open_joint"
    )
oracle_plan = migrated_oracle_plans[0]
oracle_plans = migrated_oracle_plans
```

并确保：

```python
assert oracle_plans[0] == oracle_plan
```

多个 plan 可以引用同一个 interactions 表。`interactions` 是 episode 内所有有效 oracle plan 的 joint 并集。每个 plan 的 `required_interaction_ids` 只保存该 plan 实际执行的 joint，并按首次执行顺序排列。

## 9. Initial State 迁移

v2 的：

```text
all_doors_open
container_joints_closed
articulation_states
target_visible
```

可以作为附加诊断保留，但 v3 必须增加：

```python
initial_state = {
    "interaction_states": [
        {
            "interaction_id": interaction["interaction_id"],
            "joint_fraction": interaction["initial_state"]["joint_fraction"],
            "semantic_state": interaction["initial_state"]["semantic_state"],
        }
        for interaction in interactions
    ],
    "all_doors_open": old_initial_state["all_doors_open"],
    "container_joints_closed": old_initial_state["container_joints_closed"],
    "target_visible": old_initial_state["target_visible"],
}
```

不要在 `interactive_nav.initial_state` 中重复保存完整 articulation state 作为第二权威源。可为 legacy reader 暂时保留，但新 reader 必须只使用 `scene_modifications.articulation_states` 回放。

## 10. Generation Validation 重构

推荐映射：

| v2 | v3 |
|---|---|
| `start_validation`、navigate goal/pose | `navigation_validation` |
| `oracle_validations` | `interaction_validations` 或 plan validation 附加字段 |
| selected `visibility_trace` | `oracle_prefixes` |
| `object_binding` | `compartment_evidence` |
| final visibility + 新增距离 | `success_evidence` |
| 依赖/controlling joint 最小性验证 | `minimal_plan_verified` |

推荐结构：

```python
generation_validation = {
    "navigation_validation": {
        "start_validation": old_validation["start_validation"],
        "navigate_goal_point": old_validation["navigate_goal_point"],
        "navigate_goal_yaw": old_validation["navigate_goal_yaw"],
        "interaction_pose": old_validation["interaction_pose"],
        "interaction_pose_collision_free": old_validation[
            "interaction_pose_collision_free"
        ],
    },
    "interaction_validations": old_validation["oracle_validations"],
    "oracle_prefixes": old_validation["visibility_trace"],
    "compartment_evidence": (
        old_validation["object_binding"]
        if old_validation["object_binding"].get("applicable")
        else None
    ),
    "success_evidence": success_evidence,
    "minimal_plan_verified": minimal_plan_verified,
    "reveal_mode": old_validation["reveal_mode"],
    "view_profile": old_validation["view_profile"],
    "joint_assignment_ambiguous": old_validation[
        "joint_assignment_ambiguous"
    ],
}
```

不能直接把旧 visibility trace 原样写入 `oracle_prefixes`。每个 prefix 至少归一化为：

```python
{
    "plan_id": plan_id,
    "completed_step_count": completed_step_count,
    "robot_reachable_to_next_goal": reachable,
    "target_distance_passed": distance_passed,
    "target_visibility_fraction": visibility_fraction,
    "target_visible_pixels": visible_pixels,
    "task_success": bool(distance_passed and visibility_fraction > 0.0),
    "opened_interaction_ids": opened_interaction_ids,
}
```

### 10.1 Success evidence

v2 已测得 interaction pose 下的最终可见性，但还要明确计算机器人与指定目标实例的平面距离：

```python
planar_distance_m = np.linalg.norm(
    robot_base_position[:2] - target_object_position[:2]
)
distance_passed = planar_distance_m < succ_pos_threshold
visibility_passed = final_visibility_fraction > 0.0
```

如果这两个值都在同一终态实测：

```python
success_evidence = {
    "status": "passed" if distance_passed and visibility_passed else "failed",
    "validation_mode": "simulated_terminal_state",
    "target_object_name": target_object_name,
    "planar_distance_m": planar_distance_m,
    "distance_threshold_m": succ_pos_threshold,
    "camera_name": "head_camera",
    "visibility_fraction": final_visibility_fraction,
    "visible_pixels": final_visible_pixels,
    "distance_passed": distance_passed,
    "visibility_passed": visibility_passed,
    "expected_task_success": distance_passed and visibility_passed,
}
```

若只迁移旧 JSON、没有可靠终态机器人位置或距离，则写：

```text
status=not_executed
validation_mode=path_feasibility_only
```

不要仅凭 `final_visible_pixels > 0` 写成完整 NavToObj success。

### 10.2 Slide motion evidence

`slide_object_motion` 和 object binding 只证明物体属于 moving compartment，不替代最终 head-camera 可见性。

v3 中：

- 一致运动证据写入 `compartment_evidence`。
- `success_evidence.visibility_passed` 仍必须由 `visibility_fraction > 0` 决定。
- 只有运动、没有可见像素时，不能按当前 NavToObj 成功条件生成有效终态样本。

### 10.3 Minimal plan

只有满足以下条件才写 `minimal_plan_verified=true`：

- 初始目标不可见。
- 仅执行 prerequisites 后仍不可见。
- 执行 controlling joint 后首次可见。
- 删除任一声明为必要的 prerequisite 后，后续 controlling joint 不能形成有效计划。
- 没有循环依赖。

旧 visibility prefix 只能证明“当前顺序中首次可见”，不能总是证明所有 prerequisite 的因果必要性。没有完整验证时写 `null`。

## 11. 统一 Payload

容器 episode 最终结构：

```python
episode["interactive_nav"] = {
    "schema_version": "interactive_nav_v3",
    "case_id": case_id,
    "parent_benchmark_episode_index": source_episode_index,
    "interaction_domains": ["container"],
    "interaction_requirement": "required",
    "target": target,
    "success_criteria": success_criteria,
    "initial_state": initial_state,
    "interactions": interactions,
    "oracle_plan": oracle_plans[0],
    "oracle_plans": oracle_plans,
    "generation_validation": generation_validation,
}
```

不再写单数 `interaction_domain: container` 作为主字段。为短期 legacy reader 兼容可以保留附加字段，但新代码只读取 `interaction_domains`。

## 12. 推荐代码修改顺序

### 第一步：增加共享 v3 helper

建议新增可被 door/container builder 共用的模块，例如：

```text
scripts/InteractiveNav/interactive_nav_v3.py
```

至少包含：

```text
normalize_specific_instance_task
build_language_spec
build_target_spec
build_success_criteria
build_interaction_id
build_initial_state
validate_interaction_references
validate_v3_episode
```

### 第二步：修改 `build_oracle_plan()`

函数额外接收：

```text
interaction_id_by_joint_name
visibility_threshold=0.0
```

输出只使用 v3 固定 reason code，并给 navigate/open_joint 添加 interaction ID。

### 第三步：修改 `generated_episode()`

在该函数中完成：

- task selection mode。
- language 固定字段。
- target 重构。
- interactions 构建。
- initial state 重构。
- oracle plan 迁移。
- generation validation 重构。
- schema version 更新。

### 第四步：更新 Pydantic Schema

`molmo_spaces/evaluation/benchmark_schema.py` 当前 `InteractiveNavSpec` 固定为 `interactive_nav_v2`。需要新增 v3 model，或用 `schema_version` 做 discriminated union。

迁移期间推荐同时支持：

```text
door_interaction_nav_v1 只读
interactive_nav_v2 只读
interactive_nav_v3 读写
```

不要直接把旧 `InteractiveNavSpec` 的 literal 改成 v3 后导致历史数据无法加载。

### 第五步：更新并行/串行合并器

`collect_container_fine_parallel.py` 和串行入口通常只合并 episode，不应重新解释字段。需要检查：

- 是否按旧 `interaction_domain` 过滤。
- 是否读取旧 target key `object_name`。
- 是否读取旧 generation validation 扁平字段。
- summary/evaluator 是否接受 `interaction_domains` 和新结构。

## 13. Builder 级伪代码

```python
episode = copy.deepcopy(template_episode)
restore_source_start_and_target(episode, source_episode, object_record)
normalize_specific_instance_task(episode)
normalize_language(episode)

interaction_specs = build_container_interactions(
    container=container,
    oracle_candidates=oracle_candidates,
    articulation_states=articulation_states,
)
interaction_id_by_joint_name = {
    item["joint_name"]: item["interaction_id"]
    for item in interaction_specs
}
oracle_plans = [
    migrate_or_build_v3_oracle_plan(
        container,
        candidate,
        object_record["name"],
        interaction_id_by_joint_name,
        visibility_threshold=0.0,
    )
    for candidate in oracle_candidates
]
for plan_index, plan in enumerate(oracle_plans):
    plan["plan_id"] = f"oracle_{plan_index}"
    plan["required_interaction_ids"] = ordered_open_interaction_ids(plan)
validation = build_v3_generation_validation(
    selected=selected,
    oracle_candidates=oracle_candidates,
    target_object=object_record,
    succ_pos_threshold=episode["task"]["succ_pos_threshold"],
)

episode["interactive_nav"] = {
    "schema_version": "interactive_nav_v3",
    "case_id": case_id,
    "parent_benchmark_episode_index": source_episode_index,
    "interaction_domains": ["container"],
    "interaction_requirement": "required",
    "target": build_v3_target(...),
    "success_criteria": build_success_criteria(...),
    "initial_state": build_initial_state(interaction_specs),
    "interactions": interaction_specs,
    "oracle_plan": oracle_plans[0],
    "oracle_plans": oracle_plans,
    "generation_validation": validation,
}
validate_v3_episode(episode)
```

## 14. 验收条件

每条迁移后的 container episode 必须满足：

- `schema_version == interactive_nav_v3`。
- `interaction_domains == ["container"]`。
- `interaction_requirement == required`；若以后生成无需容器交互的对照样本，则使用 `unnecessary` 和空 interactions。
- task、target 和 success criteria 的 selection mode 一致。
- 指定目标 ID 在三个目标字段中一致。
- interactions 覆盖所有 oracle plan 中的 open joint。
- 每个 open joint 的 interaction ID、object、joint name/index 完全一致。
- prerequisite IDs 存在且依赖无环。
- prerequisites 使用带 `mechanical`、`reachability` 或 `visibility` 类型的结构，不再使用 ID 裸列表。
- scene articulation state 与 interaction initial state 一致。
- hinge/slide 类型来自真实 joint type。
- controlling slide 使用 force control。
- observe target 使用 `head_camera` 和 threshold `0.0`。
- `oracle_plans[0] == oracle_plan`。
- 每个 plan 有唯一 `plan_id`，其 `required_interaction_ids` 与 open steps 顺序一致。
- success evidence 同时检查距离和可见性。
- object motion evidence 不替代可见性成功。
- episode 通过 v3 JSON Schema 和跨字段 validator。

## 15. 不应采用的快捷迁移

以下做法会产生语义不一致的数据：

- 只把 `interactive_nav_v2` 改成 `interactive_nav_v3`。
- 保留 `interaction_domain`，但不生成 `interaction_domains`。
- 只在 target 中保存 controlling joint，不生成 interactions。
- 继续输出全局单值 `role` 或 `prerequisite_interaction_ids`，而不是 `effect_types` 和 typed `prerequisites`。
- open_joint 不增加 interaction ID。
- 继续使用 `approach_target_container` 或 `prerequisite_for_target_compartment`。
- observe target 继续把 `1e-4` 当作 NavToObj 成功阈值。
- 只凭 slide object motion 将 visibility 标记为通过。
- 只凭 final visible pixels 将完整 success 标记为通过，而不检查距离。
- 将扁平 v2 validation 原样塞入 v3，但缺少固定必需栏目。
- 直接覆盖 Pydantic v2 model，导致历史 benchmark 无法读取。
