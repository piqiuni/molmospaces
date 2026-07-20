# Door Interaction v1 到 Interactive Navigation v3 迁移指南

本文用于指导修改旧通道门采集代码，使 `door_interaction_nav_v1` builder 直接生成 `interactive_nav_v3` episode。

适用代码与数据：

```text
scripts/InteractiveNav/build_door_interaction_benchmark.py
scripts/InteractiveNav/explore_molmo_interactions.py
scripts/InteractiveNav/dataset_definition/v1/benchmark.json
```

目标定义：

```text
scripts/InteractiveNav/dataset_definition/v3/interactive_nav_episode.schema.json
scripts/InteractiveNav/dataset_definition/v3/examples/channel_episode.json
```

## 1. 迁移范围

迁移目标是修改旧 builder 的生成逻辑，不是把历史 JSON 中的 `schema_version` 改成 v3。

door v1 已有的信息：

- 原始 MolmoSpaces `NavToObjTask` episode。
- 初始关闭的 door root ID。
- 为恢复路径需要打开的 door root ID。
- all-open、initial、oracle-restored 路径是否存在和路径长度。
- critical door、distractor door 和采样诊断。

door v1 缺少、但 v3 强制要求的信息：

- `task.selection_mode`。
- Instruction 类型、语言和交互披露级别。
- door leaf 的 `joint_name`、`joint_index` 和初始 qpos。
- 统一的 `target` 和 `success_criteria`。
- typed `interactions`。
- 门前交互 pose 和 typed `oracle_plan.steps`。
- 结构化 `generation_validation`。
- 最终 NavToObj 距离加 head-camera 可见性证据。

因此，对旧 v1 文件做纯离线转换只能得到不完整 v3。推荐在 `build_case_sample()` 仍持有真实环境、doorway analysis、路径和目标信息时直接构建 v3。

### 1.1 Interaction-unnecessary 历史负例

当前归档的 34 条 v1 episode 中：

```text
25 条 required_open_doors 数量为 1
9 条 required_open_doors 数量为 0
```

这 9 条均为 `distractor_doors_closed`，用于证明关闭无关门后任务仍可完成。当前 v3 已支持 interaction-unnecessary episode，因此可以正式迁移：

```python
interaction_requirement = (
    "required" if required_open_doors else "unnecessary"
)
interactions = build_required_interactions(required_open_doors)
```

对于 distractor-only case：

```text
interaction_requirement=unnecessary
interactions=[]
oracle_plan.required_interaction_ids=[]
oracle steps 不包含 open_joint
```

distractor door 的关闭状态继续保存在 `scene_modifications.articulation_states` 和 generation validation 中。`mixed_critical_and_distractor_closed` 使用 `interaction_requirement=required`，只把 critical door 写入必要 interactions。

## 2. 必须保持不变的 episode 字段

继续从原 benchmark episode 复制：

```text
source
house_index
scene_dataset
data_split
seed
robot
cameras
img_resolution
task.robot_base_pose
task.succ_pos_threshold
scene_modifications.object_poses
```

不要修改原目标物体 pose，也不要用门 ID 替换 `task.pickup_obj_name`。

当前门数据先使用单目标语义：

```python
selected_instance = episode["task"]["pickup_obj_name"]
episode["task"]["selection_mode"] = "specific_instance"
episode["task"]["pickup_obj_candidates"] = [selected_instance]
```

即使原 episode 包含多个同类 candidate，v3 门采集首版也只保留 `pickup_obj_name` 对应实例为成功目标。

## 3. 顶层字段补全

### 3.1 Task

目标结构：

```json
{
  "task_cls": "molmo_spaces.tasks.nav_task.NavToObjTask",
  "task_type": "nav_to_obj",
  "robot_base_pose": [0, 0, 0, 1, 0, 0, 0],
  "selection_mode": "specific_instance",
  "pickup_obj_name": "selected_object_id",
  "pickup_obj_candidates": ["selected_object_id"],
  "succ_pos_threshold": 1.5
}
```

`pickup_obj_name`、`pickup_obj_candidates`、`interactive_nav.target.selected_instance` 和 `success_criteria.target_selection` 必须一致。

### 3.2 Language

保留旧 Instruction 文本和 referral expression，并补充：

```python
episode["language"].update(
    {
        "instruction_type": "object_goal",
        "locale": "en",
        "interaction_disclosure": "hidden",
    }
)
```

首版不要求语言描述在 house 中唯一。不要在迁移阶段伪造 `grounding.unique=true`。

### 3.3 Scene modifications

door v1 历史 episode 通常只有：

```text
added_objects
object_poses
```

v3 必须补齐：

```json
{
  "added_objects": {},
  "object_poses": {},
  "removed_objects": [],
  "articulation_states": []
}
```

其中 `articulation_states` 是回放权威状态，不能只把门状态保存在 `interactive_nav.initial_state`。

## 4. 从 door root 获取可回放 leaf joint

旧 `door_state.closed_doors` 和 `oracle.required_open_doors` 保存的是 door root name。v3 `open_joint` 必须保存真实 leaf articulation object 和 joint name。

`set_door_root_state()` 已返回：

```text
door_root_name
state
hinge_body_names
transitions[]
```

其中每个 transition 已有：

```text
door_name
joint_index
joint_range
joint_position
state
```

首先修改 `set_door_state()`，在返回值中增加真实 joint name：

```python
return {
    "door_name": door_name,
    "joint_name": door.joint_names[hinge_idx],
    "joint_index": hinge_idx,
    "joint_range": [float(v) for v in joint_range],
    "joint_position": float(door.get_joint_position(hinge_idx)),
    "state": state,
}
```

这里的 `object_name` 应使用 transition 中可被 `Door(...)` 和 evaluator 重新解析的 leaf `door_name`。door root name 可以作为附加诊断字段保存，但不能代替 articulation replay 所需的 leaf object name。

### 4.1 初始 articulation state

设置完 case 的初始门状态后，记录所有参与状态构造的 leaf joint：

```python
{
    "object_name": transition["door_name"],
    "joint_name": transition["joint_name"],
    "joint_index": transition["joint_index"],
    "position": transition["joint_position"],
    "open_fraction": 0.0 if transition["state"] == "closed" else 1.0,
}
```

如果同一 leaf joint 被先 open 后 close，以最终初始状态为准，并按 `joint_name` 去重。

### 4.2 双开门 root

一个 door root 可能对应多个 leaf hinge。旧 `set_door_root_state()` 会同时操作全部 leaf。

v3 中应为每个 leaf joint 建立一个 interaction。若数据生成仍把双开门视为一次 root-level 操作，则 oracle 中依次列出这些 `open_joint` step，并在额外字段中保留共同的 `door_root_name`。

更严格的生成方式是逐个验证 leaf 子集，只保留恢复路径真正需要打开的 leaf。没有做子集验证时，不要声称某个单独 leaf 是最小必要 interaction。

## 5. 构建统一 Target

门任务的目标仍然是原 NavToObj 物体，不是门：

```python
target = {
    "selection_mode": "specific_instance",
    "category": target_category,
    "selected_instance": selected_instance,
    "instruction_consistent_candidates": [selected_instance],
    "container_name": None,
    "container_category": None,
    "grounding": {
        "unique": None,
        "matching_instance_count": None,
        "description": referral_description,
        "attributes": {},
    },
}
```

`target_category` 优先使用旧扫描阶段已经计算的类别；没有时再从目标实例名称或场景 metadata 解析。

## 6. 构建 Success Criteria

所有门和容器 episode 使用同一 NavToObj 成功条件：

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

旧 v1 的 `initial_state_path_found=false` 只能证明静态路径不可达，不能单独证明 NavToObj 失败。最终验证仍需要距离和 head-camera 可见性。

## 7. 构建 Interactions

每个 required leaf joint 生成一个 interaction：

```python
{
    "interaction_id": stable_interaction_id,
    "type": "channel_hinged_door",
    "object_name": leaf_door_name,
    "object_category": "Door",
    "joint_name": joint_name,
    "joint_index": joint_index,
    "effect_types": ["restore_reachability"],
    "prerequisites": [],
    "initial_state": {
        "joint_fraction": 0.0,
        "semantic_state": "closed",
    },
    "target_state": {
        "joint_fraction": 1.0,
        "semantic_state": "open",
    },
    "door_root_name": root_door_name
}
```

稳定 ID 建议只依赖 case 内稳定字段：

```python
interaction_id = f"channel::{root_slug}::{joint_index}"
```

当前 door builder 只控制 hinge，因此使用 `channel_hinged_door`。未来支持 slide door 时，应读取 MuJoCo joint type 后生成 `channel_sliding_door`，不要按资产名称猜测。

distractor closed door 保存在 `initial_state`、`scene_modifications.articulation_states` 和 `generation_validation` 中，但不出现在 `interactions` 或 plan-level `required_interaction_ids` 中。这样 evaluator 可以把 Agent 主动打开 distractor door 计为额外交互，而不会把它误认为 GT 必要动作。

## 8. 构建 Oracle Plan

door v1 只保存 required door root 列表和路径长度，没有门前机器人 pose，因此不能从旧 JSON 无损生成 v3 oracle。

新 builder 需要为每个 required door 计算：

- 门前 collision-free `goal_point`。
- 面向门的 `goal_yaw`。
- 起点或上一交互终点到该 pose 的 GT path。
- 开门后穿过通道或到下一交互点的 GT path。
- 最终满足 NavToObj 距离条件的目标观察 pose。

典型单门计划：

```json
{
  "plan_id": "oracle_0",
  "required_interaction_ids": ["channel::door_root::0"],
  "steps": [
    {
      "type": "navigate",
      "interaction_id": "channel::door_root::0",
      "goal_point": [0, 0, 0],
      "goal_yaw": 0.0,
      "position_tolerance_m": 0.25,
      "yaw_tolerance_rad": 0.35,
      "reason": "approach_channel_interaction"
    },
    {
      "type": "open_joint",
      "interaction_id": "channel::door_root::0",
      "object_name": "leaf_door_name",
      "joint_name": "leaf_joint_name",
      "joint_index": 0,
      "target_fraction": 1.0,
      "control_mode": "force",
      "reason": "restore_reachability"
    },
    {
      "type": "navigate",
      "interaction_id": null,
      "goal_point": [0, 0, 0],
      "goal_yaw": 0.0,
      "position_tolerance_m": 0.25,
      "yaw_tolerance_rad": 0.35,
      "reason": "satisfy_nav_to_obj_success"
    },
    {
      "type": "observe_target",
      "object_name": "selected_object_id",
      "camera_name": "head_camera",
      "visibility_threshold": 0.0,
      "reason": "verify_target_visible"
    }
  ]
}
```

多个 required door 应按 all-open GT path 上的实际通过顺序排列。不要直接使用“到起点距离”排序代替路径顺序。

`required_interaction_ids` 必须按 oracle 中首次执行 `open_joint` 的顺序排列，并与 open step 中出现的去重 interaction ID 完全一致。interaction-unnecessary plan 使用空列表。

如果开门后必须先穿过门再接近下一扇门，可以增加 reason 为 `traverse_open_channel` 的 `navigate` step。

## 9. Initial State

`interactive_nav.initial_state` 是分析镜像：

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
    "distractor_closed_door_roots": distractor_closed_doors,
}
```

回放权威值仍是 `scene_modifications.articulation_states`，两处必须一致。

## 10. Generation Validation 映射

建议映射：

| v1 字段 | v3 字段 |
|---|---|
| `paths.*` | `generation_validation.navigation_validation` |
| `diagnostics.*` | `generation_validation.navigation_validation` 或附加 `legacy_diagnostics` |
| required door transition | `generation_validation.interaction_validations[]` |
| 每次 open 后重新算的 path | `generation_validation.oracle_prefixes[]` |
| 无对应字段 | `generation_validation.success_evidence` |
| 无 leave-one-out 验证 | `minimal_plan_verified=null` |

最低合法结构：

```python
generation_validation = {
    "navigation_validation": {
        "all_open_path_length_m": old_paths["all_open_path_length_m"],
        "initial_state_path_found": old_paths["initial_state_path_found"],
        "oracle_restored_path_found": old_paths["oracle_restored_path_found"],
        "oracle_restored_path_length_m": old_paths["oracle_restored_path_length_m"],
    },
    "interaction_validations": interaction_validations,
    "oracle_prefixes": oracle_prefixes,
    "compartment_evidence": None,
    "success_evidence": {
        "status": "not_executed",
        "validation_mode": "path_feasibility_only",
        "target_object_name": selected_instance,
        "planar_distance_m": None,
        "distance_threshold_m": episode["task"]["succ_pos_threshold"],
        "camera_name": "head_camera",
        "visibility_fraction": None,
        "visible_pixels": None,
        "distance_passed": None,
        "visibility_passed": None,
        "expected_task_success": None,
    },
    "minimal_plan_verified": None,
    "legacy_case_type": old_case_type,
    "legacy_sampling": old_sampling,
    "legacy_diagnostics": old_diagnostics,
}
```

每个 `oracle_prefixes` item 至少补齐：

```python
{
    "plan_id": "oracle_0",
    "completed_step_count": completed_step_count,
    "robot_reachable_to_next_goal": reachable,
    "target_distance_passed": distance_passed,
    "target_visibility_fraction": visibility_fraction,
    "target_visible_pixels": visible_pixels,
    "task_success": bool(distance_passed and visibility_fraction > 0.0),
    "opened_interaction_ids": opened_interaction_ids,
}
```

只有真实设置终态机器人 pose，并同时测得：

```text
planar_distance < succ_pos_threshold
visibility_fraction > 0
```

才能将 `success_evidence.status` 写为 `passed`。

只有对 required interaction 做过 leave-one-out，删除任一必要 interaction 后任务重新不可完成，才能写 `minimal_plan_verified=true`。

## 11. 推荐修改函数

建议保留旧扫描逻辑，集中修改以下位置：

| 文件/函数 | 修改内容 |
|---|---|
| `explore_molmo_interactions.py::set_door_state` | 返回 `joint_name` |
| `build_door_interaction_benchmark.py::apply_closed_door_state` | 返回并保留全部 root/leaf transition |
| `build_door_interaction_benchmark.py::build_case_sample` | 计算门前 pose、oracle prefix path 和最终成功证据 |
| `build_door_interaction_benchmark.py::make_interactive_nav_payload` | 替换为 v3 payload builder |
| `build_door_interaction_benchmark.py::write_sample_json` | 在写出前执行 v3 schema 与语义校验 |
| `molmo_spaces/evaluation/benchmark_schema.py` | 增加 v3 Pydantic model 或 v2/v3 discriminated union |

不要删除旧 v1 reader。推荐先允许 evaluator 同时读取 legacy v1 和 v3，确认新 builder 稳定后再停止生成 v1。

## 12. Builder 级伪代码

```python
episode = copy.deepcopy(original_episode)
normalize_task_for_specific_instance(episode)
normalize_language(episode)

initial_transitions = apply_initial_door_state_and_capture_transitions(...)
articulation_states = flatten_final_joint_states(initial_transitions)

interactions = build_channel_interactions(
    required_open_doors=required_open_doors,
    transition_by_root=transition_by_root,
)
interaction_poses = compute_collision_free_door_poses(interactions)
oracle_plan = build_channel_oracle_plan(
    interactions=interactions,
    interaction_poses=interaction_poses,
    target_terminal_pose=target_terminal_pose,
)
oracle_plan["plan_id"] = "oracle_0"
oracle_plan["required_interaction_ids"] = ordered_required_interaction_ids
validation = validate_channel_oracle_prefixes(...)

episode["scene_modifications"] = normalized_scene_modifications(
    original=episode.get("scene_modifications", {}),
    articulation_states=articulation_states,
)
episode["interactive_nav"] = {
    "schema_version": "interactive_nav_v3",
    "case_id": case_id,
    "parent_benchmark_episode_index": episode_index,
    "interaction_domains": ["channel"],
    "interaction_requirement": (
        "required" if interactions else "unnecessary"
    ),
    "target": build_target(...),
    "success_criteria": build_success_criteria(...),
    "initial_state": build_initial_state(interactions, ...),
    "interactions": interactions,
    "oracle_plan": oracle_plan,
    "oracle_plans": [oracle_plan],
    "generation_validation": validation,
}
validate_v3_episode(episode)
```

## 13. 验收条件

每条迁移后的 door episode 必须满足：

- `schema_version == interactive_nav_v3`。
- `interaction_domains == ["channel"]`。
- `interaction_requirement` 与 interactions 是否为空一致。
- `selection_mode == specific_instance` 且三个目标字段一致。
- 每个 required leaf joint 同时存在于 articulation state、interaction 和 open_joint step。
- 每个 `open_joint.interaction_id` 能引用唯一 interaction。
- 初始 joint fraction 与 scene replay qpos 一致。
- required door 打开顺序与 GT path 通过顺序一致。
- `oracle_plans[0] == oracle_plan`。
- 每个 plan 都有唯一 `plan_id`，且 `required_interaction_ids` 等于其 open steps 中的 interaction ID 顺序。
- unnecessary episode 的 interactions 和 required interaction IDs 均为空，oracle 不包含 open step。
- 路径验证不能冒充完整 NavToObj 成功证据。
- 没有 leave-one-out 时 `minimal_plan_verified` 为 `null`，而不是 `true`。
- episode 通过 v3 JSON Schema 和跨字段 validator。

## 14. 不应采用的快捷迁移

以下做法会产生伪 v3 数据：

- 只把 `door_interaction_nav_v1` 改名为 `interactive_nav_v3`。
- 把 door root name 当作可直接回放的 joint name。
- 只保存 closed/open 标签，不保存真实 joint position。
- 把 `required_open_doors` 直接当作完整 oracle plan。
- 使用旧 path-found 结果填充 `success_evidence.status=passed`。
- 对双开门 root 未验证 leaf 子集，却声称单个 leaf 是唯一必要动作。
- 把 distractor door 当作 required interaction。
