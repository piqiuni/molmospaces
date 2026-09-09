# Container Interaction Probe 与 Benchmark 采集

本文档记录 `scripts/InteractiveNav/container_scene_probe.py` 及容器交互 benchmark 采集脚本当前已经实现的能力。它们用于读取 MolmoSpaces / ProcTHOR 场景，分析容器、目标物体、关节几何、可见性变化，生成可复现的 GT 交互计划，并提供可复用的开关动作封装、调试渲染和数据采集入口。

`container_scene_probe.py` 是 benchmark 和方法开发前的实验探针，不是最终在线 policy；`build_container_interaction_benchmark.py`、粗目录和并行采集脚本则负责把这些能力固化为可回放的 benchmark episode。当前代码主要回答：

- 场景里有哪些可交互容器、门和潜在目标物体。
- 哪些物体严格几何上位于容器内部。
- 打开/关闭某个 joint 后，目标是否从机器人 head camera 中变得可见。
- 多 joint 容器中，某个 joint 是否依赖另一个 joint 先打开。
- 机器人站在容器/门前时，开关前后的第一视角图像是否合理。
- 是否可以通过 episode spec 在场景中临时增加物体，用于容器内目标构造测试。

## 统一数据集定义 v3

通道交互、容器交互和二者混合 episode 的统一 JSON 规范位于：

```text
scripts/InteractiveNav/dataset_definition/v3/
```

该目录包含格式说明、JSON Schema，以及 channel、container、mixed、interaction-unnecessary 四类完整示例。v3 当前先固定 episode、Instruction、交互对象、初始状态、GT oracle 动作和生成验证字段，不强制目标语言 grounding 唯一；门交互示例也暂时采用单一指定目标。

当前容器 fine collection 已直接输出 `interactive_nav_v3`，并使用 `selection_mode=specific_instance`、显式 `interactions`、typed prerequisite、multi-oracle 和结构化 `generation_validation`。门交互与 mixed builder 不属于本次迁移范围，仍不能与新容器 JSON 直接拼接。

## 运行入口

推荐在 MolmoSpaces 环境中运行，并使用 EGL 做 headless MuJoCo 渲染：

```bash
source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate mlspaces
MUJOCO_GL=egl python scripts/InteractiveNav/container_scene_probe.py <command> ...
```

默认配置：

- 默认数据集：`procthor-10k`
- 默认 split：`train`
- 默认机器人：`rby1`
- 默认输出目录：`${REPO_ROOT}/scripts/InteractiveNav/output/container_scene_probe`
- 默认 benchmark：`${REPO_ROOT}/assets/benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/NavToObjProcthor10kBench_20260112_json_benchmark/benchmark.json`

脚本加载场景时会把资源准备到可写 mirror：

- `WRITABLE_ASSET_MIRROR=/tmp/container_scene_probe_assets`
- 支持从 `MLSPACES_ASSETS_DIR`、`MLSPACES_CACHE_DIR`、本仓库 `assets/`、`/tmp/container_probe_assets_*`、`/tmp/container_probe_resources_*` 等路径解析场景资源。

## 场景加载与机器人默认姿态

核心入口是 `load_scene_context(args, house_ind)`。它会构建 `NavToObjBaseConfig`，使用 `SceneOnlyTaskSampler` 只加载场景而不采样具体导航任务。

加载后会记录初始化时的机器人头部和躯干关节：

- `initial_head_qpos`
- `initial_torso_qpos`

随后脚本会应用默认姿态来减少遮挡：

- `apply_default_arm_pose(env)`：设置 RBY1 左右臂到预定义收纳姿态，避免机械臂遮挡 head camera。
- `apply_default_torso_pose(env, default_qpos)`：恢复或设置躯干默认姿态。
- `apply_default_head_pose(env, default_qpos)`：恢复初始化时读取到的头部姿态，若不可用则使用 `DEFAULT_HEAD_QPOS`。

针对抽屉内观察，还封装了两类视角动作：

- `lean_torso_for_drawer_view(env, pitch_delta=0.35)`：增加躯干前倾，使相机更靠近抽屉内部。
- `lower_head_for_drawer_view(env, tilt_delta=0.35, pan=None)`：在当前头部姿态基础上低头。
- `raise_head_to_pose(env, head_qpos=None)`：抬头恢复到默认或指定姿态。

## 场景对象扫描

`collect_scene_records(ctx)` 会遍历场景 metadata 中的所有对象，并输出统一 record：

- `name`
- `body_id`
- `category`
- `position`
- `quat`
- `aabb_center`
- `aabb_size`
- `parent`
- `parent_chain`
- `support_below`
- `room`
- `is_structural`
- `is_articulable`
- `has_free_joint`
- `interaction_group`
- `is_receptacle`
- `asset_id`

`interaction_group` 的粗分类包括：

- `container`：名称或类别命中 drawer、cabinet、fridge、refrigerator、microwave、oven、dishwasher、box、chestofdrawers 等 token，且是 articulable 对象。
- `portal`：门、gate-like 对象等通道交互对象。
- `other`：普通对象。

对 articulable container，脚本还会记录每个 joint：

- `joint_index`
- `joint_name`
- `joint_type`
- `joint_range`
- `current_value`
- `closed_value`
- `open_value`

`collect_door_records(ctx)` 会独立收集可交互门：

- `name`
- `body_id`
- `hinge_joint_index`
- `hinge_joint_name`
- `hinge_joint_range`
- `closed_value`
- `open_value`

## 容器-物体关系判断

`compute_relation(container_rec, object_rec)` 会计算容器和目标物体之间的关系。当前最重要的字段是：

- `inside_aabb`：目标物体 AABB 是否完全位于容器 AABB 内。
- `overlap_ratio`：目标物体体积中与容器 AABB 重叠的比例。
- `parent_chain_hit`：metadata parent chain 是否命中容器。
- `parent_hit`：metadata parent 是否命中容器。
- `support_hit`：support_below 是否命中容器。
- `distance_to_container_center`：物体与容器中心距离。
- `label`：最终关系标签。
- `score`：综合打分。

当前 `inside` 判断已经改为严格几何优先：

- `inside`：`inside_aabb=True`，或 `overlap_ratio > 0.8`。
- `attached_to_container`：metadata parent/support 命中，但几何上不满足 inside。
- `likely_inside`：parent chain 命中，或 `overlap_ratio > 0.25`。
- `near_container`：距离容器中心小于 `0.75m`。
- `unrelated`：无明显关系。

这个修改的原因是 ProcTHOR/MolmoSpaces 的 parent/support metadata 经常把“放在柜子上、桌面上、抽屉柜表面上”的物体标为容器相关。如果直接把 parent/support 当作 inside，会产生大量假阳性。

## 可见性变化测量

`measure_container_visibility(ctx, container_rec, object_names, output_dir=None)` 用于测试打开/关闭容器 joint 前后，目标物体在机器人 head camera 中的可见性变化。

流程如下：

1. 对容器每个 joint 读取 `closed_value` 和 `open_value`。
2. 调用 `choose_pose_valid_for_joint_states(...)` 寻找同一个机器人 base pose，要求 closed 和 open 两种状态下都不碰撞。
3. 将机器人放到共享 pose，并恢复默认手臂/相机姿态。
4. 关闭 joint 后渲染 `head_camera` RGB、segmentation，并调用 `env.check_visibility("head_camera", *object_names)`。
5. 打开 joint 后重复渲染与 visibility 检查。
6. 输出每个目标的 `closed_visibility`、`open_visibility`、`delta_visibility` 和 `became_visible`。

如果传入 `output_dir`，会额外保存：

- `closed_rgb`
- `open_rgb`
- `closed_seg`
- `open_seg`
- `closed_mask`
- `open_mask`

其中 segmentation preview 来自 `env.render_segmentation_frame("head_camera")`，mask 来自 `get_geom_seg_mask(...)`。

## 机器人站位策略

容器和门的调试渲染都不会直接使用 `env.place_robot_near`。当前脚本会根据容器/joint 几何计算机器人候选位置。

关键函数：

- `container_front_axis(container_rec)`：根据容器 AABB 尺寸和朝向估计正面方向。
- `joint_target_geometry(env, container_rec, joint)`：用 joint 驱动 body 的 visual AABB 作为目标 joint 的几何中心。
- `choose_collision_free_pose(...)`：从 thormap free points 中找距离候选点最近且无碰撞的位置。
- `choose_pose_valid_for_joint_states(...)`：寻找 closed/open 状态共用的无碰撞 pose。
- `place_robot_in_front_of_container(...)`：将机器人放在容器正前方。
- `place_robot_for_container_joint(...)`：将机器人放在指定 joint 前方。

默认期望距离：

- 一般容器/门视角：约 `0.8m`
- 抽屉内部调试：`debug-drawer-bound-object` 默认 `--view_distance 0.45`

站位候选会尝试：

- 正面位置。
- 正面稍微后退。
- 左右侧偏移。
- 后侧 fallback。

最终选择在 free map 上最近、且 collision check 通过的 base pose。

## 容器开关动作封装

当前已有可复用的容器 joint 开关函数：

```python
set_container_joint_fraction(env, container_name, joint_index, open_fraction)
open_container_joint(env, container_name, joint_index)
close_container_joint(env, container_name, joint_index)
```

语义：

- `open_fraction=0.0`：目标 joint 设到 joint range 的最小值，视为 closed。
- `open_fraction=1.0`：目标 joint 设到 joint range 的最大值，视为 fully open。
- 中间值按 joint range 线性插值。

底层不是直接写 qpos，而是调用：

```python
drive_joint_to_value_with_force(env, joint_name, target_value)
```

它会根据 MuJoCo joint 类型选择外力/外力矩 PD 控制：

- slide joint：对 joint body 施加线性 force。
- hinge joint：对 joint body 施加 torque。

返回 metadata 包括：

- `method`
- `joint_name`
- `joint_type`
- `joint_range`
- `target_value`
- `final_value`
- `final_error`
- `steps`
- `reached`

脚本中也保留了 `set_articulation_state_by_record(...)`，可直接写 joint position。它主要用于几何分析和 deterministic debug，不建议把它当成最终交互动作接口。

## 门开关动作封装

门交互和容器交互使用同样的封装风格：

```python
set_door_open_fraction(env, door_name, open_fraction)
open_door_space(env, door_name)
close_door_space(env, door_name)
```

门对象通过 `Door(door_name, env.current_data)` 读取 hinge joint：

- `get_hinge_joint_index()`
- `get_joint_range(hinge_idx)`
- `door.joint_names[hinge_idx]`

`debug-door-view` 会测试 closed/open 两个状态，并要求两个状态使用同一个机器人 pose。输出包括 RGB 和 segmentation 图。

## Joint Box 与依赖关系推断

当前脚本把每个 articulation joint 关联到 MuJoCo body：

```python
joint_id = mj_name2id(model, mjOBJ_JOINT, joint_name)
body_id = model.jnt_bodyid[joint_id]
```

再用该 body 的 visual AABB 作为 joint box proxy：

```python
body_aabb(model, data, body_id, visual_only=True)
```

`collect_joint_box_state_records(...)` 会记录每个 joint：

- closed 状态下的 box。
- 单独打开该 joint 后的 open box。
- `open_delta`，即 open center 相对 closed center 的位移。
- MuJoCo joint type：`slide`、`hinge` 等。

`infer_joint_open_dependencies(...)` 用于判断“打开当前 joint 前，是否需要先打开其他 joint”。

当前默认方法是：

```bash
--dependency_method front_occlusion
```

推荐理解为“从正面看，内部 slide joint 是否被外部门体 hinge joint 阻挡”。主要规则：

- 只把 slide joint 作为可能需要先打开其他 joint 的目标。
- 只把 hinge joint 作为可能的 prerequisite。
- 对 slide joint 计算 closed box 到 open box 的 swept AABB。
- 在 slide 运动方向坐标系中比较 swept box 与候选 hinge closed box。
- 需要满足明显的 z overlap、depth overlap，以及 lateral overlap 或足够接近。
- 每个 slide target 只选择得分最高的 hinge blocker。

这样做的目的是避免早期 `open_aabb_overlap` 方法产生环状依赖，例如 `j0 -> j3` 同时 `j3 -> j0`。旧方法仍保留为可选对照：

```bash
--dependency_method open_aabb_overlap
```

但当前不建议把它作为 fridge/cabinet 的主判断方式。

依赖输出中包含：

- `prerequisite_joint_indices`
- `prerequisite_joints`
- `closed_box`
- `open_box`
- `open_delta`
- `front_axis_xy`
- `front_axis_source`
- `slide_hinge_blocker_evidence`

## Joint 3D 可视化

`save_joint_dependency_plot(...)` 会为 articulation 绘制 3D box 图：

- 每个 joint 的 closed box。
- 每个 joint 的 open box。
- closed 到 open 的运动虚线。
- prerequisite 到 target 的箭头。
- 统一的 front 方向箭头。

图中坐标轴不是世界坐标，而是为了方便比较的局部坐标：

- `x=lateral`
- `y=front_depth`
- `z=z`

这样可以保证不同冰箱/柜子的 front 指向图中同一侧，便于人工检查遮挡关系。

## 物体增加测试

`add-object-test` 用于测试“是否能在场景中增加一个物体，并读取回其位置”。它会构造一个 synthetic `EpisodeSpec`，其中包含：

- `scene_modifications.added_objects`
- `scene_modifications.object_poses`
- `task.pickup_obj_name`
- `task.pickup_obj_candidates`
- `language.task_description`

如果没有指定 `--object_relpath`，脚本会从当前场景中选择一个 portable preferred asset 作为复用对象。候选类别 token 包括：

- apple
- mug
- bowl
- atomizer
- alarmclock
- vase
- laptop
- basketball

输出包括：

- `synthetic_episode.json`
- `add_object_result.json`
- `add_object_debug.png`

需要注意：这是验证 scene modification 是否可行的测试 harness，不是最终的数据集生成器。当前 object pose 只是根据容器 AABB 粗略放置，不能保证真实物理稳定地落入具体抽屉空间。

## CLI 子命令

### scan-houses

扫描一个或多个 house 的容器、目标物体、关系和可见性变化。

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/container_scene_probe.py \
  scan-houses \
  --house_inds 1 2 3
```

每个 house 输出：

- `all_objects.json`
- `containers.json`
- `target_objects.json`
- `relations.json`
- `visibility.json`
- `top_candidates.json`
- 若干 `relation__*.png`
- `visibility_images/*.png`

总输出：

- `scan_report_all_houses.json`

### scan-container-target-overlap

把 nav benchmark episode 中的目标与场景里的严格几何 inside 物体做关联。

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/container_scene_probe.py \
  scan-container-target-overlap \
  --max_episodes 2000
```

输出：

- `inside_objects.json`
- `summary.json`
- 运行中会写 `.partial.json`，方便长任务中断后查看进度。

### scan-joint-dependencies

扫描一个 house 中所有容器和门的 joint dependency，也可以指定单个 articulation。

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/container_scene_probe.py \
  scan-joint-dependencies \
  --house_ind 1 \
  --dependency_method front_occlusion \
  --save_plots
```

指定某个冰箱/抽屉柜：

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/container_scene_probe.py \
  scan-joint-dependencies \
  --house_ind 1 \
  --articulation_name Fridge_5 \
  --dependency_method front_occlusion \
  --save_plots
```

输出：

- `joint_dependencies.json`
- `joint_dependency_plots/*.png`

### debug-container-view

设置某个容器 joint 到指定值，并从 front/left/right 三个机器人视角保存 head camera 图像。

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/container_scene_probe.py \
  debug-container-view \
  --house_ind 1 \
  --container_name <container_name> \
  --joint_index 0 \
  --joint_value 1.0
```

输出：

- `front_head_rgb.png`
- `left_head_rgb.png`
- `right_head_rgb.png`
- `front_head_default_rgb.png`
- `front_head_look_down_rgb.png`
- `front_head_restored_rgb.png`
- `debug_view_result.json`

### debug-drawer-bound-object

查找某个抽屉 joint box 内的 free-joint 目标物体，并输出关/开抽屉后的低头、俯身第一视角图像。

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/container_scene_probe.py \
  debug-drawer-bound-object \
  --house_ind 1 \
  --container_name <container_name> \
  --joint_index 0 \
  --view_distance 0.45 \
  --torso_lean_delta 0.45 \
  --head_tilt_delta 0.25
```

默认会输出四种 case：

- `closed_look_down`
- `open_without_move_look_down`
- `open_with_moved_object_look_down`
- `open_with_force_look_down`

每个 case 保存：

- RGB 图。
- segmentation preview。
- visibility 数值。
- object AABB。
- head qpos。
- torso qpos。
- 实际 joint value。

其中 `open_with_moved_object_look_down` 是保留的诊断项，不应当视为正确物理结果。它使用 joint body global delta 手动移动目标物体，可能把物体错误地绑定到另一个抽屉空间。

### debug-door-view

测试门 closed/open 两个状态下的机器人第一视角。

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/container_scene_probe.py \
  debug-door-view \
  --house_ind 1
```

也可以指定门：

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/container_scene_probe.py \
  debug-door-view \
  --house_ind 1 \
  --door_name <door_name>
```

输出：

- `closed_head_rgb.png`
- `closed_head_seg.png`
- `open_head_rgb.png`
- `open_head_seg.png`
- `debug_door_result.json`

### add-object-test

测试向场景中增加一个物体，并读取回新增物体 AABB。

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/container_scene_probe.py \
  add-object-test \
  --house_ind 1 \
  --container_name <container_name>
```

也可以指定对象 asset：

```bash
MUJOCO_GL=egl python scripts/InteractiveNav/container_scene_probe.py \
  add-object-test \
  --house_ind 1 \
  --container_name <container_name> \
  --object_relpath objects/<asset>/<asset>.xml \
  --object_name custom_probe_object
```

输出：

- `synthetic_episode.json`
- `add_object_result.json`
- `add_object_debug.png`

## 已验证/参考输出

当前已有的调试输出主要保存在：

```text
${REPO_ROOT}/scripts/InteractiveNav/output/container_scene_probe
```

与 fridge joint dependency 相关的参考产物包括：

- `fridge_joint_dependency_geometry_summary_0_10.md`
- `fridge_joint_dependency_geometry_plots_montage.png`
- `new_fridge_joint_dependency_geometry_summary.md`
- `new_fridge_joint_dependency_geometry_plots_montage.png`

这些输出可用于人工检查不同 fridge asset 的 joint box、front 方向和依赖关系是否合理。

## 当前已知 Bug

- 抽屉/冰箱内部物体不一定随 joint 真实运动。当前测试中发现，直接打开抽屉时，抽屉内物体的位置可能没有跟随抽屉移动；`debug-drawer-bound-object` 中保留的 `open_with_moved_object_look_down` 只是诊断用的手动位姿补偿，不应被当作正确物理结果。
- 手动用 joint body delta 移动物体可能绑定到错误抽屉。对于多抽屉柜，如果只根据某个 joint body 的 global delta 搬动物体，可能出现“打开一个抽屉，却把另一个抽屉中的物体带出来”的错误。
- 多 joint 容器的依赖关系仍需要人工抽查。当前 `front_occlusion` 已经比 `open_aabb_overlap` 更稳定，但它主要覆盖 fridge 中“外部门体挡住内部 slide 抽屉”的情况；复杂 cabinet、drawer、oven 等结构仍可能漏判或误判。
- `open_aabb_overlap` 依赖判断会产生循环依赖。典型错误是 `j0 -> j3` 与 `j3 -> j0` 同时成立，因此目前只作为 debug 对照，不作为推荐结果。
- 机器人视角仍可能被机身、手臂、容器门体或狭窄空间遮挡。虽然脚本已经支持手臂默认姿态、躯干前倾、低头和 shared pose 检查，但某些 house 中容器前方 free space 太少，仍可能需要调 `view_distance`、`torso_lean_delta`、`head_tilt_delta`。
- closed/open 可见性有时会受站位失败影响。如果 `choose_pose_valid_for_joint_states(...)` 找不到同一个 closed/open 都无碰撞的 pose，该 joint 的 visibility 测量会跳过，导致 `visibility.json` 中缺少对应条目。
- 物体增加测试还不能保证“放入容器内部”。`add-object-test` 目前按容器 AABB 粗略放置 object pose，只能验证 scene modification/readback 成功，不能保证物体稳定落在具体抽屉、冰箱层板或柜格内。
- 当前可见性仍依赖 GT object name。`measure_container_visibility(...)` 直接调用 `env.check_visibility("head_camera", object_name)`，还没有接入真实感知输出的 box/name，也没有实现感知误差下的相似匹配。
- segmentation/mask 预览只是调试图。当前 `save_segmentation_preview(...)` 使用 geom id 和 object type 做伪彩色，可用于人工检查，但不是语义分割真值图的最终发布格式。
- force-drive 开关动作可能不收敛。`drive_joint_to_value_with_force(...)` 会返回 `reached=False` 和 `final_error`，后续调用方需要检查该字段；当前 CLI 主要保存 metadata，没有做失败重试策略。

## 待实现内容

- 封装面向 planner 的稳定交互 API。当前已有 `open_container_joint`、`close_container_joint`、`open_door_space`、`close_door_space`，但还需要统一成“给定对象 id/name + joint index + desired state，自动处理依赖、动作结果、失败回退”的接口。
- 实现依赖感知的开关序列。对于 `j1` 依赖 `j0` 的冰箱内部抽屉，最终接口应该先打开 prerequisite joint，再打开 target joint；关闭时也需要定义逆序或安全序列。
- 增加从感知输出到 GT 对象的匹配。输入应支持 detector 输出的 `category/name`、2D/3D box、置信度和可选语言描述，通过类别相似度、空间 IoU/距离、可见性和 room/context 匹配到 MolmoSpaces object id。
- 构建容器交互 benchmark episode。需要从扫描结果中选择合适 house、容器、joint、目标物体、机器人起点和语言描述，形成可复现 JSON benchmark，而不是只输出 probe 结果。
- 生成真实容器内目标数据。对于现有场景没有合适 inside 目标的情况，需要扩展 `add-object-test`，支持选择具体容器 cavity、具体 joint 空间、稳定放置 pose、碰撞检查、重力 settling 和 readback 验证。
- 完善容器内部空间建模。目前只用 joint body AABB 和 container AABB，后续需要显式表示“抽屉内部空间/冰箱层板空间/柜格空间”，否则很难判断目标是放在容器里、放在顶部，还是放在旁边。
- 增加开关前后物体绑定验证。需要系统检查 open/close 过程中目标物体是否随容器部件运动，若不随动，需要判断是资产建模问题、MuJoCo joint/parent 问题，还是物体本身 free joint 不应绑定。
- 为不同容器类型分别调依赖规则。fridge、chest of drawers、cabinet、microwave、oven、dishwasher 的 joint 语义不同，后续应按 asset/category 统计 joint 数量、joint 类型、motion delta 和依赖模式。
- 保存更完整的可视化报告。当前已有 RGB、seg、mask 和 3D box 图，后续可以自动生成每个 articulation 的 montage，把 closed/open、不同 joint、依赖箭头、机器人 pose 和 visibility delta 放在一张报告图里。
- 接入交互导航评测指标。容器交互不应只看 `became_visible`，还应记录交互次数、错误交互次数、打开依赖是否满足、交互前后路径变化、最终导航成功率和目标可见/可达状态变化。
- 拆分脚本结构。当前 `container_scene_probe.py` 同时承担扫描、动作、渲染、debug、episode 构造。后续建议拆成 `scene_scan`、`interaction_oracle`、`debug_render`、`benchmark_build` 等模块，保留当前 CLI 作为入口。
- 增加最小单元测试。优先覆盖 AABB inside 判断、joint dependency 推断、front/local 坐标投影、感知 box 到 GT 匹配、open/close fraction 到 joint target 的转换。

## 当前限制与注意事项

- AABB 是 joint body 的 visual box proxy，不是精确 mesh，也不包含把手、孔洞、容器内部 cavity 的精确几何。
- `front_occlusion` 依赖推断主要针对“外部门体 hinge 阻挡内部 slide 抽屉”的 fridge/cabinet 模式，其他复杂机构仍需要人工检查 3D 图。
- `open_aabb_overlap` 方法容易产生循环依赖，目前只建议作为对照。
- 容器内物体是否随抽屉运动仍存在物理/建模问题。当前脚本保留了诊断图，但不能把手动移动物体的结果当成真实交互。
- `add-object-test` 只能证明 scene modification 和 readback 成功，不保证新增物体已经稳定、合理地放入某个具体抽屉或柜格。
- head camera 图像质量依赖机器人站位、躯干/头部姿态和场景 free map。如果某些容器前方空间很窄，仍可能需要按场景调参。
- 输出目录下的 JSON 和图片是实验产物，不应直接作为固定 benchmark 真值使用。

## Container Interaction Benchmark 采集

### 采集目标与边界

当前 benchmark 只研究容器交互，不把通道门作为任务变量：初始化时打开可直接控制的 door，关闭所有容器 joint。样本目标是严格位于 Fridge 或 Dresser 内部的已有 free-joint 物体，不注入新物体，也不复用原 `nav_to_obj` episode 的 `pickup_obj_name`。

每个有效 episode 保存：

- 原 benchmark house、场景、机器人、相机和一个原始机器人起点。
- 唯一 GT 目标物体、GT 容器和 controlling joint 候选。
- 所有 door 打开、所有容器 joint 关闭的初始 articulation state。
- 到容器交互位姿的 GT shortest path 可达性和长度。
- 使目标首次从 head camera 中可见的一个或多个 `oracle_plan`。
- 每个 oracle prefix 的 joint 状态、目标像素、visibility fraction、物体位置和 joint box。

运行时 policy observation 不应暴露 `interactive_nav.oracle_plan`、GT 容器 ID 或 controlling joint；这些字段是 benchmark 构建真值和训练监督。

### 两阶段主线

正式数据管线只保留两个必要阶段：

```text
原始 NavToObj benchmark
  -> Stage 1: rough catalog
  -> Stage 2: fixed-house fine collection
  -> benchmark + validation + evidence
```

`select_container_interaction_candidates.py` 仅保留为旧 candidate manifest 复现和离线统计工具，不再属于正式采集主线。新的精采入口直接读取粗目录，在内存中生成并保存 `collection_plan.json`。

### Stage 1：粗目录扫描

入口：

```bash
python scripts/InteractiveNav/collect_container_rough_catalog_parallel.py \
  --benchmark_dir assets/benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/NavToObjProcthor10kBench_20260112_json_benchmark \
  --output_dir scripts/InteractiveNav/output/container_interaction_all_houses/rough_catalog \
  --workers 4 \
  --mujoco_gl egl
```

粗扫描按 unique house 加载场景，只做静态/几何目录构建，不做以下操作：

- 不构建 GT occupancy path。
- 不寻找机器人交互位姿。
- 不执行容器开关的完整物理验证。
- 不渲染每个 pair 的 RGB、segmentation 或 mask。
- 不因为路径、可见性或 joint oracle 失败而删除容器-物体 pair。

`rough_catalog.json` 中每个 house 主要保存：

- `house_index`、模板 episode 和该 house 的全部 `source_episode_indices`。
- Fridge/Dresser 实例的 `name`、`category`、`asset_id`、AABB 和 joint 列表。
- 每个 strict contained object 的实例 ID、类别、AABB 中心和尺寸。
- 原 benchmark 起点以及起点到物体/容器中心的欧氏距离和二维距离。
- 冰箱 slide joint box 中的几何候选物体，用于后续 multi-interaction 检查。

`asset_id` 表示可复用的 3D/articulation 模板，例如 `Fridge_17` 或 `Dresser_220_1`；`container_name` 才是当前 house 中的唯一实例 ID。

当前已完成的全量粗扫描结果：

| 指标 | 数量 |
|---|---:|
| 原 benchmark episode | 2000 |
| unique house | 898 |
| 完成扫描 house | 898 |
| 失败 house | 0 |
| strict container-object pair | 2603 |
| Fridge 实例 | 629 |
| Dresser 实例 | 1158 |
| 包含严格内部目标的 Fridge | 558 |
| 包含严格内部目标的 Dresser | 709 |
| 冰箱 slide joint | 1416 |
| slide joint box 中有物体的 joint | 29 |
| slide joint box 中物体 | 32 |

4 worker 全量扫描耗时约 `5654 s`，即 `1.57 h`。结果位于：

```text
scripts/InteractiveNav/output/container_interaction_all_houses/rough_catalog/
  rough_catalog.json
  house_catalog.json
  summary.json
  shards/
```

### Stage 2：固定 House 动态精采

并行入口：

```bash
python scripts/InteractiveNav/collect_container_fine_parallel.py \
  --benchmark_dir assets/benchmarks/molmospaces-bench-v2/procthor-10k/NavToObjDataGenConfig/NavToObjProcthor10kBench_20260112_json_benchmark \
  --rough_catalog scripts/InteractiveNav/output/container_interaction_all_houses/rough_catalog/rough_catalog.json \
  --output_dir scripts/InteractiveNav/output/container_interaction_all_houses/benchmark \
  --target_house_count 100 \
  --max_samples 200 \
  --samples_per_house 2 \
  --workers 4 \
  --mujoco_gl egl \
  --no-save_images \
  --no-save_plots
```

串行或两阶段统一入口：

```bash
python scripts/InteractiveNav/collect_container_interaction_benchmark_serial.py \
  --target_house_count 100 \
  --max_samples 200 \
  --samples_per_house 2 \
  --rough_workers 4 \
  --fine_workers 4 \
  --no-save_images \
  --no-save_plots
```

已有粗目录时可跳过 Stage 1：

```bash
python scripts/InteractiveNav/collect_container_interaction_benchmark_serial.py \
  --skip_rough \
  --target_house_count 100 \
  --max_samples 200 \
  --samples_per_house 2 \
  --fine_workers 4
```

### Fixed 采样语义

`fixed` 表示在精采开始前确定 house 集合，而不是先建立 150 个 house 再保留成功的 100 个：

- `target_house_count` 限制固定 house 数；也可用 `--house_indices 0,1,2` 显式指定。
- `samples_per_house=n` 是每个固定 house 的请求上限。
- `max_samples` 是全部固定 house 的总请求上限，配额按 round-robin 分配。
- 每个 house 的候选池包含粗目录中的全部 strict pair，不再只有少量 slot backup。
- 候选成功后才占用样本配额；失败后继续尝试同 house 下一个候选。
- 候选池耗尽仍不足 `n` 条时，保留已经成功的 0 到 `n-1` 条，然后继续下一个固定 house。
- 不完整 house 不回滚，也不会自动使用固定集合之外的 house 替换。

`collection_plan.json` 是精采入口根据粗目录自动生成的可复现计划，不是用户需要提前运行的第三阶段。计划首先按容器类别、目标类别、container asset 和 near/medium/far 起点距离做全局均衡，然后把该 house 的其余 strict pair 全部追加为 fallback。

当前粗目录中有 712 个 house 至少包含一个 Fridge/Dresser strict pair：

- `samples_per_house=2` 时最多请求 1424 条。
- `samples_per_house=3 --max_samples=2000` 时可以请求 2000 条。
- 2603 是粗目录 pair 上限，不代表 2603 条都会通过 GT path 和交互验证。

### 单个候选的精采验证流程

每个候选按以下顺序验证：

1. 按原 episode object pose 恢复场景，打开 door，关闭所有容器 joint。
2. 根据 joint 正面遮挡关系推断 prerequisite，并生成 `dependencies + target_joint` 序列。
3. 对 hinge/slide joint 分别生成容器前方交互位姿；要求同一位姿对整个开关序列无碰撞。
4. 遍历该 house 的原 benchmark 起点，要求起点处目标不可见。
5. 在 door-open GT occupancy map 上计算起点到交互位姿的 shortest path；`path is None` 的起点不可用。
6. 在 `default` 和 `drawer_low_view` 观察姿态下执行 visibility trace。
7. hinge 候选要求目标像素从 0 变为大于 0；多个 joint 都可揭示目标时保留为 `oracle_plans` 并标记歧义。
8. slide 候选使用力控制；joint 必须达到目标开度，且目标最终必须在 head camera 中产生正 visibility fraction。
9. 成功后生成 `NavToObjTask` episode；失败候选写入 `rejected_pairs.json`，然后继续同 house fallback。

当前 reveal mode 包括：

- `hinge_pixel_reveal`：打开 hinge 后 head camera 中目标像素由 0 变为正数。
- `force_slide_visibility`：力控制打开 slide joint 后目标被 head camera 观察到。
目标与 moving compartment 的一致位移只写入 `compartment_evidence`，用于证明物体属于抽屉空间；它不能替代 `visibility_fraction > 0` 的 NavToObj 终态成功条件。

GT path 的终点是满足容器正面、朝向、开关空间和碰撞约束的 `interaction_pose`，不是物体中心，也不是直线距离。当前容器样本的路径通常比原始 NavToObj 更长，因为原任务可导航到同类别任一候选附近，而容器任务绑定具体 GT pair 和具体交互位姿。

### Oracle Plan 与 Episode JSON

生成 episode 继续使用 `NavToObjTask`，但替换为严格目标物体 ID，并增加 `interactive_nav`：

```text
interactive_nav
  schema_version: interactive_nav_v3
  interaction_domains: [container]
  interaction_requirement: required
  case_id
  parent_benchmark_episode_index
  target
  success_criteria
  initial_state
  interactions
  oracle_plan
  oracle_plans
  generation_validation
```

`target` 保存 `selected_instance`、目标类别、GT 容器、AABB 和 grounding 诊断。`grounding.matching_instance_count` 来自当前 scene 全部目标记录的同类别实例计数，`grounding.unique` 由该实测计数是否等于 1 得出；容器类别缺失时直接拒绝样本，不写入 `unknown`。controlling joint 不再放在 target 主字段中，而是转换为 episode 级 `interactions`。自然语言只使用目标类别，例如 `find the pencil.`，不向 policy 暴露 GT 容器。

`scene_modifications.articulation_states` 是 articulation 回放的权威来源，其中 `position` 和 `open_fraction` 均由设置状态后的 MuJoCo qpos 读回计算；`interactive_nav.initial_state.interaction_states` 保存与 interaction ID 对齐的状态镜像，并附加：

- `all_doors_open=true`
- `container_joints_closed=true`
- 每个 interaction 的 `joint_fraction=0` 和 `semantic_state=closed`

`interactions` 将所有 oracle 中出现的 joint 去重。hinge 映射为 `container_hinged_door`，slide 映射为 `container_sliding_drawer`；前置 joint 使用 `enable_interaction`，最终 controlling joint 使用 `reveal_target_object`，冰箱外门与内部抽屉依赖写为 `mechanical` prerequisite。

`oracle_plan.steps` 支持：

- `navigate`：容器交互 `goal_point`、`goal_yaw`、容差、首个 interaction ID 和 `approach_container_interaction` 原因。
- `set_view`：需要低头/俯身时设置 head/torso qpos，原因为 `improve_target_visibility`。
- `open_joint`：通过 `interaction_id` 依次打开 prerequisite 和 controlling joint；前置原因为 `prerequisite_for_interaction`，slide controlling joint 使用 `control_mode=force`。
- `observe_target`：使用 `head_camera` 和严格 `visibility_threshold=0.0` 检查目标可见性，原因为 `verify_target_visible`。

存在多个独立有效交互方案时，`oracle_plan` 保存第一个方案，`oracle_plans` 保存全部方案。当前不保存 waypoint；运行时应根据 `navigate.goal_point` 在线规划路径。

`generation_validation` 按 v3 固定栏目保存：

- `navigation_validation`：GT path、起点可见性、goal 和 collision-free interaction pose。
- `interaction_validations`：各 oracle joint sequence 的仿真诊断。
- `oracle_prefixes`：逐 prefix 已打开 interaction、visibility 和任务成功状态。
- `compartment_evidence`：slide joint 与内部物体的一致运动证据，仅作 compartment 归属验证。
- `success_evidence`：同一终态的 GT 平面距离与 head-camera visibility；二者必须同时通过。
- `door_state_validation`：逐门保存读回 qpos、closed/open 端点、归一化 fraction 和阈值判定；只有全部门通过才生成 episode。
- `container_state_validation`：逐容器 joint 保存相同的读回信息；只有全部容器 joint 处于关闭阈值内才生成 episode。
- `minimal_plan_verified`：未执行 prerequisite leave-one-out 时为 `null`，并由 `minimal_plan_validation.status=not_executed` 和稳定 reason 明确说明，不能伪写为 `true`。
- 自动候选 rank、优先 source episode 和粗距离分桶。

写出前会执行 v3 强一致性校验，包括指定实例、grounding 计数、interaction/prerequisite 图、初始 articulation 镜像、门和容器状态读回、oracle 顺序、prefix 初始失败与终态成功、距离/可见性成功证据及 JSON Schema。可选字段中的无信息 `None` 会被移除；当前唯一有意保留的 `null` 是上述尚未执行的 `minimal_plan_verified`。

`scene_modifications.articulation_states` 会在 JsonEvalTaskSampler 恢复 object pose 之后、创建 task 之前按 joint name 应用，并执行 `mj_forward`。

### 输出文件与证据

并行精采输出：

```text
<output_dir>/
  benchmark.json
  valid_pairs.json
  rejected_pairs.json
  house_catalog.json
  fridge_slide_compartment_candidates.json
  failures.json
  collection_plan.json
  summary.json
  shards/shard_*/
```

开启 `--save_images` 时，每个有效 pair 的 `evidence/<case_id>/` 保存 closed/final RGB、segmentation、target mask 和各 oracle joint 的 trace。开启 `--save_plots` 时保存 joint/object 3D box 和失败诊断图。大规模采集建议先关闭图片和 plot，完成 JSON 后再对选定样本补充 evidence。

常见候选拒绝原因：

- `no_controlling_joint`：所有 hinge/slide 候选均未通过 reveal、binding、依赖或安全位姿验证；详细 joint 原因位于 `diagnostics.joint_failures`。
- `no_valid_source_start`：没有原 benchmark 起点同时满足“初始不可见”和“到交互位姿存在 GT path”。
- `cyclic_dependency`、`no_collision_free_interaction_pose`、`object_not_in_joint_compartment_box`、`object_not_bound_to_moving_compartment` 和 `force_drive_not_reached` 等保存在 joint-level diagnostics。
- `shard_process_failed`：并行子进程非零退出，由父进程写入最终 `failures.json`。

新的 fixed 模式不会再产生 house-level `incomplete_house_quota` 回滚；不足配额通过 `summary.json` 中的 `partial_collection_house_count` 和 `zero_sample_house_count` 表达。

### 并行实现与耗时

`collect_container_fine_parallel.py` 使用线程池调度子进程，真正的 MuJoCo 采集发生在独立 Python subprocess 中：

- house 在 shard 间互斥，不会被重复采集。
- 每个 shard 有独立 benchmark 输出、日志和 Matplotlib cache。
- 父进程按 `case_id` 合并 benchmark 和 valid pair。
- 任一 shard 非零退出都会进入最终 failure 统计。

4 worker 已在旧 complete-house 200 条测试中稳定运行：耗时约 `3847 s`，即 `1.07 h`。8/16 worker 尚未实测，单 GPU 上可能受到 EGL context、显存和渲染争用影响；正式大规模运行优先使用 4 worker，再用少量 house 测试 8 worker 吞吐。

按现有 200 条实测线性外推，2000 个成功样本的精采时间约为：1 worker `39-42 h`、2 worker `20-22 h`、4 worker `10.7-12 h`。8/16 worker 不建议在未做显存测试时直接用于全量采集。

### 当前数据与验证状态

现有 200 条结果位于：

```text
scripts/InteractiveNav/output/container_interaction_100house_test/
  benchmark_parallel_complete100/
  quality_report/
```

这批数据是在旧的 complete-house 语义下生成：从更大的 house pool 中保留 100 个完整 house，每个 house 两条。它不是新 fixed 语义重新运行的结果，但其单候选 GT path、oracle、visibility、force-slide 和 episode schema 与当前实现一致。

该批数据统计：

| 指标 | 数值 |
|---|---:|
| episode | 200 |
| house | 100 |
| Fridge / Dresser | 104 / 96 |
| 目标类别 | 20 |
| container asset | 10 |
| hinge pixel reveal | 102 |
| force slide visibility | 84 |
| slide object motion | 14 |
| GT path 平均 / 中位数 | 10.432 m / 7.331 m |
| GT path 最小 / 最大 | 0.081 m / 42.182 m |

原始 NavToObj benchmark 现有 GT path 扫描中，1913 条 path-found episode 的平均值为 `6.372 m`、中位数为 `5.799 m`。两者终点定义和地图分辨率配置不完全相同，因此该比较用于观察数据分布，不作为算法性能结论。

当前已完成的代码验证：

- dynamic plan 固定 100 house 时请求 200 条，并保留每个 house 的全部 fallback。
- fixed merge 会保留只有一条成功样本的 partial house。
- `samples_per_house=2` 生成 1424 个请求配额；`samples_per_house=3` 可生成 2000 个请求配额。
- 容器 benchmark、multi-oracle、articulation replay 和 fixed-plan 相关测试通过。

新的 fixed 管线尚未重新执行 GPU 全量精采。后续正式生成前应先运行 5-10 个固定 house，检查 `summary.json` 中的 complete/partial/zero house 数、GT path 分布和 shard 显存占用。

### 质量评估现状

`evaluate_container_interaction_dataset.py` 当前可以统计容器类别、目标类别、asset、距离分桶、reveal mode、candidate rank、GT path 和初始可见性，并生成 JSON/Markdown 报告。现有实现的 completeness quality gates 仍按旧 manifest slot 和每 house 两条设计；对新的 `collection_plan_v2` fixed partial-house 数据使用前，需要将 gate 改为按 `target_sample_count`、`partial_collection_house_count` 和 `zero_sample_house_count` 判断。新 fixed 运行期间应以采集器 `summary.json`、`valid_pairs.json` 和 `failures.json` 为主。

## 与交互式导航主线的关系

这个脚本目前提供的是交互导航 benchmark 构建和 oracle/probe 级能力：

- 为容器类交互提供候选目标发现。
- 为“交互前后可见性变化”提供可计算信号。
- 为多 joint 容器提供打开顺序依赖估计。
- 为门类通道交互提供同风格的 open/close 动作封装和第一视角验证。
- 为后续从感知 box/name 到 GT object id 的匹配接口预留基础数据结构。

后续如果要接入在线 planner，建议把当前脚本中的能力拆为三层：

- `scene_scan`：对象、容器、门、joint、AABB、可见性关系扫描。
- `interaction_oracle`：容器/门 open/close、joint dependency、可达/可见状态转移。
- `debug_render`：机器人站位、head/torso view control、RGB/seg/mask 输出。
