# MolmoSpaces GT Path 与交互控制工作流

本文把当前已经梳理出来的代码入口，按你的目标 `2.1 -> 3.2` 串成一条可执行工作流。

## 1. 目标对应关系

### 1.1 `2.1`：`nav_to_obj` 如何得到 GT path

核心结论：

- 现有 `nav_to_obj` 的 planner 路线就是 `occupancy map + graph search`
- 入口命令仍然是：

```bash
python scripts/datagen/run_pipeline.py \
  --task_type nav_to_obj \
  --policy planner \
  --robot rby1 \
  --house_inds 1 \
  --samples_per_house 1
```

关键代码：

- [run_pipeline.py](../../scripts/datagen/run_pipeline.py:139)
- [nav_task_sampler.py](../../molmo_spaces/tasks/nav_task_sampler.py:38)
- [astar_planner.py](../../molmo_spaces/planner/astar_planner.py:28)
- [astar_planner_policy.py](../../molmo_spaces/policy/solvers/navigation/astar_planner_policy.py:458)

当前推荐探索入口：

```bash
python scripts/InteractiveNav/explore_molmo_interactions.py nav-gt \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --target_types Apple
```

这个命令会输出：

- 采样后的 `robot_base_pose`
- 目标实例
- `nav_goal`
- GT path waypoint
- path length

### 1.2 `2.2`：把开关门嵌入占据图并重算路径

核心结论：

- 现有 `AStarPlanner` 默认按 `model_path` 重建 map，不会自动读取运行时 door state
- 所以要做“全开门路径 -> 关 N 个门路径 -> 全关门路径”，要对 live `model + data` 重新建图

当前推荐探索入口：

```bash
python scripts/InteractiveNav/explore_molmo_interactions.py door-path-study \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --target_types Apple \
  --close_doors_on_path 1 \
  --study_state closed
```

关键逻辑：

- 先采样一个 `nav_to_obj` episode
- 用当前 live 状态建 baseline occupancy map
- 计算 baseline GT path
- 根据 baseline path 选离路径最近的门
- 直接改 door joint state
- 再用 live 状态重建 occupancy map
- 重算 GT path

这一步用于找“开门提高效率 / 关门造成绕行或阻断”的 candidate scene。

### 1.3 `2.3`：如何固定 task，init 后关闭指定门，再算 GT path

推荐两层做法：

1. 固定 `task_config`
2. task init 后直接做 GT/oracle door override

固定 task 的字段见：

- [task_configs.py](../../molmo_spaces/configs/task_configs.py:156)
  - `NavToObjTaskConfig.robot_base_pose`
  - `NavToObjTaskConfig.pickup_obj_name`
  - `NavToObjTaskConfig.pickup_obj_candidates`

模板导出：

```bash
python scripts/InteractiveNav/explore_molmo_interactions.py task-config-template \
  --task-kind nav_to_obj
```

benchmark JSON 骨架：

```bash
python scripts/InteractiveNav/explore_molmo_interactions.py benchmark-episode-template \
  --task-kind nav_to_obj
```

task init 后关门：

```python
door = Door(door_body_name, env.current_data)
hinge_idx = door.get_hinge_joint_index()
door.set_joint_position(hinge_idx, 0.0)
```

然后重新建图并重算 GT path。

## 2. 通道属性、容器属性、灯光控制

### 2.1 通道属性：door

当前最直接接口：

```python
door = Door(door_body_name, env.current_data)
hinge_idx = door.get_hinge_joint_index()
joint_range = door.get_joint_range(hinge_idx)
door.set_joint_position(hinge_idx, target_position)
```

适用场景：

- GT study
- oracle interaction
- fixed-state benchmark 构造

### 2.2 容器属性：fridge / drawer / cabinet / microwave / oven

当前最直接接口：

```python
obj = env.object_managers[0].get_object_by_name(object_name)
joint_range = obj.get_joint_range(joint_index)
obj.set_joint_position(joint_index, target_position)
```

探索命令：

```bash
python scripts/InteractiveNav/explore_molmo_interactions.py inspect-scene --house_ind 1

python scripts/InteractiveNav/explore_molmo_interactions.py set-articulation \
  --house_ind 1 \
  --object-name <ARTICULATED_OBJECT_NAME> \
  --joint-index 0 \
  --open-fraction 0.0
```

### 2.3 灯光控制

当前结论：

- 有低层 light 控制
- 没有现成高层 task / benchmark 主线

低层接口：

```python
env.current_model.light_active[light_id] = 0  # off
env.current_model.light_active[light_id] = 1  # on
```

因此灯光目前更适合写进 future extension，而不是第一阶段主任务。

## 3. `3.1` / `3.2`：导航如何调用 open / close

### 3.1 如果你要“直接开启/关闭门等通道”

当前最现实的实现路径不是新造一个统一 semantic action，而是先选一种：

1. `oracle / GT action`
2. `planner execution`

door 的 oracle 接口：

```bash
python scripts/InteractiveNav/explore_molmo_interactions.py action-schema \
  --mode door_oracle

python scripts/InteractiveNav/explore_molmo_interactions.py integration-recipe \
  --mode door_oracle_nav_loop
```

door 的 planner 接口：

```bash
python scripts/InteractiveNav/explore_molmo_interactions.py action-schema \
  --mode door_planner

python scripts/InteractiveNav/explore_molmo_interactions.py integration-recipe \
  --mode door_planner_handoff
```

建议：

- 如果当前目标是验证“导航何时需要交互”，先用 `door_oracle`
- 如果当前目标是验证“执行层是否能真的开门”，再接 `door_planner`

### 3.2 如果你要“直接开启/关闭冰箱、抽屉等容器”

container 的 oracle 接口：

```bash
python scripts/InteractiveNav/explore_molmo_interactions.py action-schema \
  --mode container_oracle

python scripts/InteractiveNav/explore_molmo_interactions.py integration-recipe \
  --mode container_oracle_nav_loop
```

container 的 planner 接口：

```bash
python scripts/InteractiveNav/explore_molmo_interactions.py action-schema \
  --mode container_planner

python scripts/InteractiveNav/explore_molmo_interactions.py integration-recipe \
  --mode container_planner_handoff
```

建议同样是：

- 做 GT benchmark / 状态图研究：优先 `container_oracle`
- 做真实 step 执行：再切到 `container_planner`

## 4. 当前建议的最小实验流程

### A. 找场景

1. `inspect-scene`
2. 看 house 里有哪些 door / drawer / fridge
3. 记录 door 名称和容器名称

### B. 取 baseline nav path

1. `nav-gt`
2. 拿到固定 `robot_base_pose`、目标实例、GT path

### C. 研究关门后的路径变化

1. `door-path-study --close_doors_on_path 1`
2. 看 baseline / changed path length 差异
3. 如果需要，再扩到手工指定 `--door_names`

### D. 固化成可复现实验

1. `task-config-template`
2. `benchmark-episode-template`
3. 把实际采样到的 `robot_base_pose / pickup_obj_name / door_body_name` 填进去

### E. 选择执行层方案

1. 若当前只做交互图 / 状态图研究：用 `door_oracle` / `container_oracle`
2. 若当前要验证 step 执行链：用 `door_planner` / `container_planner`

## 5. 目前还缺什么

当前还没有真正完成的部分是：

- 在完整 `mlspaces` 环境里批量跑多个 house
- 真正筛出“开门提高效率”的 candidate scene
- 把这些结果固化成一批 benchmark JSON episode

但从接口和代码梳理角度，当前已经具备：

- scene inspection
- GT path extraction
- live door override + replan
- container articulation override
- task config template
- action schema template
- benchmark episode template
