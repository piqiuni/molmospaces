# MolmoSpaces 交互属性与接口梳理

本文聚焦 `scripts/InteractiveNav` 当前需要用到的 MolmoSpaces 接口，目标是回答四个问题：

1. `nav_to_obj` 的 GT path 现在是怎么来的
2. 固定起点/目标后，如何切换 door 状态并重算路径
3. task step 层面的 action dict 长什么样
4. door、container、light 目前分别有哪些可直接调用的接口

## 1. `nav_to_obj` GT path 链路

入口命令：

```bash
python scripts/datagen/run_pipeline.py \
  --task_type nav_to_obj \
  --policy planner \
  --robot rby1 \
  --house_inds 1 \
  --samples_per_house 1
```

主要代码链：

- [run_pipeline.py](../../scripts/datagen/run_pipeline.py:139)
  - `task_type == "nav_to_obj"` 时，使用 `NavToObjBaseConfig + AStarNavToObjPolicyConfig`
- [nav_task_sampler.py](../../molmo_spaces/tasks/nav_task_sampler.py:38)
  - `init_scene()` 会先生成 occupancy map，并缓存到 `self._cached_thormap`
  - `_sample_task()` 会把这个 map 挂到 `task.occupancy_map`
- [astar_planner.py](../../molmo_spaces/planner/astar_planner.py:28)
  - 把 occupancy map 下采样成 grid，再建 `networkx` graph 做图搜索
- [astar_planner_policy.py](../../molmo_spaces/policy/solvers/navigation/astar_planner_policy.py:458)
  - 每 step 返回下一个 base waypoint

当前关键限制：

- [AStarPlanner.map](../../molmo_spaces/planner/astar_planner.py:44) 默认按 `model_path` 重新生成 map
- 它不是直接消费运行时 `MjData`
- 这意味着 door joint 在 episode 里被改了以后，现成 `AStarPlanner` 不会自动反映该变化

因此，如果要做：

- 固定起点
- 固定目标
- 全开门路径
- 关 N 个门后的路径
- 全关门路径

更稳的方式不是直接复用现成 planner，而是对当前 live `model + data` 重新建图，再做图搜索。

## 2. 固定 episode 的 config 入口

### 2.1 固定 `nav_to_obj`

见 [task_configs.py](../../molmo_spaces/configs/task_configs.py:156)：

- `NavToObjTaskConfig.robot_base_pose`
- `NavToObjTaskConfig.pickup_obj_name`
- `NavToObjTaskConfig.pickup_obj_candidates`

含义：

- `robot_base_pose`：固定导航起点
- `pickup_obj_name`：固定目标实例
- `pickup_obj_candidates`：固定同类候选集，避免 sampler 每次变

### 2.2 固定 door opening episode

见 [task_configs.py](../../molmo_spaces/configs/task_configs.py:127)：

- `DoorOpeningTaskConfig.door_body_name`
- `DoorOpeningTaskConfig.robot_base_pose`
- `DoorOpeningTaskConfig.articulated_joint_range`
- `DoorOpeningTaskConfig.articulated_joint_reset_state`

这套配置适合做：

- “指定哪一扇门”
- “指定机器人起点”
- “指定门初始是关还是开”

### 2.3 固定通用 container open/close episode

见 [task_configs.py](../../molmo_spaces/configs/task_configs.py:100)：

- `OpeningTaskConfig.pickup_obj_name`
- `OpeningTaskConfig.joint_name`
- `OpeningTaskConfig.joint_index`
- `OpeningTaskConfig.joint_start_position`
- `OpeningTaskConfig.joint_goal_position`

这套配置适合 drawer / cabinet / fridge / microwave / oven / dishwasher 等通用 articulation object。

## 3. 运行时直接改状态的接口

### 3.1 通道属性：door

door 的专用接口在 [data_views.py](../../molmo_spaces/env/data_views.py:605)：

- `Door(name, env.current_data)`
- `get_hinge_joint_index()`
- `get_joint_range(i)`
- `get_joint_position(i)`
- `set_joint_position(i, position)`

最常用模式：

```python
door = Door(door_name, env.current_data)
hinge_idx = door.get_hinge_joint_index()
joint_range = door.get_joint_range(hinge_idx)
door.set_joint_position(hinge_idx, 0.0)  # close
```

注意：

- `Door.set_joint_position()` 内部会 `mj_forward`
- 因此更适合做 GT/oracle 状态切换，而不是自己手写低层控制器

### 3.2 容器属性：drawer / cabinet / fridge / microwave / oven

通用接口来自 [MlSpacesArticulationObject](../../molmo_spaces/env/data_views.py:500)：

- `ObjectManager.get_object_by_name(name)`
- `joint_names`
- `get_joint_range(i)`
- `get_joint_position(i)`
- `set_joint_position(i, position)`

最常用模式：

```python
obj = env.object_managers[0].get_object_by_name(object_name)
joint_range = obj.get_joint_range(joint_index)
target = joint_range[0] + (joint_range[1] - joint_range[0]) * open_fraction
obj.set_joint_position(joint_index, target)
```

目前 drawer/cabinet 的 category 封装主要在：

- [cabinet.py](../../molmo_spaces/env/arena/cabinet.py:1)
- [drawer.py](../../molmo_spaces/env/arena/drawer.py:1)

这些类更多负责物理参数随机化，不是额外的高层 action API。

### 3.3 灯光 / 开关类

当前仓库里“灯光”已有低层接口，但没有成型任务：

- [mj_extensions.py](../../molmo_spaces/env/mj_extensions.py:10)
  - 暴露了 `light_names / light_name2id / light_id2name`
- [lighting.py](../../molmo_spaces/env/arena/randomization/lighting.py:377)
  - `set_active(light_id, value)`
  - `get_active(light_id)`

如果只从 MuJoCo 低层改，可以直接写：

```python
env.current_model.light_active[light_id] = 0  # off
env.current_model.light_active[light_id] = 1  # on
```

但目前没有：

- `LightSwitchTask`
- `nav_to_obj + light toggle` 的现成 benchmark/task sampler
- 与 planner 对齐的“light changes observability”高层状态机

所以灯光目前更适合作为 future extension。

## 4. task step 层 action dict

`task.step()` 的入口见 [task.py](../../molmo_spaces/tasks/task.py:252)。

它接受：

- 单环境：一个 `dict`
- 多环境：`list[dict]`

特殊字段：

- `done: True/False`
  - [task.py](../../molmo_spaces/tasks/task.py:323) 会先把它剥掉，再记录 `_done_action_received`

也就是说，实际 action payload 的核心是各 move group 的控制量，例如：

- `base`
- `left_arm`
- `right_arm`
- `left_gripper`
- `right_gripper`
- `head`
- `torso`

具体格式由 robot move group 决定。

### 4.1 导航 action

见 [astar_planner_policy.py](../../molmo_spaces/policy/solvers/navigation/astar_planner_policy.py:482)：

```python
{"done": False, "base": waypoint}
```

结束时：

```python
{**robot_view.get_noop_ctrl_dict(["base"]), "done": True}
```

因此对导航来说，最小动作接口就是：

- `base`
- `done`

### 4.2 door opening action

见 [opening_solver.py](../../molmo_spaces/policy/solvers/opening_solver.py:393)。

door opening solver 会按 phase 产出组合动作，可能包含：

- `base`
- `left_arm` / `right_arm`
- `left_gripper` / `right_gripper`
- `head`
- `done`

这说明目前 door 的“任务执行层动作”不是单个 `open_door(door_id)` primitive，而是一个分 phase 的 policy。

### 4.3 container open/close action

drawer / cabinet / fridge 等通用 container 走的是：

- [OpeningTask](../../molmo_spaces/tasks/opening_tasks.py:18)
- [OpenClosePlannerPolicy](../../molmo_spaces/policy/solvers/object_manipulation/open_close_planner_policy.py:18)

它同样不是一个单独的：

- `{"open": object_name}`

而是 planner policy 生成一系列 arm/gripper 轨迹控制。

所以如果要“供导航调用，直接开启/关闭容器”，当前更现实的办法是两种：

1. GT/oracle 模式：
   - 直接 `set_joint_position()`
2. task/policy 模式：
   - 构造 `OpeningTaskConfig`
   - 交给 `OpenClosePlannerPolicy`

## 5. 当前新增探索工具

见 [explore_molmo_interactions.py](./explore_molmo_interactions.py:1)。

推荐命令：

```bash
conda activate mlspaces

python scripts/InteractiveNav/explore_molmo_interactions.py inspect-scene --house_ind 1

python scripts/InteractiveNav/explore_molmo_interactions.py nav-gt \
  --house_ind 1 \
  --target_types Apple

python scripts/InteractiveNav/explore_molmo_interactions.py door-path-study \
  --house_ind 1 \
  --target_types Apple \
  --close_doors_on_path 1 \
  --study_state closed

python scripts/InteractiveNav/explore_molmo_interactions.py task-config-template \
  --task-kind open_close

python scripts/InteractiveNav/explore_molmo_interactions.py action-schema \
  --mode door_oracle

python scripts/InteractiveNav/explore_molmo_interactions.py benchmark-episode-template \
  --task-kind nav_to_obj

python scripts/InteractiveNav/explore_molmo_interactions.py integration-recipe \
  --mode container_planner_handoff
```

如果没激活环境，脚本现在会明确提示先 `conda activate mlspaces`。

补充说明：

- `task-config-template`
  - 用来导出固定 `nav_to_obj` / `door_opening` / `open_close` episode 的字段草稿
  - 更适合回答“2.3 如何配置 task”
- `action-schema`
  - 用来导出导航、door/container oracle、door/container planner 五类 action 模板
  - 更适合回答“3.1/3.2 导航如何调用 open/close”
- `benchmark-episode-template`
  - 用来导出 `benchmark_schema.EpisodeSpec` 级别的 JSON 骨架
  - 更适合把“固定起点 + 固定目标 + 固定门状态”的实验直接收成可评测 episode
- `integration-recipe`
  - 用来导出导航层如何调用 oracle / planner 交互的伪代码
  - 更适合回答“3.1 / 3.2 这一动作到底该怎么接入导航循环”
