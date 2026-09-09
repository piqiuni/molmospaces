# MolmoSpaces 交互属性探索目标对照

本文把当前目标逐项映射到已经完成的产物、现有证据，以及仍待补足的部分。

## 1. 探索 MolmoSpaces 场景交互方式

目标内容：

- 通道属性如何操作：door open/close
- 容器属性如何操作：fridge / drawer / cabinet / microwave / oven
- 后续扩展：是否有灯光控制

当前产物：

- [explore_molmo_interactions.py](./explore_molmo_interactions.py:1)
  - `inspect-scene`
  - `set-articulation`
  - `action-schema`
  - `integration-recipe`
  - `env-check`
- [molmo_interaction_interfaces.md](./molmo_interaction_interfaces.md:1)
- [molmo_gt_workflow.md](./molmo_gt_workflow.md:1)
- [TODO.md](../../TODO.md:287)

当前结论：

- door：已有专用 `Door(...)` 数据视图接口，可直接读写 hinge joint
- container：已有通用 `MlSpacesArticulationObject` 接口，可直接读写 joint
- light：已有低层 `light_active` 与 lighting randomizer，但没有现成高层 task 主线

当前状态：

- 代码梳理：已完成
- 模板沉淀：已完成
- 真实场景运行验证：未完成

## 2.1 `nav_to_obj` 如何通过占据图 + 图搜索得到 GT path

当前产物：

- [explore_molmo_interactions.py](./explore_molmo_interactions.py:609)
  - `nav-gt`
- [molmo_interaction_interfaces.md](./molmo_interaction_interfaces.md:1)
- [molmo_gt_workflow.md](./molmo_gt_workflow.md:1)

证据：

- `NavToObjTaskSampler.init_scene()` 会生成 occupancy map
- `AStarPlanner` 基于 occupancy map 下采样建 graph
- `AStarPlannerPolicy` 输出 base waypoint action

当前状态：

- 代码链梳理：已完成
- 脚本入口：已完成
- 在真实 `mlspaces` 环境里跑出示例 path：未完成

## 2.2 将开关门嵌入占据图并比较路径

当前产物：

- [explore_molmo_interactions.py](./explore_molmo_interactions.py:636)
  - `door-path-study`
- live map 重建逻辑已写入同脚本

证据：

- 已明确现有 `AStarPlanner` 默认按 `model_path` 重建 map，不直接读取 live door state
- 已补 live `model + data` 重建 occupancy map 的脚本逻辑
- 已实现“按 baseline path 附近选门 -> 关门/开门 -> 重算 path”的脚本流程

当前状态：

- 逻辑设计：已完成
- 代码实现：已完成
- 在真实环境中筛出 candidate scene：未完成

## 2.3 固定 task，init 后关闭指定门，再计算 GT path

当前产物：

- `task-config-template`
- `benchmark-episode-template`
- 文档中关于 `NavToObjTaskConfig`、`DoorOpeningTaskConfig` 的字段说明

证据：

- 已明确固定起点字段：`robot_base_pose`
- 已明确固定目标字段：`pickup_obj_name` / `pickup_obj_candidates`
- 已明确 init 后的 door override 接口：`Door.set_joint_position(...)`

当前状态：

- 配置入口梳理：已完成
- 配置模板导出：已完成
- 用真实 episode 数据回填一份 frozen config / benchmark JSON：未完成

## 3.1 导航如何直接调用开关门动作

当前产物：

- `action-schema --mode door_oracle`
- `action-schema --mode door_planner`
- `integration-recipe --mode door_oracle_nav_loop`
- `integration-recipe --mode door_planner_handoff`

证据：

- 已区分 oracle 调用与 planner handoff 两条路径
- 已写明导航 loop 中如何触发 `Door.set_joint_position(...)`
- 已写明如何把 door opening 当作单独 task/policy handoff 接入

当前状态：

- 接口梳理：已完成
- 伪代码接线方案：已完成
- 在实际 nav loop 中完成一次真接线：未完成

## 3.2 导航如何直接调用开关 fridge / drawer 等动作

当前产物：

- `action-schema --mode container_oracle`
- `action-schema --mode container_planner`
- `integration-recipe --mode container_oracle_nav_loop`
- `integration-recipe --mode container_planner_handoff`

证据：

- 已区分 GT/oracle container state change 与 planner execution handoff
- 已写明通用 articulation object 的 joint 接口

当前状态：

- 接口梳理：已完成
- 伪代码接线方案：已完成
- 在实际循环中完成一次真接线：未完成

## 4. 明确梳理后代码接口

当前产物：

- [molmo_interaction_interfaces.md](./molmo_interaction_interfaces.md:1)
- [molmo_gt_workflow.md](./molmo_gt_workflow.md:1)
- [explore_molmo_interactions.py](./explore_molmo_interactions.py:1)

覆盖范围：

- GT path 接口
- door/container/light 状态接口
- task config 模板
- benchmark episode 模板
- action schema
- integration recipe

当前状态：

- 已完成

## 5. 基于目标完成情况，补充 TODO

当前产物：

- [TODO.md](../../TODO.md:287)

已补内容：

- 新增脚本入口列表
- 新增 GT / 交互接口总结
- 新增 step-level action 说明
- 新增配套说明文档入口

当前状态：

- 已完成

## 剩余缺口总览

当前真正还缺、且只能靠真实环境验证的内容：

1. 在合适 house 上跑通 `nav-gt`，拿到 live GT path
2. 在完整环境里运行 `door-path-study`
3. 用实跑结果筛出一组“开门提高效率 / 关门阻断路径”的 house
4. 用真实数值回填 frozen task config 或 benchmark episode JSON

当前已确认的真实环境 blocker：

1. 默认资源缓存目录不可写
   - `molmospaces_resources` 默认尝试写 `$HOME/.cache/molmo-spaces-resources/.lock`
   - 当前环境可通过 `/tmp` 可写 proxy cache 绕开
2. Linux headless 环境必须走 EGL
   - 需显式设置：
     - `MUJOCO_GL=egl`
     - `PYOPENGL_PLATFORM=egl`
3. 某些 episode 会在 `NavGoalSampler` 阶段失败
   - 这不是 scene-loading blocker，而是 candidate house / target object 筛选问题
   - 当前脚本已改为结构化返回 `nav_goal_sampling_error`

换句话说：

- 接口梳理、模板沉淀、工作流组织：已经基本完成
- `inspect-scene` 实跑：已验证
- `nav-gt` 实跑到 task/goal sampling：已验证
- 候选场景筛选与 `door-path-study`：尚未验证
