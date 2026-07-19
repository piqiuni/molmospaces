# 交互导航阶段分析与 TODO

最后更新：2026-07-17

## 1. 当前课题定位

本项目当前的交互导航方向，不是做一个更强的 manipulation 控制器，也不是在现有语义图上简单增加一层交互标签，而是把**交互性作为导航空间中的一等变量**来建模。

更具体地说，论文立意应当建立在以下判断上：

1. 传统导航、ObjectNav 和大多数语义图导航方法，默认环境的可达性与拓扑关系基本静态。
2. 真实室内环境中，门、柜门、抽屉、冰箱等可交互对象会主动改变**可达性（reachability）**、**可见性（observability）**和**路径代价（cost）**。
3. 因而交互式导航的核心不是“学会更复杂的开门控制”，而是让导航系统显式理解：**环境拓扑和最优路径会随交互动作而改变**。

基于这一立意，本项目当前的第一阶段论文应聚焦于：

1. 定义一种以 **obj-goal navigation** 与 **point-goal navigation** 为核心的 interaction-aware navigation 任务设定。
2. 构建能够表达交互状态、交互代价和状态转移的导航表示，而不是仅做静态语义增强。
3. 让规划过程显式判断“是否值得交互、交互后哪些连通性会被解锁、交互后是否需要重规划”。
4. 在执行层优先复用 oracle interaction 或已有 open/close 能力，先验证交互表示与规划闭环。

当前阶段性输出目标明确为：

- **下个月完成一篇 AAAI 投稿版本**
- 论文实验主体先围绕 `obj-goal` 与 `point-goal` 的交互导航闭环展开

因此，本项目当前更准确的核心主张是：

- **交互式导航首先是一个“状态变化环境中的导航问题”，其次才是一个操作问题。**
- **我们关注的是交互如何改变导航图，而不是单独优化底层交互控制器。**

相应地，本项目的方法叙事不应停留在：

- 为语义图增加可交互对象标签
- 在场景图上额外挂一层 interaction layer

而应提升为：

- 构建 **state-conditioned interaction graph / action-dependent reachability graph**
- 显式表示 `navigation edges`、`interaction-triggered edges` 与交互后的状态转移
- 在图搜索或规划中统一考虑导航代价、交互代价与交互收益

这类交互收益至少包括：

- `reachability gain`：交互后原本不可达的目标变得可达
- `efficiency gain`：交互后可走更短路径，而不是被迫绕行
- `observability gain`：交互后暴露新的区域、门后空间或目标线索

当前最合理的第一阶段最小闭环，仍然是 door-centric，并围绕“交互改变导航图”来验证：

- `obj-goal navigation`
- `point-goal navigation`
- door / open-close 任务能力
- state-conditioned interaction graph / 交互感知地图
- oracle 开门后继续导航

也就是先证明：**交互会改变可达性、可见性和路径代价，因此静态导航图不足以描述真实任务。**

---

## 2. 阶段分析

## 2.1 总体判断

基于当前仓库中的规划文档、讨论记录以及已有代码结构，项目已经不再是“纯想法阶段”，但也还没有进入“完整 benchmark 稳定评测阶段”。

更准确地说，当前处在：

**阶段 A：方向已明确 + 双主线基础已出现 + 正在收敛论文所需的最小可验证闭环**

这意味着目前最重要的工作，不是只做单线实验，而是并行收紧两条主线，并让它们共同服务论文产出：

1. **MolmoSpaces 主线**：基于 GT / oracle 先构建 obj-goal / point-goal 交互导航任务、表示、规划与 benchmark 叙事。
2. **ROS 模块化主线**：基于 detector-only 的模块化感知与交互图方法，形成一个可独立对照的工程基线。

两条主线最终都要回答同一件事：

`感知/交互属性 -> 交互图/地图状态 -> 导航规划判断需要交互 -> 执行交互 -> 状态更新 -> 继续导航`

---

## 2.2 已经具备的基础

以下内容是根据当前仓库代码结构整理的“已具备基础”，后续可以作为阶段边界判断依据。

### A. MolmoSpaces 原生任务与评测基础已经在仓库中

- 已有 `nav_to_obj` 相关任务与采样代码：
  - `molmo_spaces/tasks/nav_task.py`
  - `molmo_spaces/tasks/nav_task_sampler.py`
- 已有 opening / open-close 相关任务与采样代码：
  - `molmo_spaces/tasks/opening_tasks.py`
  - `molmo_spaces/tasks/opening_task_samplers.py`
- 已有 opening solver 与 benchmark / evaluation 框架：
  - `molmo_spaces/policy/solvers/opening_solver.py`
  - `molmo_spaces/evaluation/benchmark_schema.py`
  - `molmo_spaces/evaluation/eval_main.py`
- 已有场景地图工具：
  - `molmo_spaces/utils/scene_maps.py`

结论：**任务侧与评测侧已经有可复用基础，不需要从零造一个交互导航平台。**

### B. ROS 侧的 Python 语义交互图管线已经初步搭好

`Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/` 已经不是空 scaffold，而是具备明确分层：

- `object_detection_node.py`：目标检测/实例观测入口
- `room_attribute_node.py`：房间属性推理
- `semantic_mapping_node.py`：语义地图与增量图维护
- `interaction_graph_store.py`：交互图存储与更新
- `interaction_graph_viz.py`：图可视化
- `semantic_map_store.py`：地图状态落盘/维护

同时已经有：

- GT replay 能力：`semantic_mapping_gt_replay.py`
- 可视化/离线测试入口：`object_detection_visual_test.py`
- 图相关单测：`tests/test_interaction_graph_store.py`
- 配置文件：`config/default.yaml`

结论：**“交互语义层”已经开始代码化，不再只是研究设想。**

### C. 当前更像是在验证“交互图是否真能服务导航”

从现有工作重心看，难点已经不只是“能否检测到门”，而是：

1. 检测/GT信息能否稳定进入统一图结构。
2. 图中的 portal / room / object / navigation hint 能否支撑规划。
3. 图状态是否能随着交互与观测持续更新。
4. 导航模块是否真正消费这些交互信息，而不是仅做展示。

结论：**接下来的主线应该从“建图”转向“建图如何改变规划结果”。**

---

## 2.3 当前阶段的核心缺口

### 关键缺口 1：双主线闭环定义还需要进一步收紧

虽然方向上已经多次确认“先做 door-centric 最小闭环”，但从论文产出角度，第一阶段不能只是一条 demo 线，而要形成：

- 一条 **MolmoSpaces 交互导航基座线**
- 一条 **ROS 模块化 detector-only 基线线**

建议第一阶段优先回答两个对应问题：

1. **在 MolmoSpaces 中，当 obj-goal 或 point-goal 被关闭门阻挡时，系统能否利用 GT 交互信息完成“先交互再导航”的任务闭环？**
2. **在 ROS 模块化方法中，仅依赖 detector-only 输入，系统能否构建出足以支持交互导航判断的模块化感知与交互图基线？**

如果这两个问题还没有形成稳定结果，先不要过早扩展到：

- 更复杂的交互类型分类
- LLM 全流程决策
- 大规模 benchmark 自动生成

### 关键缺口 2：交互图与规划器之间的接口仍然可能不够闭合

当前图结构、navigation hints、portal 表达已经在长出来，但仍需明确：

1. 规划器拿到的最小输入是什么？
2. “需要交互”是图上可搜索状态，还是仅靠规则触发？
3. 开门后哪些边/可达区域会被更新？
4. 更新后是否会触发重规划？

如果这里定义不清，图很容易退化为“辅助可视化层”，而不是规划核心。

### 关键缺口 3：论文叙事与工程验证之间还缺一个中间层

目前已有研究 framing，也有代码雏形，但中间还缺一个“可以稳定反复验证”的门控实验集。

建议先建立一个小规模、人工可检查的验证集合，并明确区分两类用途：

1. **MolmoSpaces 论文主线验证集**：证明交互会改变 reachability / visibility / cost。
2. **ROS 模块化验证集**：证明 detector-only 管线可以产出有效交互表示并服务下游规划。

例如：

- 少量多房间场景
- 明确存在 door-blocked 路径
- 可以对比 pure nav / oracle open / interactive planner

先把验证集跑顺，再扩大到 benchmark 生成。

---

## 3. 当前建议阶段划分

## Phase 0：基础资产与入口确认

目标：确认本项目现有入口足够支撑交互导航实验，不新增无必要基础设施。

状态判断：

- [x] 已有 `nav_to_obj`、opening task、evaluation 框架
- [x] 已有 ROS 语义交互图原型
- [x] 已明确第一阶段不是单入口，而是 MolmoSpaces + ROS 双主线并行
- [ ] 仍需明确两条主线各自的最小成功标准与对齐接口

## Phase 1：第一阶段双主线闭环（下个月 AAAI 投稿）

目标：同时形成论文主线所需的任务闭环，以及模块化方法所需的工程基线，并输出下个月的 AAAI 投稿版本。

阶段完成标准建议：

- [ ] MolmoSpaces 侧能构造 `obj-goal` / `point-goal` 下“纯导航不可达 / 开门后可达”的 episode
- [ ] MolmoSpaces 侧能用 GT / oracle 跑通 door-centric 的 obj-goal / point-goal 交互导航基座
- [ ] ROS 侧能以 detector-only 输入稳定产出交互对象与交互图表示
- [ ] 能在图或地图层表达 `door -> requires interaction -> unlocks connectivity`
- [ ] 至少有一条规划链能输出“先交互再导航”的行为序列
- [ ] 交互后能更新状态并继续导航
- [ ] 至少有一组可重复的可视化、日志或案例结果，可直接服务论文材料
- [ ] 完成 AAAI 投稿版本所需的实验主表、消融表与案例图

## Phase 2：交互图接口稳定化

目标：把“能跑”变成“结构稳定、可扩展”。

阶段完成标准建议：

- [ ] 明确统一 observation schema（GT / detector 共用）
- [ ] 明确 unified graph schema 中哪些字段由规划器直接消费
- [ ] 明确 portal / room / object / support / container 的职责边界
- [ ] 明确交互状态更新机制（open / closed / unknown）
- [ ] 明确 navigation hints 与真正 planner state 的关系
- [ ] 明确 MolmoSpaces 任务侧接口与 ROS 模块化图接口之间如何对齐

## Phase 3：实验设计与 benchmark 最小版

目标：形成论文早期可用的定量对比。

阶段完成标准建议：

- [ ] 构建小规模 door-centric benchmark 子集
- [ ] 建立 `pure nav` baseline
- [ ] 建立 `nav + oracle open` baseline
- [ ] 建立 `interactive graph planner` baseline
- [ ] 建立 `ROS modular detector-only` baseline
- [ ] 输出首版主指标：SR / SPL / Interaction Success Rate / Interaction Precision / Total Cost
- [ ] 将 reachability / visibility / enablement 作为 benchmark 分层与 Interaction Success 的判定依据，而不是主表中的独立指标

## Phase 4：第二阶段语音交互导航

目标：在第一阶段交互导航闭环稳定后，引入语音交互，使系统能够理解语气、歧义和补充条件，并在必要时通过多轮问答完成导航。

阶段完成标准建议：

- [ ] 设计语音输入到导航目标/约束的表示接口
- [ ] 引入语气理解能力，例如急迫、礼貌、犹豫、强调
- [ ] 支持歧义目标澄清，例如“去那个房间”或“去我刚才说的地方”
- [ ] 支持必要时多轮问答后再执行导航
- [ ] 分析语音交互如何与交互图、状态更新和规划决策结合

## Phase 5：从门推广到更一般的交互对象

目标：在第一阶段结论成立后，再扩展对象类型和交互类型。

建议按“交互改变什么”来组织，而不是只按对象名组织。

### A. 通道属性（改变可达性 / 路径连通）

- [ ] hinge door
- [ ] sliding door
- [ ] gate / barrier
- [ ] 可移动障碍物（能推开、挪开、移走的 blocking object）

### B. 容器属性（改变可见性 / 可取性 / 内部可访问性）

- [ ] fridge
- [ ] cabinet
- [ ] drawer
- [ ] microwave / oven / box / lid-like container

说明：在当前室内家居场景下，一级主分类先聚焦在“通道属性”和“容器属性”。设备、开关、照明、窗帘等更适合作为后续扩展，而不是第一阶段主线。

### C. 后续方法扩展

- [ ] 更细交互类型（push / pull / slide / press）
- [ ] 交互代价建模
- [ ] 多步交互链规划
- [ ] 通道属性与容器属性的统一表示
- [ ] 设备与开关类交互作为 future extension 单独扩展
- [ ] 灯光、窗帘、电子解锁等感知辅助或功能控制交互的建模

## Phase 6：数据采集与轨迹构建

目标：在模块化方法跑通后，开始形成可训练、可复现的数据基础。

阶段完成标准建议：

- [ ] 搭建交互导航数据采集机制
- [ ] 明确优先采集真实执行轨迹还是先使用 GT / oracle 轨迹
- [ ] 形成 detector 输入、交互状态、规划决策、执行结果的统一记录格式
- [ ] 能导出用于后续 agent / end-to-end 方法训练的数据样本

### Phase 6 当前 mixed 数据推进状态（2026-07-15）

- [x] 从 `container_rough_catalog_v1` 二次生产 crossing rough：任一全开 GT path 穿过 interactive door 即保留；关闭门阻断目标且门前几何提示仍可达只作为 `mixed_required_verified` 子集标签，不再作为 rough 输入门槛。门前提示不视为 manipulation-validated 操作姿态。
- [x] mixed 精采集直接输出 `interactive_nav_v3`，不经过 V2 中间数据。
- [x] mixed 生产校验覆盖真实门/容器 joint readback、初始失败、开门恢复、目标可见性解锁、终态 NavToObj 成功和 leave-one-interaction-out 最小性。
- [x] 完成 10 条 mixed V3 小样本端到端 smoke。
- [x] 完成 crossing/required 拆分后的全量扫描：712 house、2603 pair、失败 0，得到 1609 个 crossing pair、覆盖 456 house；其中 917 个/262 house 为 required 子集，另有 692 个 crossing-only pair。真正无 crossing pair 的 house 为 256 个。旧 443-house 预筛仅覆盖其中 369 个 crossing house，并漏掉 87 个 crossing house，已明确禁止用于输入筛选。
- [x] 完成 crossing-only rough 双状态俯视诊断脚本与 24-house 分层图册：显示全开 rough GT path、单门关闭后的重规划路径、交互对象和全部 scene object AABB。
- [x] 完成代表性 mixed V3 场景的五步 GT storyboard：自动选择单关键门 + Fridge 样本，按起点、门前关/开、冰箱前关/开独立重放机器人 pathpoint、交互真值与右肩后上方视角，作为后续连续 GT 视频插值入口。
- [ ] 对 crossing-only 的闭门可达性做独立重放稳定性复核：当前 24-house 图册发现 house 351/candidate 427 的 catalog/replay 不一致；同时存在较多闭门前后几何路径长度近零或为负的边界样本，需要在正式配额前确定重扫、最小 detour 或路径区域裕量规则。
- [ ] 基于 full rough 分布确定正式 mixed benchmark 的场景、目标类别、路径长度和交互链配额。

## Phase 7：Agent 架构与端到端基线

目标：在模块化方法和数据基础稳定后，进一步搭建 agent 化与端到端基线。

阶段完成标准建议：

- [ ] 定义 Agent 架构中的感知、记忆、规划、交互执行接口
- [ ] 建立基于交互图的 Agent baseline
- [ ] 建立端到端架构 baseline，作为与模块化方法的对照
- [ ] 对比模块化方法与端到端方法在成功率、鲁棒性、可解释性上的差异

---

## 4. TODO 清单

## 4.1 最高优先级 TODO

- [ ] 明确第一阶段论文产出结构：MolmoSpaces 主线负责任务/benchmark/GT 闭环，ROS 主线负责模块化 detector-only 基线
- [ ] 明确第一阶段任务定义：以 `door-blocked obj-goal` 与 `door-blocked point-goal` 为核心主任务
- [ ] 明确 MolmoSpaces 主线的最小成功标准
- [ ] 明确 ROS 模块化主线的最小成功标准
- [ ] 明确两条主线的公共表示接口
- [ ] 明确 MolmoSpaces 侧交互信息来源：GT / oracle 为主
- [ ] 明确 ROS 侧交互信息来源：detector-only 为主
- [ ] 明确“需要交互”的 planner 判定条件
- [x] 完成 GT 快速版开门状态回写与 OCC 更新：首次有效关节值作为闭合参考，门完全打开后由 semantic 层持续清空闭合门整体 AABB，发布 planning OCC 与门洞增量更新，并由 move_base global costmap 实际消费
- [ ] 固定一组 `obj-goal` / `point-goal` 的最小验证场景与 episode，不先追求大规模
- [ ] 明确下个月 AAAI 投稿所需的最小实验包与文档产物

## 4.2 建图与表征 TODO

- [ ] 梳理 `interaction_graph_store.py` 当前已经表达的节点、边、状态与限制
- [ ] 明确 door 在统一图中是否始终作为 `portal` 使用
- [ ] 明确 room connectivity 是显式边还是由规则动态生成
- [ ] 定义交互状态最小集合：`unknown / closed / open`，是否需要 `locked`
- [ ] 定义第一阶段必须保留的交互属性字段，避免 schema 过重
- [ ] 明确 navigation hints 是调试视图、过渡接口，还是 planner 正式输入
- [ ] 建立更通用的交互属性分类框架：通道属性 / 容器属性
- [ ] 将设备与开关类交互保留为后续拓展，而不是第一阶段一级分类

## 4.3 感知与状态来源 TODO

- [ ] 明确 MolmoSpaces 侧 GT 字段到统一 observation 的映射表
- [ ] 明确 ROS detector-only 管线需要输出哪些最小交互字段
- [ ] 明确 door 检测成功之外还缺哪些交互属性字段
- [ ] 明确 door state 是从关节值直接读，还是从图像/几何间接估计
- [x] GT 快速版门状态：读取当前 `joint_infos`，过滤 handle joint，以首次有效门板关节值作为 `q_closed`，按 joint range 计算开合比例；双开门要求全部门板达到 open 阈值
- [ ] 真实场景门关节提取：在没有 MuJoCo joint GT 时，从多帧门板检测/跟踪与几何变化中估计 hinge/slide 类型、转轴或滑动轴、闭合参考位姿和开合比例；置信度不足时保持 `unknown`，不能直接清空 OCC
- [ ] 明确 room_id / connectivity 的可信来源

## 4.4 已梳理的 MolmoSpaces GT / 交互接口

当前已新增探索脚本：

- `scripts/InteractiveNav/explore_molmo_interactions.py`
  - `inspect-scene`：枚举场景中的 door / cabinet / drawer / fridge / microwave 等 articulation，以及 lights
  - `nav-gt`：在 `nav_to_obj` 采样结果上，读取固定起点、目标与 GT path
  - `door-path-study`：固定起终点后，切换指定 door 的 open / closed 状态并重算路径
  - `set-articulation`：对通用 articulation object 直接设置 joint open fraction
  - `task-config-template`：导出固定 `nav_to_obj` / `door_opening` / `open_close` episode 的配置草稿
  - `action-schema`：导出 navigation / door / container 的 oracle 或 planner action 模板
  - `benchmark-episode-template`：导出 `benchmark_schema.EpisodeSpec` 级别的 JSON 骨架
  - `integration-recipe`：导出导航层如何调用 oracle / planner 交互的接线伪代码
  - `env-check`：检查当前环境是否具备实跑 scene 的基本条件

当前已新增配套说明：

- `scripts/InteractiveNav/molmo_interaction_interfaces.md`
  - 汇总 task config、GT 接口、step action、灯光控制现状
- `scripts/InteractiveNav/molmo_gt_workflow.md`
  - 按 `2.1 -> 3.2` 目标串成可执行工作流
- `scripts/InteractiveNav/molmo_objective_status.md`
  - 按目标编号逐项整理当前证据、完成项与剩余缺口
- `scripts/InteractiveNav/molmo_runtime_validation.md`
  - 记录真实在 `mlspaces` 环境中运行时遇到的 cache / network blocker

当前已确认的环境侧 blocker：

- 真实 scene-loading 需要可写资源缓存目录：
  - 推荐 `MLSPACES_CACHE_DIR=/tmp/molmo-spaces-resources`
- 若本机已有 `~/.cache/molmo-spaces-resources`，可在 `/tmp` 建一个 proxy cache 根目录，并把现有 `robots / scenes / objects / grasps / benchmarks / test_data` 软链进去
- Linux headless 环境建议显式设置：
  - `MUJOCO_GL=egl`
  - `PYOPENGL_PLATFORM=egl`
- 当前已真实验证：
  - `inspect-scene` 可在 `train_10_ceiling.xml` 上成功输出 4 个 door、10 个 articulated objects、1 个 light
  - `nav-gt` 可真实跑到 `scene loading -> occupancy map -> nav_to_obj task sample -> target selection`
- 当前剩余真实问题已从“scene 无法加载”收敛为：
  - 某些 house 会在 `NavGoalSampler` 上失败，拿不到 nav goal
  - `door-path-study` 仍需在更合适的 house 上继续筛选

当前确认的 GT path 代码链路：

- `scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner`
- `NavToObjTaskSampler.init_scene()`：为 house 生成 occupancy map，并存到 `task.occupancy_map`
- `AStarNavToObjPolicy / AStarPlanner`：基于 occupancy map + graph search 生成 waypoint path
- 需要注意：现有 `AStarPlanner` 默认按 `model_path` 重建 map，不直接消费运行时 door state
- 因此若要做“固定起终点 + 改门状态 + 重算 GT path”，应优先使用 live model/data 重新建图，而不是只调用现成 policy

当前确认的固定 episode 入口：

- 固定导航起点：`NavToObjTaskConfig.robot_base_pose`
- 固定导航目标实例：`NavToObjTaskConfig.pickup_obj_name`
- 固定目标候选集：`NavToObjTaskConfig.pickup_obj_candidates`
- 固定 door episode：`DoorOpeningTaskConfig.door_body_name`、`robot_base_pose`、`articulated_joint_range`、`articulated_joint_reset_state`
- 固定通用 container/open-close episode：`OpeningTaskConfig.joint_name`、`joint_index`、`joint_start_position`

当前确认的状态修改接口：

- 通道属性（door）：
  - `Door(name, env.current_data)`
  - `get_hinge_joint_index()`
  - `get_joint_range(i)`
  - `set_joint_position(i, position)`
- 容器属性（drawer / cabinet / fridge / microwave / oven / dishwasher 等）：
  - `ObjectManager.get_object_by_name(name)` 返回 `MlSpacesArticulationObject`
  - `joint_names / get_joint_range(i) / get_joint_position(i) / set_joint_position(i, position)`
- door 打开任务和通用 open/close 任务是两条独立 task 线：
  - `DoorOpeningTask / DoorOpeningTaskSampler / opening_solver.py`
  - `OpeningTask / OpenTaskSampler / open_close_planner_policy.py`

当前确认的 step-level action 接口：

- `task.step(action)` 接受单环境 `dict` 或多环境 `list[dict]`
- `done` 是保留字段，`task.step()` 会先剥离，再标记 episode 结束
- 导航 policy 输出：
  - `{\"base\": waypoint, \"done\": False}`
  - 结束时为 `{\"done\": True, ...base noop...}`
- 若只想让导航层“直接开关门/容器”做 GT study，当前最自然的接口不是单独 task action，而是：
  - door：`Door(...).set_joint_position(...)`
  - container：`MlSpacesArticulationObject.set_joint_position(...)`
- door opening solver 输出：
  - 分 phase 生成 `base / arm / gripper / head / done` 组合动作
- 通用 container open/close planner：
  - 通过 `OpeningTask + OpenClosePlannerPolicy` 执行，不是单独的“open drawer” primitive API

关于灯光 / 开关类交互的当前判断：

- 代码里已有 light 元数据和随机化器：
  - `molmo_spaces/env/mj_extensions.py`
  - `molmo_spaces/env/arena/randomization/lighting.py`
- 当前更像“场景随机化 / 低层属性控制”，不是完整任务接口
- 低层可直接读写 `model.light_active[light_id]`
- 目前没有与 `nav_to_obj` / `OpeningTask` 对齐的高层 `light-switch task`，因此应先视为 future extension，而不是第一阶段主线
- [ ] 为后续 container / movable obstacle 预留观测字段
- [ ] 为 future extension 中的 switch / light / curtain 类交互预留扩展字段

## 4.4 规划与执行 TODO

- [ ] 设计第一阶段 planner 输出格式：导航动作、交互动作、重规划触发
- [ ] 明确 oracle 开门接口如何接入现有任务或 ROS 流程
- [ ] 明确“交互成功后继续导航”的状态回写链路
- [ ] 设计最小失败处理：门不可开 / 状态未知 / 交互后仍不可达
- [ ] 定义最小日志字段，便于后续复盘每次失败点

## 4.5 实验与评估 TODO

- [ ] 设计 MolmoSpaces 论文主线对比实验：`pure nav` vs `nav + oracle open` vs `interactive planner`
- [ ] 设计 ROS 模块化基线实验：detector-only -> interaction graph -> planner / hint consumption
- [ ] 统计第一阶段 episode 中“确实需要交互”的比例
- [ ] 输出路径长度变化、成功率变化和案例可视化
- [ ] 先形成小样本 sanity check，再考虑自动 benchmark 生成
- [ ] 明确第一版论文图/表最可能来自哪些实验结果
- [ ] 明确 obj-goal 与 point-goal 两个设置下是否共用一套指标与主表

### 4.5.1 当前固定评测标准

详细定义见 [`docs/interactive_navigation_metrics.md`](docs/interactive_navigation_metrics.md)。

主评测表保持简洁，只报告 5 个方法无关指标：

| 指标 | 定义 | 备注 |
|------|------|------|
| `SR` | 最终任务成功率，即满足 `NavToObj` 的距离阈值和 head-camera 可见性条件 | 主结果指标 |
| `SPL` | 成功加权路径效率，失败计 0，成功时按参考路径长度与实际路径长度的比值加权 | 参考路径应来自允许必要交互后的可行计划，而不是纯静态地图 |
| `Interaction Success Rate` | 需要交互的 episode 中，关键交互效果是否完成 | door 看 reachability，container 看 visibility，mixed 看必要交互链是否完成并最终服务目标成功 |
| `Interaction Precision` | 执行过的交互中，有多少是有效交互 | 用于惩罚乱开无关门、无关容器或重复无效交互 |
| `Total Cost` | `path_length + λ * interaction_count` | 后续可扩展为 door / container / failed interaction 的不同权重 |

说明：

- `reachability`、`visibility` 和 `enablement` 是论文叙事、benchmark 构建和 `Interaction Success Rate` 判定的核心语义，不作为主表中三个并列指标。
- `Interaction Success Rate` 在不同 split 下使用不同判定：通道交互要求恢复可达性，容器交互要求揭示目标或满足目标可见性，混合交互要求完成必要交互链并最终满足导航成功条件。
- 无交互样本必须保留，用于评估方法是否克制；其 `Interaction Success Rate` 可记为 `N/A`，但 `Interaction Precision`、`interaction_count` 和 `Total Cost` 仍然参与分析。
- 主结果应按 `all`、`channel`、`container`、`mixed`、`no-interaction` split 展开，并同时报告 macro average，避免某一类样本数量过大主导总体结论。

### 4.5.2 对应脚本规划

后续脚本优先保持以下职责分离：

1. `collect_mixed_rough_catalog.py`
   - 输入：`container_rough_catalog_v1` 与原始 `nav_to_obj` benchmark 起点。
   - 输出：`mixed_rough_catalog_v1`，保留所有实测全开 GT path 穿 interactive door 的 crossing 候选，并额外标注 `mixed_required_verified` 子集。
   - 规则：粗筛的入选门槛只有 GT path 穿门；闭门阻断、门前提示可达属于诊断与优先级标签。粗筛不复用 V2 fine episode，也不把容器中心粗目标冒充最终交互位姿。

2. `build_mixed_interaction_benchmark.py`
   - 输入：一个或多个 `mixed_rough_catalog_v1`。
   - 输出：直接符合统一 `interactive_nav_v3` 定义的 mixed benchmark。
   - 规则：真实容器交互位姿必须再次满足穿门/关门阻断条件；所有 interaction、initial state、success evidence 和 minimality 均来自 MuJoCo/readback 实测。
   - 规则：不把 `oracle_plan` 暴露给 policy；只作为 evaluator / scorer 的 GT 标签与参考计划。

3. `interactive_nav_v3.py` 统一校验入口
   - 结构：公共校验 + channel/container/mixed 场景专用校验。
   - 兼容：保留 `validate_channel_v3_episode`、`validate_container_v3_episode`、`validate_mixed_v3_episode` 专用入口。

4. `evaluate_interactive_nav_dataset.py`
   - 输入：v3 benchmark JSON。
   - 输出：数据集质量报告，而不是 policy 性能报告。
   - 检查：schema validity、无占位字段、`interaction_requirement` 分布、split 分布、`success_evidence` 状态、必要交互链与 prerequisite 一致性、no-interaction 样本是否确实不包含 interactions。

5. `score_interactive_nav_run.py`
   - 输入：v3 benchmark JSON + policy rollout 输出。
   - 输出：主指标表和 per-episode 诊断。
   - 主指标：`SR`、`SPL`、`Interaction Success Rate`、`Interaction Precision`、`Total Cost`。
   - 输出 split：`all`、`channel`、`container`、`mixed`、`no-interaction`，并保留每个 episode 的失败原因用于附录分析。

6. `summarize_interactive_nav_results.py`
   - 输入：一个或多个 scorer 输出。
   - 输出：论文表格用 CSV / Markdown。
   - 作用：对多个方法统一生成主表，避免每个方法单独写统计脚本。

## 4.6 中远期规划 TODO

- [ ] 模块化方法跑通后，搭建数据采集机制
- [ ] 评估是否直接复用 GT / oracle 轨迹作为早期训练数据
- [ ] 设计交互导航轨迹数据结构：观测、图状态、动作、交互结果、重规划事件
- [ ] 为二阶段语音交互预留语言输入、澄清问题和用户反馈记录字段
- [ ] 搭建 Agent 架构原型
- [ ] 搭建端到端架构 baseline
- [ ] 对比模块化方法与端到端方法的优缺点与适用阶段

## 4.7 文档与协作 TODO

- [ ] 在本文件持续维护阶段判断，避免计划和实际开发脱节
- [x] 建立“历史文档 -> 4 个核心文档”的吸收映射，并随迁移过程持续更新
- [x] 逐步降低 `scripts/InteractiveNav/interactive_navigation/` 作为高层入口的必要性
- [ ] 在 4 个核心文档中持续维护“第一阶段闭环定义”，而不是再新增平行高层文档
- [x] 已在核心文档中补充“MolmoSpaces 主线与 ROS 主线对应关系”
- [x] 已在核心文档中补充“planner <- graph interface”最小说明
- [x] 已在 4 个核心文档中补充“小规模验证集说明”，而不是再新增平行高层文档

## 4.8 历史文档迁移 TODO

### A. `agent_init.md`

- [x] 检查是否还有未吸收到 `readme_pi.md` 的初始项目描述
- [x] 确认其后续仅保留历史记录角色

### B. `plan.md`

- [x] 检查是否还有未吸收到 `TODO.md` 的阶段拆解
- [x] 检查是否还有可吸收到 `readme_pi.md` 的整体论述
- [x] 标记其中已过时的 LLM-heavy / broad scope 表述

### C. `discussion.md` / `discussion_2026-04-11.md`

- [x] 提炼仍有效的平台选择与论文立意结论
- [x] 标记已不再作为当前决策依据的历史背景讨论

### D. `survey_2026-04-11.md`

- [x] 仅保留为文献综述来源，不再承担项目总览功能
- [x] 视需要将核心 research gap 精简同步到 `readme_pi.md` 或 `TODO.md`

### E. 完成标准

- [x] 新读者只看 `TODO.md`、`readme_pi.md`、`AGENTS.md`、`test.md`，即可理解当前项目方向
- [x] 历史目录仅用于追溯来源，而不是理解当前主线的必要入口

---

## 5. 我当前的判断

如果以“研究价值 + 工程可落地 + 近期能做出结果”三个维度一起看，当前最应该收紧到下面这条主线：

**第一阶段仍然以门驱动的交互可达性闭环为核心，但当前论文落点先收紧到 obj-goal 与 point-goal，并以下个月 AAAI 投稿为直接目标。**

原因很直接：

1. door 是最自然的 `portal`，最容易和 room connectivity 对齐。
2. door opening 已经有现成任务与能力基础。
3. “不开门不可达，开门后可达/更短”最容易讲清楚交互图的必要性。
4. 这条线最容易形成 `pure nav`、`oracle open`、`interactive planner`、`modular detector-only baseline` 的清晰对比。
5. 在这条线稳定之前，过早同时追求多对象泛化或语音多轮交互，会分散下个月 AAAI 投稿的主线。

---

## 6. 需要与你确认的问题

下面这些问题我先保留在 TODO 中，后面可以和你逐条对齐，避免我把后续规划写偏。

1. 第一阶段 MolmoSpaces 主线里，论文更想先突出 **任务定义 + benchmark + 表示**，还是 **表示 + planner + benchmark**？
2. ROS 模块化主线里，第一版是否要把“planner 真正消费图”作为硬要求，还是先接受 “graph + navigation hints + demo evidence”？
3. 在交互属性分类上，是否确认先按：
   - 通道属性
   - 容器属性
   这两类作为第一阶段主类推进，并把设备与开关类交互放到后续拓展？
4. 数据采集阶段，你更倾向于先做：
   - GT / oracle 轨迹导出
   - 在线执行日志采集
   - 两者都做，但先以 GT 轨迹为主
5. 端到端 baseline 你更想偏：
   - VLM / LLM Agent 架构
   - policy learning / imitation learning
   - 先保留开放，不提前定死

如果后面讨论后有明确结论，这个文件可以继续收敛成更短、更执行导向的版本。
