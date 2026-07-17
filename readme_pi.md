# 交互导航项目总览（readme_pi）

最后更新：2026-07-17

## 1. 文档定位

本文件用于维护交互导航项目的**整体立意、研究目标、系统主线与文档入口**。

在当前仓库中，交互导航相关的高层信息应逐步收敛到以下 4 个核心文档：

1. `TODO.md`：项目子任务拆解、阶段规划与开发推进
2. `readme_pi.md`：项目整体立意、研究目标、系统概述与文档索引
3. `AGENTS.md`：Agent 协作约定、工作边界与文档维护规范
4. `test.md`：开发测试命令汇总、流程测试方式与参数配置

目标是让这 4 个文档逐步成为该项目的唯一高层文档入口。  
位于 `scripts/InteractiveNav/interactive_navigation/` 的历史方案、旧讨论与早期调研，后续应被逐步吸收、整理并淘汰。

---

## 2. 项目一句话概述

本项目关注的不是“做一个更强的开门控制器”，而是：  
**把交互性作为导航空间中的一等变量进行建模，让机器人显式理解交互如何改变环境的可达性、可见性与路径代价。**

---

## 3. 当前研究立意

### 3.1 核心问题

传统导航、ObjectNav 和多数语义图导航方法，通常默认环境拓扑基本静态；  
但在真实室内家居环境中，门、柜门、抽屉、冰箱等可交互对象会改变：

- `reachability`：原本不可达的区域是否变得可达
- `observability`：原本不可见的空间或目标是否暴露出来
- `cost`：原本需要绕行的路径是否被缩短

因此，交互式导航的关键不在于单独优化底层操作控制，而在于：

1. 识别哪些对象会改变导航结构
2. 表达这些对象的交互状态与状态转移
3. 在规划中判断何时值得交互
4. 在交互后更新地图/图结构并继续导航

### 3.2 当前论文主张

当前第一篇论文更适合围绕以下主张组织：

1. 定义 **interaction-aware navigation / interactive object navigation** 任务设定
2. 构建能够表达交互状态、交互代价与状态转移的导航表示
3. 让规划显式考虑导航边、交互触发边与交互后拓扑变化
4. 用 oracle interaction 或现有 open/close 能力验证交互表示与规划闭环

换句话说：

- **交互式导航首先是状态变化环境中的导航问题，其次才是操作问题。**
- **我们关注交互如何改变导航图，而不是单独优化底层交互控制器。**

---

## 4. 当前范围与边界

### 4.1 当前主线

当前主线已经形成以下在线闭环：

`实时 GT 观测 -> 动态语义交互图 -> frontier/interaction/target 候选 -> 独立决策模块 -> 导航或力交互原子动作 -> 图与 planning OCC 更新 -> 继续规划`

其中：

- 动态语义交互图负责维护场景、房间、通道、容器、物体、状态和操作历史
- 决策模块只消费图与探索候选，不反向承担图构建职责
- 当前交互执行默认使用 force backend；RBY1 API 保留为后续可替换 backend
- 当前规则策略是可复现实验基线，模型评分接口是后续增强项

### 4.2 当前不作为第一阶段主线的内容

以下内容目前更适合作为后续扩展，而不是第一阶段核心主类：

- 设备与开关类交互
- 灯光、窗帘、电子解锁等感知辅助或功能控制交互
- 大而全的多交互对象统一 benchmark
- 端到端操作控制优化

### 4.3 当前对 LLM / foundation model 的定位

历史方案中，LLM / VLM 曾被写成较重的系统核心模块。结合当前项目阶段，更合适的定位是：

- `VLM / detector`：可作为感知侧候选能力来源，尤其在 ROS 模块化主线中有现实意义
- `LLM`：保留为后续增强方向，而不是第一阶段必须成立的核心模块

当前第一阶段更重要的是先证明：

1. 交互状态与状态转移能否被稳定表示
2. 图或地图状态能否在交互后更新
3. 规划能否显式利用这些变化完成“先交互再导航”

因此，当前高层文档不把 LLM 作为第一阶段必要卖点，而把它保留为：

- 后续 Agent 架构候选能力
- 长时序语义推理与开放世界扩展方向

---

## 5. 当前系统主线

## 5.1 双主线推进

当前项目采用双主线推进，并共同服务论文产出：

### A. MolmoSpaces 主线

目标：基于 GT / oracle 搭建交互导航任务、表示、规划与 benchmark 主线。

当前职责：

- 交互导航任务设定
- GT 交互信息使用
- benchmark / evaluation 主线
- 论文中的核心任务闭环与定量对比

### B. ROS 模块化主线

目标：基于 detector-only 构建模块化感知与交互图基线方法。

当前职责：

- detector-only 感知输入
- 模块化交互图构建
- 可视化、debug 与系统验证
- 作为与论文主线配套的工程基线

### C. 两条主线的共同问题

两条主线最终都在回答同一件事：

`感知/交互属性 -> 交互图/地图状态 -> 导航规划判断需要交互 -> 执行交互 -> 状态更新 -> 继续导航`

### 5.2 当前正式平台选择

虽然历史讨论中比较过 AI2-THOR、Habitat、BEHAVIOR 等平台，但当前正式主线已经收敛到 **MolmoSpaces**。

当前选择 MolmoSpaces 的主要原因是：

1. 交互与导航能力能够在同一平台中成立
2. 多房间室内环境规模足够，支持 door-blocked path 这类关键实验设定
3. 当前仓库已经具备 `nav_to_obj`、opening task、evaluation、scene map 等直接可复用基础
4. 有利于围绕 `nav_to_obj + oracle open + continued navigation` 快速搭建第一篇论文的最小闭环

因此：

- 历史平台比较保留为背景材料
- 当前正式实验与文档主线，以 MolmoSpaces 为准
- ROS 模块化主线是与 MolmoSpaces 配套的工程验证线，而不是替代平台线

### 5.3 MolmoSpaces 主线与 ROS 主线的对应关系

为了避免“论文主线”和“工程系统”彼此脱节，当前双主线的对应关系应理解为：

| 维度 | MolmoSpaces 主线 | ROS 模块化主线 |
|------|------------------|----------------|
| 目标 | 形成论文中的任务、表示、benchmark 与定量结果 | 形成 detector-only 的工程基线与系统验证链 |
| 交互信息来源 | GT / oracle 为主 | detector-only 为主 |
| 核心输出 | benchmark 设定、任务闭环、对比实验结果 | object detections、unified graph、navigation hints、可视化与调试证据 |
| 价值 | 回答“交互为什么是导航问题的一部分” | 回答“模块化方法能否真实构建可用的交互表示” |
| 风险 | 只有 paper story，没有真实系统链路 | 只有系统 debug，没有清晰 benchmark 和论文叙事 |

当前最理想的关系不是二选一，而是：

- 用 MolmoSpaces 主线证明任务与方法成立
- 用 ROS 模块化主线证明表示与系统链路可落地
- 两条线共享对“交互改变导航图”的核心定义

也因此，后续文档与实验设计中应尽量保持：

1. 主线术语一致
2. 交互状态定义一致
3. “需要交互”的判断逻辑尽量可对齐
4. 结果表达能互相支撑，而不是彼此孤立

### 5.4 当前已验证的决策闭环

当前 ROS/MolmoSpaces 联调已经不再依赖固定路线触发交互，而是拆为三个独立模块：

1. `semantic_candidate_node`：统一生成 frontier、portal interaction 与 target navigation 候选
2. `semantic_rule_decision_node`：基于探索收益、可见性收益、语义收益、目标相关性、距离与交互代价选择行为
3. `semantic_behavior_executor`：执行 `EXPLORE / NAVIGATE / INTERACT`，并等待探索、move_base、力交互和图状态反馈

House 7 同起点、全部可交互门初始关闭、1000 step 验证中：

- `frontier_only` 探索覆盖率为 `22.60%`
- `interactive_rule` 探索覆盖率为 `84.73%`
- 交互策略在第 `27`、`60` 个仿真 step 打开两扇门，最终动态图包含 `3` 个房间
- 两路视频均为 `1000/1000` 精确 step 匹配、`15 FPS` 六联图

进一步的 fridge obj-goal 测试中，系统先开门与探索，在目标进入动态图后切换到 `NAVIGATE` 并成功到达。通用模型评分 backend 已提供 `mock / command / HTTP` 接口，但第一阶段论文结果仍应以规则基线为主，避免把不可复现的远程模型作为必要条件。

---

## 6. 第一阶段目标

第一阶段的最小闭环仍然以 **door-centric** 为核心。

当前最值得验证的问题是：

1. 在 MolmoSpaces 中，当目标被关闭门阻挡时，系统能否利用 GT 交互信息完成“先交互再导航”的任务闭环？
2. 在 ROS 模块化方法中，仅依赖 detector-only 输入，系统能否构建出足以支持交互导航判断的模块化感知与交互图基线？

这意味着第一阶段优先关注：

- `nav_to_obj`
- door / open-close 能力
- interaction-aware graph / 交互感知地图
- oracle 开门后继续导航

不急于同时扩展到：

- fridge / cabinet / drawer 的完整统一评测
- LLM 全流程决策
- 大规模 benchmark 自动生成
- 端到端 agent 学习

### 6.1 当前保留的 research gaps

当前从历史方案和文献综述中保留下来、并且仍与本项目直接相关的 research gaps，主要有：

1. **交互状态持续跟踪不足**
   - 现有不少方法把交互视为一次性事件，而不是可持续维护的环境状态变化

2. **交互改变拓扑后的规划闭环不足**
   - 很多方法会检测可交互对象，但没有把“交互后连通性变化”真正纳入统一规划过程

3. **交互代价与收益缺乏统一建模**
   - 当前项目希望显式比较交互成本与导航收益，而不只是“看到门就开”

4. **缺少围绕交互必要性的清晰 benchmark**
   - 第一篇论文最需要证明的，不是交互控制有多强，而是交互为什么对导航不可或缺

5. **模块化工程基线与论文主线常常脱节**
   - 当前项目保留 MolmoSpaces 主线和 ROS 模块化主线并行，就是为了避免只剩 paper story 或只剩 debug system

### 6.2 planner 消费 graph 的最小接口

当前高层文档层面，planner 不需要一次性消费一个很重的大而全图结构；  
但至少应能从 graph / map 层拿到以下最小信息：

1. **交互对象位置**
   - 例如 door / container 在哪里

2. **交互对象类型**
   - 当前是通道属性还是容器属性

3. **当前状态**
   - 例如 `unknown / closed / open`

4. **交互后可能带来的变化**
   - 是否解锁连通性
   - 是否暴露新区域
   - 是否降低路径代价

5. **是否值得交互的决策信号**
   - 当前目标是否被阻挡
   - 当前区域是否存在更优路径
   - 当前交互是否可能带来有效收益

这意味着第一阶段 planner 的要求可以收敛为：

- 不要求一开始就做复杂长链推理
- 但必须能显式判断“先交互再导航”是否比“直接绕行或放弃”更合理

对 ROS 模块化主线来说，这个最小接口通常会体现在：

- `unified_graph`
- `navigation_hints`
- 以及后续需要收敛的 planner-facing state fields

### 6.3 小规模验证集定义

在进入大规模 benchmark 生成之前，当前项目需要先维护一组**小规模验证集**，用于反复验证第一阶段闭环是否真实成立。

这组验证集的目标不是追求规模，而是提供：

1. 可重复
2. 人工可检查
3. 能清楚体现“交互改变导航结果”

当前建议的小规模验证集应满足：

- 场景数量少，但每个场景语义明确
- 存在明确的 `door-blocked path`
- 能对比至少三种行为：
  - `pure nav`
  - `nav + oracle open`
  - `interactive graph / modular baseline`

这组验证集的作用是：

- 检查任务设定是否成立
- 检查 graph / map 更新是否合理
- 检查 planner 是否真正消费了交互信息
- 为后续 benchmark 扩展提供 sanity check

也就是说，第一阶段不是先做“大 benchmark”，而是先做“小而准的闭环验证集”。

---

### 6.4 当前评测标准

当前交互导航评测采用简洁主指标，避免把调试诊断项全部放入论文主表。主指标固定为：

| 指标 | 含义 |
|------|------|
| `SR` | 最终任务成功率，沿用 `NavToObj` 的距离阈值加 head-camera 可见性条件 |
| `SPL` | 成功加权路径效率，失败为 0，成功时按参考路径长度与实际路径长度的比值加权 |
| `Interaction Success Rate` | 需要交互的 episode 中，关键交互效果是否完成 |
| `Interaction Precision` | 执行过的交互中，有多少是有效交互 |
| `Total Cost` | 总代价，首版使用 `path_length + λ * interaction_count` |

其中 `reachability`、`visibility` 和 `enablement` 是 benchmark 设计与论文叙事中的交互收益类型：

- 通道交互主要体现 `reachability`：开门或打开通道后，原本不可达的目标区域变得可达。
- 容器交互主要体现 `visibility`：打开冰箱、柜门或抽屉后，原本不可见的目标变得可见。
- 混合交互中的 `enablement` 是中间机制：某个交互不一定直接暴露目标，但会使后续交互或后续导航变得可执行。

这些收益类型不作为主表中的三个独立指标，而作为 `Interaction Success Rate` 的判定依据。报告结果时应按 `all`、`channel`、`container`、`mixed`、`no-interaction` 等 split 展开；无交互样本用于惩罚不必要交互，其 `Interaction Success Rate` 可以记为 `N/A`，但 `Interaction Precision` 和 `Total Cost` 仍然有意义。

---

## 7. 当前代码与实验基础

### 7.1 MolmoSpaces 侧

当前仓库已具备可复用基础：

- `molmo_spaces/tasks/nav_task.py`
- `molmo_spaces/tasks/nav_task_sampler.py`
- `molmo_spaces/tasks/opening_tasks.py`
- `molmo_spaces/tasks/opening_task_samplers.py`
- `molmo_spaces/policy/solvers/opening_solver.py`
- `molmo_spaces/evaluation/benchmark_schema.py`
- `molmo_spaces/evaluation/eval_main.py`
- `molmo_spaces/utils/scene_maps.py`

### 7.2 ROS 模块化侧

当前交互图原型主要位于：

- `Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/`

其中已经具备的关键模块包括：

- `object_detection_node.py`
- `room_attribute_node.py`
- `semantic_mapping_node.py`
- `interaction_graph_store.py`
- `interaction_graph_viz.py`
- `semantic_map_store.py`
- `semantic_mapping_gt_replay.py`
- `object_detection_visual_test.py`

这说明当前项目已经从纯概念阶段进入了：

**方向已明确 + 任务基础已存在 + 模块化交互图已成型 + 正在收敛最小闭环**

---

## 8. 文档入口

## 8.1 当前应优先阅读

### 任务与推进

- [TODO.md](/home/user/ldl/molmospaces/TODO.md)

### 项目总览

- [readme_pi.md](/home/user/ldl/molmospaces/readme_pi.md)

### Agent 协作规范

- [AGENTS.md](/home/user/ldl/molmospaces/AGENTS.md)

### 测试与命令

- [test.md](/home/user/ldl/molmospaces/test.md)

## 8.2 历史文档入口

以下文档目前仍有参考价值，但后续应逐步被上面 4 个核心文档吸收：

- `scripts/InteractiveNav/interactive_navigation/agent_init.md`
- `scripts/InteractiveNav/interactive_navigation/plan.md`
- `scripts/InteractiveNav/interactive_navigation/discussion.md`
- `scripts/InteractiveNav/interactive_navigation/discussion_2026-04-11.md`
- `scripts/InteractiveNav/interactive_navigation/survey_2026-04-11.md`

使用原则：

- 子任务拆解与开发推进，以 `TODO.md` 为准
- 项目立意、系统主线与文档入口，以 `readme_pi.md` 为准
- Agent 工作边界与文档维护约定，以 `AGENTS.md` 为准
- 命令与测试流程，以 `test.md` 为准

---

## 9. 历史文档吸收状态

当前不再把历史目录视为高层主入口，而是把它当作待吸收的来源。当前吸收关系如下：

### 9.1 `agent_init.md`

主要内容：

- 早期项目简介
- 基础交互导航概念
- 初始模块拆分思路

当前吸收状态：

- 项目总立意与整体主张，已主要吸收到 `readme_pi.md`
- 第一阶段 door-centric 主线与双主线推进，已主要吸收到 `TODO.md`

后续处理：

- 保留为历史记录
- 不再作为理解项目主线的必要入口

### 9.2 `plan.md`

主要内容：

- 早期技术路线
- 早期 phase 划分
- 早期 benchmark 和平台设想

当前吸收状态：

- 阶段拆解与中远期规划，已主要吸收到 `TODO.md`
- 整体任务主张与边界，已主要吸收到 `readme_pi.md`

仍需注意：

- 其中较重的 LLM 规划层表述，不再直接代表第一阶段主线
- 其中更广的交互类型铺陈，需要以当前 `TODO.md` 为准

### 9.3 `discussion.md` / `discussion_2026-04-11.md`

主要内容：

- 论文立意讨论
- MolmoSpaces 平台选择理由
- 交互导航与 manipulation 的边界判断

当前吸收状态：

- “交互图而非底层控制器”这一主张，已吸收到 `readme_pi.md` 与 `AGENTS.md`
- MolmoSpaces 作为当前正式平台主线，已吸收到 `readme_pi.md` 与 `AGENTS.md`

仍需注意：

- 其中关于平台候选的历史讨论保留为背景，不再作为当前平台决策依据

### 9.4 `survey_2026-04-11.md`

主要内容：

- 文献综述
- 研究空白整理
- 候选技术路线与相关工作地图

当前吸收状态：

- 与当前项目直接相关的立意和研究空白，已被提炼进 `readme_pi.md` 与 `TODO.md`

仍需注意：

- survey 仍然是文献参考材料，但不应承担当前项目总览功能

### 9.5 当前结论

当前 4 个核心文档与历史目录的职责分工应理解为：

- `TODO.md`：当前项目任务和推进状态的唯一主入口
- `readme_pi.md`：当前项目总览和研究立意的唯一主入口
- `AGENTS.md`：当前协作边界和维护约定的唯一主入口
- `test.md`：当前运行命令与测试流程的唯一主入口
- `scripts/InteractiveNav/interactive_navigation/`：历史背景、来源材料和待吸收记录

---

## 10. 后续文档维护目标

后续文档维护的方向不是继续堆积新的“方案文档”，而是：

1. 把历史讨论中的有效结论吸收到 4 个核心文档
2. 把已经过时、发散或与当前主线冲突的表述剔除
3. 让读者不必依赖 `scripts/InteractiveNav/interactive_navigation/` 才能理解项目整体方向

理想状态下，这 4 个文档应分别承担：

- `TODO.md`：项目拆解与阶段状态
- `readme_pi.md`：项目总入口与研究立意
- `AGENTS.md`：协作规范与维护边界
- `test.md`：运行方式、调试方式、测试方式

---

## 11. 当前待确认的高层问题

以下问题已经收敛到当前主线，但仍需要在后续讨论中继续定稿：

1. 第一阶段论文更突出“任务定义 + benchmark”，还是“表示 + planner”？
2. ROS 模块化主线第一版是否要求 planner 真正消费图，还是先接受 graph + hints + demo evidence？
3. MolmoSpaces 主线与 ROS 主线的公共表示接口应收敛到什么粒度？
4. 模块化方法跑通后，数据采集优先使用 GT / oracle 轨迹，还是在线执行日志？
5. Agent 架构与端到端 baseline 应在哪个阶段正式进入主线？
