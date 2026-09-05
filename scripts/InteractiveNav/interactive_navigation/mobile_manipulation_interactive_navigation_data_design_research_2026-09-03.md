> 研究笔记说明
>
> 调研时间：2026-09-03。本文只把论文、作者项目页、官方数据文档和本仓库文档作为事实来源；“建议/推断”会明确标注。本文是交互导航数据设计的研究材料，不修改现有代码或数据。

# 移动操作与交互导航：面向 VLA 的 dense 数据定义建议

## 0. 结论先行

1. **移动操作（mobile manipulation）和交互导航（interactive navigation）不是同义词。**移动操作描述机器人能力和任务族：底盘、机械臂（可能是双臂）需要协同完成操作；交互导航描述导航问题的目标和环境语义：机器人为了到达位置/目标而有意改变门、容器或障碍物状态，之后继续导航。两者有交集，但前者可以是不涉及“通行”的取放、擦拭、倒液体，后者也可以是只推障碍物的非抓取交互。
2. **不要在“全身关节”与“底盘+末端”之间二选一。**采集时保存可回放的原始全身状态/控制量；在其上派生一个稳定、低维、带显式 mask 的 canonical action 给 VLA。建议的第一版 canonical action 是 `mode + base twist/ΔSE(2) + 每只手的 EE ΔSE(3)+gripper + object/primitive + active mask + terminate`。
3. **交互导航的关键标签不是一串开门关节值，而是状态转移。**每个交互事件都应能回答：是否需要交互、目标对象是谁、交互前后可达性/可见性/拓扑如何变化、实际代价和是否成功。高层 `oracle_plan` 与低层轨迹要分开保存。
4. **dense 的含义应是时间对齐且可重放，而不是把所有物理步都直接喂给 VLA。**保留传感器和控制器原生频率的 raw stream，再生成固定频率（例如 5–10 Hz）的训练视图；训练视图与 raw 之间保留明确的时间戳、坐标系、命令/测量来源和索引映射。

上述判断与 HRL4IN 对三类任务的区分一致：普通导航改变机器人位置而不改变环境配置，静态操作改变环境配置而不改变机器人位置，交互导航把二者组合起来，并可在纯导航、纯操作和组合阶段之间切换（[HRL4IN 论文](https://arxiv.org/abs/1910.11432)，[项目页](https://sites.google.com/view/hrl4in/home)）。Interactive Gibson 进一步把“推物体/开门以完成导航”和路径质量、交互扰动/努力放在同一 benchmark 中（[论文](https://arxiv.org/abs/1910.14442)，[项目页](https://sites.google.com/view/interactivegibsonenv/home)）。

## 1. 任务概念的边界

| 概念 | 首要目标 | 是否有意改变环境状态 | 典型动作/阶段 | 主要评价 |
|---|---|---|---|---|
| 导航（navigation） | 到达位置、房间或目标对象附近 | 通常否 | 底盘位姿/速度、避障、到达 | 成功率、路径长度、SPL/时间 |
| 操作（manipulation） | 改变物体状态或完成接触任务 | 是；机器人可基本固定 | 抓取、放置、推/拉、开关 | 物体/任务状态、接触安全、成功率 |
| 移动操作（mobile manipulation） | 在移动工作空间内完成操作任务 | 通常是 | `navigate → approach → manipulate → relocate`，底盘和一只/两只手协同 | 任务成功、全身协调、时间/碰撞 |
| 交互导航（interactive navigation） | **通过交互改变可达性、可见性或路径代价后完成导航目标** | 是，且该改变服务导航 | `navigate → approach interaction → actuate/push → verify → continue navigation` | 导航成功 + 交互成功 + 路径/交互代价 |
| NAMO/非抓取交互（一个子类） | 推开或搬移障碍物以清路 | 是 | 判断可推动性、推到目标位、重新规划 | 到达率、推力/位移/扰动代价 |

“移动操作”是以能力/embodiment 为中心的宽任务族；“交互导航”是以导航因果目标为中心的任务定义。比如“把锅放进柜子”是移动操作，但不一定是交互导航；“开门后进入另一房间”是交互导航，底层可由移动操作、专用开门器或 oracle executor 实现。自适应非抓取工作也明确把“可绕行则绕行、可操纵则重定位障碍物”作为交互导航决策，而不等同于抓取式移动操作（[Adaptive Non-prehensile Interactive Navigation](https://arxiv.org/abs/2410.13418)）。

## 2. 现有移动操作/VLA 数据集一般长什么样

### 2.1 共同的数据单元

主流数据集通常以**变长 episode/trajectory**为单位，每个 step 对齐视觉、语言/任务、proprioception、action 和终止/成功信息；仿真数据还常保留可恢复环境的 state snapshot。RLDS 的官方约定把 episode 表示为 `steps` 序列，要求 `is_first/is_last`，并允许 observation、action、reward、discount、metadata 等字段（[RLDS 官方仓库](https://github.com/google-research/rlds)）。不同数据集采用 HDF5、RLDS/TFDS 或自定义 JSON/H5，但“原始记录 + 派生训练视图”是更稳妥的组织方式。

### 2.2 代表性数据集对照

下表只摘录一手来源明确写出的规模和字段；规模以论文/当前官方发布页为准，版本更新可能造成小幅差异。

| 数据集/来源 | embodiment、任务和规模 | 观测/动作/状态形态 | 对本项目的启示 |
|---|---|---|---|
| **Open X-Embodiment / RT-X** | 官方页称汇集 60 个已有数据集、1M+ 条真实轨迹、22 种 embodiment，覆盖单臂、双臂和四足；论文摘要报告 527 skills/160,266 tasks（[项目页](https://robotics-transformer-x.github.io/)，[论文](https://arxiv.org/abs/2310.08864)，[官方 repo](https://github.com/google-deepmind/open_x_embodiment)）。 | RT-X 将动作统一到 gripper frame 的 7D `(x,y,z,roll,pitch,yaw,gripper)` 或相应 rates；未使用的维度在其训练管线中置零。 | 跨 embodiment 需要 canonical action，但零填充不应掩盖“该维度不存在”与“该维度确实为零”，本项目建议额外保存 `valid/active mask`。 |
| **RT-1** | 官方项目页报告 13 台机器人、17 个月、130K+ episodes、700+ tasks，控制频率 3 Hz；任务包含抽屉开关等移动/操作技能（[项目页](https://robotics-transformer1.github.io/)，[论文](https://arxiv.org/abs/2212.06817)）。 | 7 个 arm 变量（位姿+夹爪）+ 3 个 base 变量 `(x,y,yaw)`，再加离散 mode（arm/base/terminate）；模型闭环逐步输出。 | “离散 mode + 连续底盘/末端”已经是可行的 VLA 接口，不要求模型直接预测每个关节。 |
| **BridgeData V2** | 官方当前发布页列出 60,096 条轨迹（50,365 teleop、9,731 scripted）、24 个环境、13 类技能；WidowX 250 固定臂、VR teleop，5 Hz，平均约 38 steps。论文发表版本曾报告 53,896，属于 release 版本差异（[项目页](https://rail-berkeley.github.io/bridgedata/)，[论文](https://proceedings.mlr.press/v229/walke23a.html)）。 | TFDS schema 为每步 7D float action/state、语言指令、`is_first/is_last/is_terminal`、reward/discount 和多路 RGB/RGB-D/腕部图像（[官方 TFDS schema](https://github.com/tensorflow/datasets/blob/master/docs/catalog/bridge.md)）。 | 大量数据仍是固定底座、7D EE 语义动作；可作为视觉/语言预训练，但不能代替交互导航的底盘与环境状态。 |
| **DROID** | 官方数据卡/论文报告约 76K 轨迹、350 小时、564 场景、86 tasks、50 collectors；Franka Panda 固定臂，双 Zed 立体相机加腕部相机，Quest teleop（[数据文档](https://droid-dataset.github.io/droid/the-droid-dataset)，[论文](https://arxiv.org/abs/2403.12945)）。 | RLDS 视图同时提供 gripper/cartesian/joint proprio；`action_dict` 保留 gripper、Cartesian、joint 的位置/速度，而 canonical `action` 是 7D（6 joint velocity + gripper）；raw episode 另存 H5 与视频。 | 同一数据集保留“多种动作坐标系”是常见做法：VLA 可用一个 canonical head，回放/控制器仍需要 joint 或 Cartesian raw。 |
| **RH20T** | 官方页称有 110K+ contact-rich real sequences，含视觉、力、音频、动作和对应人类示范视频；论文列出 48 个 RLBench、29 个 MetaWorld 和 70 个自定义任务（表中约 140+ tasks）（[项目页](https://rh20t.github.io/)，[论文 PDF](https://rh20t.github.io/static/RH20T_paper_compressed.pdf)）。 | 明确记录多频率流：RGB/depth/IR 10 Hz，joint angle/torque 10 Hz，TCP pose 100 Hz，力/力矩 100 Hz，音频 30 Hz，触觉最高 200 Hz；同时提供变换后的 TCP、joint、gripper、FT 等字段。 | dense 采集应保留原生频率和变换后的统一视图，不能只存一段视频或单一动作数组；接触/力信息对开门失败分析很有价值。 |
| **Mobile ALOHA** | 低成本双臂移动平台；论文用每任务 50 demos 展示柜门、电梯、推椅、烹饪等 whole-body 任务（[论文](https://arxiv.org/abs/2401.02117)，[官方 repo](https://github.com/MarkFzp/mobile-aloha)）。 | 直接把两臂 14-DoF joint positions 与底盘线/角速度拼成 16D action；同步记录底盘速度和臂关节位置，3 路 RGB 相机约 50 Hz。 | 全身 joint action 可以工作，但强烈依赖具体硬件；建议把它作为 raw/control target，同时派生 EE+base canonical 视图。论文还显示底盘速度误差会使开环 replay 失败，说明必须记录实测状态并做闭环评估。 |
| **MoMaRT** | 手机/摇杆 teleop 的移动操作框架；官方页报告 1,200+ successful demos、5 个长时域厨房任务和超过 11 小时数据（[项目页](https://sites.google.com/view/il-for-mm/home)，[数据页](https://sites.google.com/view/il-for-mm/datasets)，[论文](https://proceedings.mlr.press/v164/wong22a.html)）。 | 数据 HDF5 中每个 demo 有 `actions (N,10)`、RGB/depth head+wrist、`proprio (N,10)`（head joint、grasped、EE pose/quaternion）、`proprio_nav (N,2)`（底盘线速度幅值、角速度）、lidar、object/GT nav 等。 | 移动操作数据已经把导航 proprio、EE、视觉、任务阶段和成功信息放在同一 episode；本项目可沿用“导航流 + 操作流 + 事件/目标流”的分层。 |
| **LaNMP** | 语言条件的长时域 room-to-room pick-and-place；574 条轨迹、8 个仿真/真实环境，524 sim + 50 real，Spot 四足移动操作器（[官方论文 PDF](https://llhomerobots.github.io/docs/corl_llhomerobots_2024-cr_paper_16.pdf)）。 | 每条轨迹 20+ attributes，包括 RGB-D、分割、body/EE/被抓物 pose；真实采集约 3 Hz，另有 body/arm velocity、joint states。其 RT-1 改造输出 7D token：body、EE、grasp、控制 mode、terminate；论文报告预测相邻差分比绝对坐标更稳定。 | 最接近“语言+导航+操作+感知”的公开范式；明确说明多数旧数据缺导航或只覆盖短时域，支持为交互导航单独增加长时域、事件和图状态。 |
| **LIBERO** | 官方 repo/site 提供 4 个 suite、130 个语言任务和高质量 human teleop demos；任务由 BDDL/PDDL 风格场景与自然语言定义（[官方 repo](https://github.com/Lifelong-Robot-Learning/LIBERO)，[数据页](https://libero-project.github.io/datasets)，[论文](https://arxiv.org/abs/2306.03310)）。 | HDF5 demo 保留任务描述、MuJoCo XML/初始 states、images/depth、proprio、actions/rewards/dones；常见控制 action 为 7D EE pose + gripper。 | 仿真任务应把可精确恢复的 state、任务语言和观测分开；VLA 训练视图可以低维，评估/重放仍依赖 full simulator state。 |
| **ManiSkill demos** | 官方文档定义所有 demo 为 HDF5；raw demo 重点保存 actions、随机种子和 `env_states`，观测通常为可选项以节省空间；可 replay/conversion 生成 RGB-D、reward 或换 control mode（[demos 文档](https://github.com/haosulab/ManiSkill/blob/main/docs/source/user_guide/datasets/demos.md)，[replay 文档](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html)）。 | `env_states` 含 actors/articulations 的完整状态；action 与控制模式可能不覆盖所有下游需求，重放时按目标观测/控制模式重新导出。 | “lossless raw state + 可重复派生训练视图”比一次性决定唯一 action schema 更耐用，尤其适合 MolmoSpaces 仿真。 |
| **Interactive Gibson**（交互导航 benchmark，非主要示范库） | 106 个场景、1,984 个可交互 CAD 对象/对齐实例；包含门等 articulated objects 和 Fetch/JackRabbot 等移动机器人（[论文](https://arxiv.org/abs/1910.14442)，[项目页](https://sites.google.com/view/interactivegibsonenv/home)）。 | 观测可含 RGB、深度、语义、里程计、碰撞和机器人状态；控制可作用于轮子/各关节；指标同时考虑路径效率和物体扰动/交互努力。 | 交互导航样本必须有对象/joint 状态和交互代价；仅保存底盘轨迹无法判断“是否因为开门才可达”。 |

### 2.3 从数据集对照得到的规律

- **Episode 是主单位，step 是同步索引。**每个 step 最少应有 `observation_t`、`action_t`、低维状态、语言/任务 ID 和终止标志；动作到底是绝对值、差分还是速度必须写在 metadata 中。RLDS 的 `is_first/is_last` 约定可作为互操作层（[RLDS](https://github.com/google-research/rlds)）。
- **同一条轨迹保留多个坐标系。**DROID 同时保留 joint、Cartesian 和 gripper action；RH20T 同时保留 TCP、joint、力/力矩；Mobile ALOHA 同时有底盘与臂动作（[DROID schema](https://droid-dataset.github.io/droid/the-droid-dataset)，[RH20T](https://rh20t.github.io/)，[Mobile ALOHA](https://arxiv.org/abs/2401.02117)）。由此可见，公开一个 canonical action 不意味着应该丢弃 raw joint/传感器。
- **多速率是常态。**相机、关节、TCP、力传感器的频率不同；应以 timestamp 对齐，而不是假设所有数组长度/频率相同。LaNMP 的真实采集约 3 Hz、RH20T 的 TCP/FT 达 100 Hz 以上，正好说明训练频率和记录频率可以分离（[LaNMP PDF](https://llhomerobots.github.io/docs/corl_llhomerobots_2024-cr_paper_16.pdf)，[RH20T PDF](https://rh20t.github.io/static/RH20T_paper_compressed.pdf)）。
- **语言、场景和成功条件是一等字段。**LIBERO 用 BDDL/语言描述任务，RT-1 为每个 episode 提供自然语言指令，LaNMP 将语言、导航、操作和感知放进同一长时域轨迹（[LIBERO](https://libero-project.github.io/datasets)，[RT-1](https://robotics-transformer1.github.io/)，[LaNMP PDF](https://llhomerobots.github.io/docs/corl_llhomerobots_2024-cr_paper_16.pdf)）。
- **公开移动操作/VLA 数据仍不等于交互导航数据。**LaNMP 的相关工作分析指出，很多大数据集主要是固定底座、短时域或 manipulation-only，缺少多房间导航；因此不能直接把 OXE/Bridge/DROID 的数量当成交互导航覆盖度（[LaNMP 论文](https://llhomerobots.github.io/docs/corl_llhomerobots_2024-cr_paper_16.pdf)）。

## 3. 与仓库现有 V3 / full H5 设计的对照

这一节只描述当前仓库已有能力，不暗示本次要修改实现。

| 层/文件 | 当前已有字段和语义 | 对 VLA dense 数据仍缺或需显式化 |
|---|---|---|
| **V3 benchmark JSON** | `interactive_nav_v3` 已表达 `interaction_domains`、`interaction_requirement`（required/beneficial/unnecessary/unknown）、target/success criteria、`initial_state`、每个 interaction 的 id/type/object/joint/effect/prerequisite/初始→目标状态、`oracle_plan(s)` 和 `generation_validation`；oracle step 固定为 `navigate/set_view/open_joint/observe_target`。这些是高层 GT/评估定义，不是低层 action trajectory（[V3 README](../dataset_definition/v3/README.md)）。 | 没有逐 step 的视觉—状态—动作对齐；没有 `event_id/t0/t1`、实际交互 primitive、命令与测量的成对记录，也没有标准的 `graph_snapshot/graph_delta`。`oracle_plan` 不应直接当作 VLA action label。 |
| **标准生成 H5** | `actions/commanded_action`、`joint_pos`、`joint_pos_rel`、`ee_pose`、`ee_twist`；`obs/agent/qpos/qvel`；视频引用与 camera intrinsics/extrinsics；`obs_scene`、reward、success、terminated、truncated（[docs/data_format.md](../../../docs/data_format.md)）。这已经区分了“命令动作”和“观测 proprio”的大方向。 | 命令/测量的来源、时间戳、控制模式、维度 mask、base/left/right arm 的稳定命名仍未形成跨 episode 契约；`qpos/qvel` 没有必然对应实际 actuator feedback（如 torque/current/contact）。 |
| **`interactive_nav_full_rollout_v1` H5** | recorder 逐 step 写 `images/*`、`step_index/timestamp/dt`、`segment/phase`、`actions/{type,vector,json}`、`states/{json,qpos,qvel}`、reward、terminal/truncated、info；文件 attrs 有 schema、episode、success、terminal reason、step count（[full_rollout_recorder.py](../collection/full_rollout_recorder.py)，[V3 README 的 full 模式说明](../dataset_definition/v3/README.md)）。当前 runner 在 `task.step(action)` 后再调用 `capture(observation, action=...)`，因此 action/observation 的“动作前/动作后”语义需要在训练转换中显式固定（[container_scene_probe.py](../container_scene_probe.py)）。 | `actions/vector` 是 variable-length 拼接向量，虽有 `components` JSON，但没有强制的 named base/EE schema 或 `active_mask`；`states/qpos/qvel` 是全 MuJoCo 数组，未强制保存 base pose/twist、左右 EE pose/twist、gripper、joint command/readback 的成对字段；没有独立 event 表和 graph delta。 |
| **RBY1 sensors** | 已有 `qpos/qvel`、`left_tcp_pose/right_tcp_pose`、`robot_base_pose`、`door_state`、`env_states`、`object_poses`、`object_image_points`、`policy_phase`，以及 `LastCommandedJointPos/RelativeJointPos/EEPose/EETwist`（[rby1_sensors.py](../../../molmo_spaces/env/rby1_sensors.py)，[sensors.py](../../../molmo_spaces/env/sensors.py)）。 | 这些是 simulator observation/sensor 接口；full recorder 当前主要落盘 MuJoCo `qpos/qvel` 和 action JSON，命名 sensor 是否进入训练 H5、command 与 measured 如何配对、缺失维度如何 mask，仍需在数据契约层固定。 |
| **ROS/raw recording** | raw 目录已有 `step_boundaries`、camera manifest、地图栅格、navigation 的 odom/subgoal/plan、semantic 的 `gt_observations/unified_graph/candidates/selection/execution_state`，并强调按 receipt 时间做因果对齐（[raw_recording_format.md](../raw_recording_format.md)）。 | 已有 graph/导航原始来源，但还没有供 VLA 直接消费的统一 `graph_delta`（节点/边新增、状态变化、可达性变化）和交互 event 区间；H5 dense stream 与 JSONL receipt 的主键/时间映射也应在 schema 中明确。 |
| **V3 当前 TODO** | 当前一个 articulation joint 对应一个 interaction、一个 oracle `open_joint`；双门/多 joint 的 interaction group、`all/any/ordered` 联合语义暂未定义（[V3 TODO](../dataset_definition/v3/TODO.md)）。 | 双臂协同或双开门任务不能只靠把两个 joint 向量拼接来表达；需要 `interaction_group_id`、成员 interaction、执行模式和联合成功条件。 |

**对照结论：**仓库已经有很好的高层交互 GT 和相当完整的 dense rollout 骨架，缺口集中在“动作/状态的稳定命名和 mask、command/measured 配对、事件与图状态增量”，而不是再造一套 benchmark JSON。建议保持 V3 作为 episode/GT 层，在 full H5 或 sidecar 中增加下面的 VLA 层。

现有 `segment/phase` 名称（例如 `nav_to_door`、`force_open_door`）可以原样保留用于回放；新增一个规范化的 `event_type`/`mode` 映射即可，不必为了 VLA 改写历史字符串。

另一个需要明确的边界是：V3 统一生产配置目前不生成 `open_gt_control`，也不主动构造错误动作 rollout；full 失败文件只用于诊断、不会标记为 `training_eligible`（[V3 README](../dataset_definition/v3/README.md)）。这对 benchmark 质量控制是合理的，但若研究 VLA 的失败恢复，应在不污染主 benchmark 的前提下另存带 `failure_reason/recovery` 的负样本 split。

## 4. 建议的数据分层与 schema

### 4.1 四层数据模型

建议把一条 episode 组织成四层；每层可以独立过滤，避免把 privileged GT 误送给 policy。

| 层 | 内容 | 作用 |
|---|---|---|
| **L0：episode/task** | episode ID、scene/house、robot embodiment、自然语言指令、起点/目标、地图版本、相机标定、控制/采样频率、坐标系、split、随机种子、隐私/质量元数据；复用 V3 的 target、interaction requirement、success criteria。 | 定义“这条轨迹要完成什么”，支持按场景/对象/任务拆分。 |
| **L1：interaction event** | `event_id`、`type`、`target_object_id`、`t_start/t_end`、阶段、交互前后 articulation/semantic state、可达性、可见性、拓扑/地图变化、路径长度和交互代价、success/failure reason。 | 表达交互导航的因果单元，支持 event-level 训练和评估。 |
| **L2：dense synchronized step** | 统一时间戳下的 RGB/depth/seg/LiDAR、base/arm proprio、物体/joint 状态、graph snapshot/delta、`action_cmd`、`action_measured/readback`、reward/terminal。 | 监督 VLA、诊断时序和重建闭环。 |
| **L3：raw/replay** | 原生传感器流、全身 q/dq（必须；tau/current/contact/FT 按平台可得性记录）、轮速/底盘控制、controller 内部目标、仿真 `qpos/qvel/env_state` 和原始 receipt。不存在的传感器用 capability/valid metadata 显式表示。 | 精确 replay、低层控制、sim-to-real、未来重新导出其他 action schema。 |

V3 的 `interactive_nav.interactions/oracle_plan/generation_validation` 属于 L0/L1 的 privileged GT；不要把它和 policy 可见观测混在同一个默认输入字典中（[V3 policy/privileged 约定](../dataset_definition/v3/README.md)）。

### 4.2 每个 step 的最小字段

建议每个 step 明确以下语义（字段名可按现有 H5 风格实现）：

```text
steps/
  timestamp_ns, dt, source_seq, is_first, is_last, terminal, truncated
  obs/
    rgb/{camera}, depth/{camera}, segmentation/{camera}, lidar
    base/{pose_se2, twist_body, odom_cov}
    left_arm/{q, dq, tau, ee_pose_base, ee_twist, gripper}
    right_arm/{q, dq, tau, ee_pose_base, ee_twist, gripper}
    torso_head/{q, dq}                 # 没有该部件时仍有 schema/version 信息
    objects/{object_id: pose, velocity, joint_state, visibility, contact}
  action_cmd/                           # 控制器/teleop 实际发出的命令
    raw/{base, joint, ee, gripper, mode}
    canonical/{mode, base, left_ee, right_ee, gripper, object_id, primitive}
    active_mask, valid_mask, horizon
  action_measured/                      # 执行后 readback；没有时显式 null/quality
    base_twist, joint, ee, gripper, contact/effort
  graph/{snapshot_or_ref, delta}
  reward, info, quality
events[]                                 # L1，建议单独 group/JSONL
```

关键约定：

- 采用 `action_cmd[t]` 作用于 `obs[t]` 并产生 `obs[t+1]` 的约定，另存 `next_obs` 或明确索引映射；不要让“第 i 个 state 对应第 i+1 个 action”的历史约定悄悄进入新训练集。现有 MolmoSpaces 标准 H5 明确首 action 是 dummy、末尾有 done sentinel，训练转换时必须写出 trim 规则（[docs/data_format.md](../../../docs/data_format.md)）。
- 所有连续量写单位、坐标系和语义：base twist 是 body/world frame？EE pose 是 base/world/gripper frame？旋转用 quaternion、rotation vector 还是 Euler？`commanded`、`measured`、`target`、`readback` 不要共用一个无说明的数组。
- 图状态可以是完整 snapshot 或稀疏 delta，但 delta 至少要能重建：对象状态（closed/ajar/open/locked）、joint fraction、可达性/可见性、受影响的拓扑边，以及 `source_seq`/时间戳。已有 raw `unified_graph` 可作为输入来源，但仍应定义 VLA-facing 的稳定字段（[raw recording format](../raw_recording_format.md)）。

### 4.3 推荐的 canonical action

对于第一阶段 door/container 交互，建议 VLA 的每个动作 token/向量为：

```text
canonical_action_t = {
  mode: NAVIGATE | APPROACH | PRE_CONTACT | ACTUATE | VERIFY | RECOVER | STOP,
  base: [vx, vy, wz] or delta_SE2,       # 按底盘类型选择其一，并保存 frame
  left_ee:  [dx, dy, dz, d_rx, d_ry, d_rz, grip],
  right_ee: [dx, dy, dz, d_rx, d_ry, d_rz, grip],
  active_mask: {base, left_ee, right_ee, head, torso},
  target_object_id: string | null,
  primitive: OPEN | CLOSE | PUSH | PULL | VERIFY | NONE,
  actuation: {joint_target_fraction?, effort?, wrench?}, # 交互控制可选头
  terminate: bool
}
```

这是**设计建议**，不是声称所有现有数据集都采用同一向量。其依据是：RT-1 已采用 arm+base+mode+terminate；RT-X 使用 gripper-frame 7D canonical action；LaNMP 将 body/EE/grasp/mode/terminate 组合成 token，并报告相邻差分比绝对坐标更稳定（[RT-1](https://robotics-transformer1.github.io/)，[RT-X](https://robotics-transformer-x.github.io/)，[LaNMP PDF](https://llhomerobots.github.io/docs/corl_llhomerobots_2024-cr_paper_16.pdf)）。

具体建议：

1. 底盘是差速时将 `vy` 标记为 invalid，而不是把“不可控”与“要求横移 0”混成同一个零；全向底盘可使用 `[vx,vy,wz]`。同时保留原生轮速/底盘控制作为 raw。
2. 双臂各保留一个固定 slot；单臂或单手交互用 `active_mask`/`arm_mask`，不要通过改变向量长度或无说明的零填充表达“没有这只手”。这也能兼容 OXE 的统一维度，同时避免其零值语义在本项目中丢失信息。
3. EE 量优先使用**相对 base/当前 EE 的小步差分**，并记录绝对 pose 作为 state；绝对 world pose 只作为可选高层目标。这样更容易跨场景、跨 embodiment，并与 LaNMP/RT-1 的差分经验一致。
4. `mode`、`target_object_id`、`primitive` 是离散监督头；连续 base/EE/gripper 是并行回归或离散 bin 头。交互对象 ID 可以在训练时映射为候选/检测 token，评估时再还原到 V3 的稳定 interaction ID。
5. 在一个采样 tick 内如果底盘和双臂同时动，不要拆成互相覆盖的三条 action；用同一时间戳的联合向量表示，并在 raw 中保留各控制器的实际命令。

对于力控开门、抽屉卡滞等任务，可在 canonical action 增加可选的 `joint_target_fraction`、effort 或 wrench head；若该控制由现有 oracle/controller 完成，则把它标为 executor/privileged channel，而不要假装 EE 位姿单独包含了接触力语义。

建议把 mask 再细分成三个概念：`present_mask`（该 embodiment 是否有该执行器/自由度）、`active_mask`（本 tick 是否有意控制该分支）、`valid_mask`（数值是否通过同步/质量检查）。例如差速底盘的 `vy` 是 `present=false`；开门时未动的右臂是 `present=true, active=false`；丢帧或校准失败则 `valid=false`。三者不能都用一个零向量代替。

可进一步采用两时间尺度（**工程建议**）：1–3 Hz 的高层 `event/mode/target` token 负责“去门边、开门、验证、继续导航”，5–10 Hz 的 canonical base/EE head 负责连续跟踪，50–200 Hz 的 joint/力/控制器流只用于 L3 replay。RT-1 和 LaNMP 的公开系统都采用约 3 Hz 的高层动作/采集，而 RH20T 展示了 TCP/FT 等底层流可远高于训练频率（[RT-1](https://robotics-transformer1.github.io/)，[LaNMP PDF](https://llhomerobots.github.io/docs/corl_llhomerobots_2024-cr_paper_16.pdf)，[RH20T PDF](https://rh20t.github.io/static/RH20T_paper_compressed.pdf)）。这样既能保留 dense 证据，又不会要求 VLA 逐物理步预测高维关节。

### 4.4 全身 joint action 与 base+EE action 的取舍

| 选择 | 优点 | 风险 | 建议用途 |
|---|---|---|---|
| 全身 `q/dq/tau` 序列 | 可精确 replay，保留 null-space、碰撞和接触细节；与仿真 controller 直接对接 | 维度高、强 embodiment-specific；不同机器人 joint 数/顺序/控制频率不一致，VLA 容易学到硬件而非任务 | **必须采集并保存为 raw/L3**；用于低层 policy、重放、动力学/失败分析 |
| 底盘 + 单/双臂 EE | 维度低、语义清楚、较易跨机器人；与 RT-1/RT-X/LaNMP 公开接口相容 | 需要 IK/whole-body controller；可能丢失肘部构型、接触力和不可达原因 | **作为 VLA canonical/L2 action 首选**；由控制器落到 joint |
| 只有底盘动作 | 适合纯导航基线 | 无法表达“伸手开门/拉抽屉”，也无法判断交互是否完成 | 只作为 no-interaction 对照或导航子任务 |

因此对用户提出的二选一问题，答案是：**采集层两者都存；训练层用带 mask 的 `base + 双 EE + gripper + mode/primitive`；评估和 replay 保留全身关节及实测 readback。**Mobile ALOHA 的 16D 全身拼接说明 joint+base 方案可行，但其论文也展示了底盘速度误差导致开环 replay 漂移；DROID/RH20T 则说明同时保留 joint、Cartesian、TCP 和传感器流更有利于重导出（[Mobile ALOHA](https://arxiv.org/abs/2401.02117)，[DROID schema](https://droid-dataset.github.io/droid/the-droid-dataset)，[RH20T](https://rh20t.github.io/)）。

### 4.5 交互事件与图增量标签

每个 `event` 建议至少包含：

```text
event = {
  event_id, type, target_object_id, interaction_id,
  t_start, t_end,
  pre:  {joint_fraction, semantic_state, reachable, visible, graph_hash},
  post: {joint_fraction, semantic_state, reachable, visible, graph_hash},
  intended_effect: restore_reachability | reduce_navigation_cost |
                   reveal_target | enable_interaction,
  path_cost_before, path_cost_after, interaction_cost,
  success, failure_reason, operator_or_executor, quality
}
```

其中 `pre/post` 应由 simulator joint/contact/visibility 与导航图计算器自动生成，再做少量人工 QA；不要只依赖操作者按键时刻。V3 已有 `effect_types`、typed prerequisites、oracle prefixes 和 success evidence，可直接作为 event 标签的 GT 来源（[V3 README](../dataset_definition/v3/README.md)）。交互导航的必要性至少要区分 `required`、`beneficial`、`unnecessary`、`unknown`，否则模型会把“可选捷径”误学成“必须开门”。

## 5. 如何采集：一个可执行的分阶段协议

### 5.1 先定义覆盖矩阵，不要先追求轨迹数量

第一阶段建议以项目已经支持的 **channel/container** 为主：

- channel：铰链门、滑动门；开/关、推/拉侧、门前/门后、单门/潜在双门；
- container：冰箱/柜门、抽屉；目标可见性从关闭到打开的变化；
- 对照：`required`、`beneficial`、`unnecessary` 和失败/锁定/被阻挡；
- 每个 interaction 至少改变一种因果量：可达性、目标可见性或路径代价；纯“动作看起来像开门”但没有状态变化的轨迹标为失败/诊断。

建议先做一个小 pilot（例如 30–50 条用于 schema/replay smoke，再扩展到 100–300 条平衡 episode）；这是工程配额建议，不是文献事实。比起把同一门重复 10,000 次，更应优先覆盖不同 scene、对象实例、初始开度、机器人起始侧、目标位置和失败原因。

### 5.2 冻结 episode/scene GT

采集前为每条 episode 冻结：V3 episode JSON、scene/object/joint 稳定 ID、joint range/初始状态、机器人起点和目标、地图/occupancy 版本、相机标定、语言 instruction、允许的 interaction types、成功条件。V3 已把 joint name 作为回放主键、把 `scene_modifications.articulation_states` 作为权威初始值（[V3 README](../dataset_definition/v3/README.md)）。

### 5.3 Teleop/executor 设计

采集接口要能**同时**控制底盘和双臂，并提供明确的 phase/mode 或交互事件按钮；如果第一阶段使用 oracle force/open policy，仍要记录它发出的 command、控制器 target 和实际 readback，且将 oracle 身份写入 metadata。Mobile ALOHA 的全身 teleop 以及 MoMaRT 的底盘+手臂同步控制说明了这种接口的必要性（[Mobile ALOHA](https://arxiv.org/abs/2401.02117)，[MoMaRT](https://sites.google.com/view/il-for-mm/home)）。

操作者视角建议以机器人 onboard/egocentric camera 为主，同时可另存外部监控视频；不要只用上帝视角完成操作，否则模型学不到部分可观测下的接近、找把手和验证。若使用 privileged GT 作为 executor 输入，要在 episode metadata 标注 `observation_privilege`，训练 policy view 时过滤掉它。

### 5.4 时钟、校准和多速率落盘

每个来源写硬件/仿真时间戳、sequence number、frame 和 receipt time；相机、joint、EE、base、FT、地图/graph 采用各自原生频率写 raw。然后生成一份固定 `train_hz` 的训练 view（建议先 5–10 Hz；仓库 full rollout 当前为 5 Hz，RT-1/LaNMP 的公开控制/采集约 3 Hz，但这些数字不是本项目必须遵循的标准）。插值规则要区分 pose、twist、离散状态和事件：事件不能用线性插值“制造”出来。

仓库 raw recording 已有 causal step-boundary/receipt 约定，应该沿用“只能使用边界之前收到的数据”的原则（[raw_recording_format.md](../raw_recording_format.md)）。

### 5.5 自动事件切分 + 人工 QA

仿真中可由 joint fraction、接触、EE—handle 距离、base 速度、可见像素、occupancy/graph 重算自动切出 `NAVIGATE/APPROACH/PRE_CONTACT/ACTUATE/VERIFY/RECOVER`；真实数据则用 teleop mode、按钮和检测器提供候选，再抽样人工修正。每条 event 保存开始/结束 step、pre/post、成功和失败原因，避免只保留“最终成功”布尔值。

### 5.6 失败数据和 split

保留导航碰撞、未抓住把手、门锁/过重、开度不足、目标仍不可见、超时、错误交互等失败轨迹，标注 `success=false` 和细分原因；训练时可把它们放进 failure/recovery 子集，评估时不能悄悄删除。按 **scene/object instance/task/operator** 做 split，避免同一门模型/同一轨迹的近重复泄漏到 test。当前 full 模式把失败 rollout 标作诊断而非 `training_eligible`，这对 benchmark 有利；若用于 VLA 研究，建议保留原始文件并另外发布 quality/success mask（[V3 README](../dataset_definition/v3/README.md)）。

### 5.7 先仿真后真实

本项目第一阶段贡献是交互图的构建/感知和规划，不是底层开门控制。建议先在 MolmoSpaces：

1. 用 V3/GT 自动生成 `required/beneficial/unnecessary` 对照和 oracle interaction；
2. 记录完整 simulator state、base/EE/joint/action/readback，验证 schema 能 replay；
3. 用渲染噪声、遮挡、感知误差和失败 executor 做 hard cases；
4. schema 稳定后再用 ROS/真实底盘采集少量 teleop，保留同样的 canonical fields 和 event/graph labels。

这条顺序能先回答“模型是否识别需要交互、是否维护状态、交互后路径是否更好”，再把真实 manipulation 控制误差作为独立变量。

## 6. 一个可落地的文件布局（示意）

不要求立即改现有 recorder；下面是对现有 V3 + full H5 的**目标契约**，可先用 sidecar 验证：

```text
episode_<id>/
  episode_spec.json                 # V3/L0，场景、任务、GT、success criteria
  trajectory.h5                     # dense L2 + raw refs
    meta/{schema_version, robot, frames, rates, calibration}
    steps/
      timestamp_ns, dt, segment, phase, terminal, truncated
      images/{camera}
      obs/{base, left_arm, right_arm, torso_head, objects}
      action_cmd/{canonical, raw, active_mask, valid_mask}
      action_measured/{base, ee, joint, gripper, contact}
      graph/{snapshot_ref, delta}
  events.jsonl                      # L1：event interval + pre/post + cost + failure
  raw/                               # L3：native-rate sensor/controller receipts
  summary.json                      # quality/success/train eligibility
```

最小 JSON 语义示例（仅示意字段，不是代码接口）：

```json
{
  "step": 42,
  "timestamp_ns": 8400000000,
  "obs": {
    "base": {"pose_se2": [1.2, 0.4, 1.57], "twist_body": [0.1, 0.0, 0.02]},
    "left_arm": {"ee_pose_base": [0.42, 0.12, 0.86, 1, 0, 0, 0], "q": "..."},
    "right_arm": null,
    "objects": {"door_17": {"joint_fraction": 0.31, "semantic_state": "ajar"}}
  },
  "action_cmd": {
    "canonical": {
      "mode": "ACTUATE", "base": [0, 0, 0],
      "left_ee": [0.004, 0, 0, 0, 0.01, 0, 0.8],
      "right_ee": [0, 0, 0, 0, 0, 0, 0],
      "target_object_id": "door_17", "primitive": "OPEN"
    },
    "active_mask": {"base": false, "left_ee": true, "right_ee": false}
  },
  "action_measured": {"base_twist": [0.0, 0.0, 0.0], "door_joint_readback": 0.34},
  "graph_delta": {"door_17": {"semantic_state": ["closed", "ajar"]}}
}
```

## 7. 最终决策清单

- **采集什么：**原生传感器 + 全身 `q/dq`（可得时加 `tau/current/contact`）+ base/EE/gripper + command/readback + scene/object/joint/graph + event/质量标签。
- **VLA 学什么：**固定 schema 的 `mode + base + 双 EE + gripper + primitive/object + mask + terminate`；优先相对差分和局部坐标。
- **全身关节放哪里：**raw/replay、低层控制和评估；不要把它作为唯一跨平台 VLA action。
- **交互导航的核心监督：**交互是否必要、对象/关节是谁、交互前后可达性/可见性/路径代价如何变化，以及失败/恢复。
- **仓库如何衔接：**保留 V3 JSON 作为 L0/GT，沿用 full H5 的时间/图像/phase 骨架；新增命名 action/state、mask、command-vs-measured、event 和 graph delta 的数据契约即可，不需要另造 `derived_task`。

## 8. 主要一手来源

- [HRL4IN: Interactive Navigation](https://arxiv.org/abs/1910.11432) / [项目页](https://sites.google.com/view/hrl4in/home)
- [Interactive Gibson](https://arxiv.org/abs/1910.14442) / [项目页](https://sites.google.com/view/interactivegibsonenv/home)
- [Open X-Embodiment / RT-X](https://arxiv.org/abs/2310.08864) / [项目页](https://robotics-transformer-x.github.io/)
- [RT-1](https://arxiv.org/abs/2212.06817) / [项目页](https://robotics-transformer1.github.io/)
- [BridgeData V2](https://proceedings.mlr.press/v229/walke23a.html) / [项目页](https://rail-berkeley.github.io/bridgedata/) / [TFDS schema](https://github.com/tensorflow/datasets/blob/master/docs/catalog/bridge.md)
- [DROID](https://arxiv.org/abs/2403.12945) / [official data schema](https://droid-dataset.github.io/droid/the-droid-dataset)
- [RH20T](https://rh20t.github.io/) / [paper PDF](https://rh20t.github.io/static/RH20T_paper_compressed.pdf)
- [Mobile ALOHA](https://arxiv.org/abs/2401.02117) / [official repo](https://github.com/MarkFzp/mobile-aloha)
- [MoMaRT](https://proceedings.mlr.press/v164/wong22a.html) / [datasets](https://sites.google.com/view/il-for-mm/datasets)
- [LaNMP](https://llhomerobots.github.io/docs/corl_llhomerobots_2024-cr_paper_16.pdf)
- [LIBERO](https://arxiv.org/abs/2306.03310) / [official repo](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [ManiSkill demonstrations](https://github.com/haosulab/ManiSkill/blob/main/docs/source/user_guide/datasets/demos.md) / [replay](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html)
- [RLDS format](https://github.com/google-research/rlds)
- 本仓库：[V3 定义](../dataset_definition/v3/README.md)、[full rollout recorder](../collection/full_rollout_recorder.py)、[标准 H5 format](../../../docs/data_format.md)、[raw recording format](../raw_recording_format.md)。

## 9. 追问：铰链交互、IK/SONIC 与仿真 VLA 数据

### 9.1 移动操作不等于搬运

搬运/抓取是公开移动操作数据中最常见、也最容易规模化的子类，但不是定义本身。Mobile ALOHA 的任务明确包括双门柜、进电梯、开水龙头和推椅子等需要底盘与手臂协同的任务（[论文](https://arxiv.org/abs/2401.02117)）。铰链门、抽屉和按钮任务属于移动操作，只是闭链接触、力控、关节约束和底盘—手臂同步使采集与验证成本更高。

### 9.2 物体把手轨迹不能直接变成任意机器人的关节轨迹

应区分三个对象：

```text
object_articulation / handle_pose
robot_ee_pose / contact_pose
robot_joint_command / measured_joint_state
```

给定门或抽屉的 articulation trajectory 和资产几何，可以先计算把手的 6D pose；若是刚性抓取，还需要固定的 grasp transform 才能得到机器人末端目标。随后必须针对每个机器人用其 URDF/MJCF、tool frame、关节限位、碰撞和当前 seed 做约束 IK/MPC/WBC。冗余机械臂通常有多个 IK 解；只给 3D position 不能决定姿态、肘部构型、夹爪状态或接触力。移动底盘未知时，base 和 arm 的分解也不是唯一的。

一个可执行的生成链是：

```text
door/drawer joint q_obj(t)
  -> handle pose/twist T_handle(t)
  -> grasp/contact transform
  -> robot-specific EE + base reference
  -> constrained IK / MPC / whole-body controller
  -> joint target and velocity
  -> physics replay + object-joint/contact readback
```

### 9.3 SONIC 的准确定位

如果这里的 Sonic 指 NVIDIA 的 SONIC/GEAR-SONIC，它不是“输入任意把手轨迹、输出任意机器人关节角”的通用 IK。官方 VLA workflow 是让 VLA 预测 64D SONIC latent motion tokens，再由 SONIC 在 50 Hz 解码为全身关节命令；SONIC 的运动参考也可来自仿真、动作捕捉或程序生成，但换机器人需要 URDF/MJCF、joint/body mapping、actuator 和 action-scale 配置（[VLA workflow](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html)，[new embodiments](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/new_embodiments.html)，[motion reference](https://nvlabs.github.io/GR00T-WholeBodyControl/references/motion_reference.html)）。

SONIC/GRAIL 中确实存在“任务空间/人体运动 → 机器人关节轨迹”的 retargeting：GMR 使用机器人 MJCF 做 IK，把 SMPL-X 运动转换为 G1 joint trajectories（[GRAIL retargeting](https://nvlabs.github.io/GRAIL/retargeting.html)）。这证明了 retargeting 路线可行，但不证明一个把手位置序列可以不经机器人模型而跨 embodiment 通用。

### 9.4 仿真生成 VLA/机器人学习数据的先例

这类数据是存在的，并不要求每一帧都由真实机器人采集：

- RoboCasa 使用 MimicGen 从少量源示范合成大规模轨迹；当前文档列出 1,615 小时的 MimicGen 合成预训练数据，并与 human target data 分开（[RoboCasa 数据总览](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html)）。
- MimicGen 官方项目展示了从少量 human demonstrations 在新场景、对象和机器人上生成大量仿真/机器人学习示范（[MimicGen](https://mimicgen.github.io/)）。
- ManiSkill 支持保存完整环境状态和 action，再重新 replay/render 出 RGB-D 等训练观测（[ManiSkill](https://github.com/haosulab/ManiSkill)，[demo/replay 文档](https://github.com/haosulab/ManiSkill/blob/main/docs/source/user_guide/datasets/demos.md)）。
- NVIDIA 的 GR00T-SONIC 数据采集器明确支持 `--sim`，在 MuJoCo 仿真中发布相机图像、记录状态/动作，并导出与真实采集相同的 LeRobot 数据结构（[VLA data collection](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html)）。

这些数据应标记 `source=simulation|retargeted|teleop_real|policy_rollout`。仅有 IK 生成的关节数组属于 motion reference，不自动等于完整 VLA episode；要成为 VLA 样本，还需要同一时间轴上的渲染图像、语言、可见 proprioception、action alignment、终止/成功和质量标签。

### 9.5 对本项目的补充建议

交互导航可以拆成两个可复用数据产品：

1. `interaction_decision`：图像/语言/可见状态 → 是否交互、哪个对象/joint、预期 graph change；这部分特别适合用 MolmoSpaces 大规模仿真生成。
2. `interaction_motion`：对象 articulation/handle reference + proprio/contact → EE/base subgoal → robot-specific joint command；这部分先用约束 IK/WBC 生成，再用物理 replay 过滤。

仿真生成轨迹必须通过 IK residual、关节/速度限位、碰撞、自碰撞、接触力、object-joint readback、最终 reachability/visibility 和路径代价检查；通过后才能标记 `training_eligible`。对真实 VLA 的最后微调仍建议保留少量真实 teleop，以覆盖外观、延迟、摩擦和执行器误差。

## 9. 补充问题：搬运、铰链交互、末端轨迹反解与仿真 VLA 数据

### 9.1 移动操作并不等于搬运，但搬运确实长期占主流

“移动操作”是 embodiment/task family，不是“把物体从 A 搬到 B”的同义词。搬运（`pick → move → place`）之所以常见，是因为抓取、物体状态和成功条件比较容易定义，且底盘和手臂可以在较长时间段内近似解耦。早期许多移动操作工作确实主要集中在 pick-move-place；但这更多是数据和硬件成本的历史结果，而不是概念边界。

铰链/滑轨对象是移动操作中非常典型、也更难的一类：门、柜门、抽屉、冰箱、烤箱等都要求保持接触并遵守一条受约束的末端轨迹。Open-World Mobile Manipulation 明确把过去主要的 pick-move-place 视为移动操作的一小部分，并以真实门、柜、抽屉和冰箱为主要研究对象；其训练集包含门、抽屉和柜体，测试覆盖不同建筑中的新对象（[论文](https://arxiv.org/abs/2401.14403)）。MolmoBot 的移动 RB-Y1 任务也直接包含 door opening、drawer/cabinet interaction，而不是只做搬运（[技术报告](https://arxiv.org/abs/2603.16861)）。

因此更准确的说法是：

- 大规模通用 VLA 数据中，固定底座的抓取/放置仍占大多数；
- 移动操作的公开系统往往先从搬运开始，但门/抽屉是重要的能力缺口和测试集；
- 铰链任务不是“非移动操作”，而是约束更强、对接触动力学和底盘协同要求更高的移动操作子类；
- 对交互导航而言，开门/开抽屉还要额外记录交互是否改变了可达性、可见性或路径代价，这使它比单纯的 articulated-object manipulation 多一层导航因果标签。

### 9.2 末端轨迹能否反推出任意机器人的关节轨迹？

可以把它作为一个**机器人相关的轨迹生成问题**，但不能把它理解为“有一个门/抽屉末端位置，就自动得到任何机器人的关节变化”。需要先区分三个轨迹：

1. **物体轨迹**：门把手、抽屉把手或物体末端随物体关节变化的轨迹；
2. **机器人末端目标轨迹**：机器人夹爪为了保持抓取/接触而应跟踪的位姿；
3. **机器人关节轨迹**：某一具体机器人在给定底盘、姿态和约束下的一组 `q(t)`。

对一个 1-DoF 对象，物体末端轨迹可以由关节变量和几何直接生成：

```text
revolute door:  handle_pose(t) = T_hinge * Rot(axis, theta(t)) * T_handle
prismatic drawer: handle_pos(t) = p0 + axis_hat * d(t)
```

如果夹爪和把手之间是刚性抓取，还需要一个抓取变换 `T_grasp`：

```text
T_ee_world(t) = T_handle_world(t) * T_grasp_handle
```

只有物体把手的**位置**通常不够；至少还需要末端姿态，或者明确的接触法向、切向和绕轴姿态约束。若夹爪有柔顺性、把手有间隙或处于推接触而非刚性抓取，`T_grasp` 也不是常数，必须把接触状态/力作为额外变量。

这个思路有直接的公开先例。`Predicting Motion Plans for Articulating Everyday Objects` 构建了 ArtObjSim，并以门、抽屉、马桶盖等对象的末端轨迹作为约束；其 SeqIK 用上一时刻的 IK 解 warm-start 下一时刻，输出连续关节轨迹。ArtObjSim 包含 3,758 个对象实例、97 个真实场景，覆盖 prismatic、vertical hinge 和水平铰链（[论文与数据页](https://arjung128.github.io/mpao/)，[论文 PDF](https://arxiv.org/pdf/2303.01484.pdf)）。真实 Stretch 系统也将一条开门/抽屉轨迹表示为 10 个末端 waypoint，再由 whole-body planner 尝试求关节角（[Opening Cabinets and Drawers](https://embodied-ai.org/papers/2024/12_Opening_Cabinets_and_Drawer.pdf)）。

但是，从末端轨迹到关节轨迹至少有以下限制：

- **必须知道机器人模型。**需要 URDF/MJCF、关节顺序和零位、末端工具变换、底盘位姿、关节范围和速度/加速度限制；没有这些信息不存在“对任意机器人”的反解。
- **存在多解、无解和分支跳变。**7-DoF 手臂对一个 6D 位姿通常有冗余；同一末端轨迹可以对应不同肘部构型。逐帧独立求 IK 很容易发生 elbow flip 或关节跳变，应使用上一帧解 warm-start，并在整段轨迹上优化连续性、姿态偏好、可操作度和碰撞代价。
- **移动底盘使问题变成 whole-body IK。**底盘 `SE(2)`、躯干、手臂和头部可能共同承担轨迹；固定底盘求出的手臂解不一定在真实移动机器人上可达。对门的拉开动作，底盘还可能需要后退、侧移或绕铰链移动。
- **位姿可达不等于能开门。**普通 IK 只约束几何位置/姿态，不能保证足够的抓取力、摩擦、闩锁释放、门的惯量、抽屉导轨摩擦、底盘稳定性或碰撞安全。必须把求得的轨迹放回物理仿真/闭环控制器中验证。
- **物体末端不等于机器人末端。**把手的圆弧是物体几何轨迹；夹爪可能相对把手有固定偏置、转动约束或柔顺偏差。只记录门尖端或抽屉前端，会丢掉真正的接触点和工具姿态。
- **时序和力也不能由位置唯一决定。**同一空间轨迹可以快拉、慢拉、停顿、重新抓取或施加不同力；若要训练接触策略，还需记录速度、加速度、接触开始/结束、力/力矩、夹爪开合和控制器延迟。

一个更准确的优化形式是对每个具体机器人求：

```text
q*(t) = argmin_q  pose_error(FK_robot(q), T_ee_target(t))
                    + lambda * ||q(t)-q(t-1)||^2
                    + collision_cost + joint_limit_cost + posture_cost
```

对移动机器人，`q` 应包含底盘、躯干和手臂；还要加入速度、加速度、接触和动力学约束。输出时应同时保存 `T_ee_target`、`q_ik`、IK residual、最小碰撞距离、关节限位裕量、可操作度和物理执行结果。这样一条对象空间轨迹可以作为跨机器人共享的**中间表示**，但每个机器人仍需单独的 feasibility/retargeting adapter。

建议的生成流水线是：

```text
object joint path q_obj(t)
  -> handle pose / articulation geometry
  -> contact frame + grasp transform
  -> desired EE pose/twist
  -> robot-specific whole-body IK / trajectory optimization
  -> closed-loop physics rollout with friction, latch, noise, delay
  -> retain trajectory only with explicit feasibility and success labels
```

数据字段中应把这些层分开，而不是只留下最后的关节数组：

```text
object_joint_state
handle_pose_world
contact_frame / grasp_transform
ee_target_pose / ee_target_twist
ik_joint_target
joint_measured / base_measured
contact_force / torque / grasp_state
ik_residual / collision / limit_margin / manipulability
physics_success / failure_reason
```

### 9.3 “Sonic”若指 NVIDIA GEAR-SONIC，它不是通用 IK 反解器

这里对用户所说的 “sonic” 做一个明确假设：很可能是 NVIDIA 的 **GEAR-SONIC / SONIC**。如果指的是另一个同名模型，应以具体链接或仓库为准。

SONIC 的核心是**全身运动跟踪和控制**，不是输入任意物体末端轨迹、输出任意机器人关节角的通用逆运动学器：

- SONIC 论文的 tracking policy 以机器人自身的关节 pose/velocity、根部状态和 motion command 为状态，并输出目标关节位置，由各关节 PD 跟踪；
- 官方 reference-motion 文档要求为目标机器人准备 robot-specific 的 `joint_pos.csv`、`joint_vel.csv`、root `body_pos/body_quat` 等文件，示例是 Unitree G1 的 29 个关节；
- 三点 VR 接口可以输入头部和双腕的 SE(3)、手指关节、腰高和导航命令，再由运动规划器生成下半身；这是一种固定 embodiment 的接口，不是跨机器人零配置映射；
- SONIC 的 VLA workflow 中，VLA 预测 64D motion token（另有双手关节），SONIC 在约 50 Hz 解码为全身关节命令；三点接口的 VLA 则输出与该接口相同的上身位姿和导航信号。

参见 [SONIC 论文](https://arxiv.org/html/2511.07820v4)、[官方 motion reference 格式](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/references/motion_reference.md) 和 [VLA workflow](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/tutorials/vla_workflow.md)。

因此，若想让 SONIC 参与门/抽屉数据生成，合理的接法是：

```text
object/articulation trajectory
  -> robot-specific EE / whole-body IK (or learned adapter)
  -> G1-compatible joint/reference motion or SONIC latent
  -> SONIC tracking controller
```

而不是：

```text
door tip position -> SONIC -> arbitrary robot joint positions
```

SONIC 的优势在于把已知 embodiment 上的全身协调、平衡和运动平滑交给一个低层控制器；它不能替代门把手检测、抓取变换估计、铰链约束识别、跨形态 IK 或接触力控制。尤其不能直接把 SONIC 的 G1 joint/token action 当作 RB-Y1、Stretch 或其他机器人的通用 action。

### 9.4 已有的仿真/合成 VLA 或 robot-learning 数据先例

“VLA 数据通常来自真实 teleoperation”是主流印象，但并不是唯一范式。需要区分：完全由仿真 expert rollout 生成、由少量真实 demo 在仿真中扩增、由人体动作 retarget 得到的运动数据，以及只用于 articulation perception/planning 的轨迹数据。

| 先例 | 合成方式 | 是否包含门/抽屉 | 对本项目的直接启示 |
|---|---|---:|---|
| **MolmoBot-Engine / MolmoBot-Data** | MuJoCo + MolmoSpaces 程序化场景、动作/相机/动力学随机化；scripted expert 分阶段规划，门用铰链圆弧 waypoint、抽屉用滑轨直线 waypoint，再用 IK/trajectory optimization（RB-Y1 使用 CuRobo）执行 | **是** | 最接近本项目的公开先例。报告版约 1.7M expert trajectories；RB-Y1 的 door-open 约 79k、articulated-object open 约 46.6k。每条轨迹同时保存 joint absolute/relative、EE twist/pose、base pose/velocity、任务状态和图像；移动任务训练使用 joint delta + base velocity，避免部署时再做 IK。见 [技术报告](https://arxiv.org/abs/2603.16861)、[数据集](https://huggingface.co/datasets/allenai/molmobot-data)。 |
| **ArtObjSim + SeqIK** | 从真实 HM3D 场景中的 2D/3D articulation 标注建立轻量仿真；给定对象末端 waypoint 和初始 `theta0`，顺序 warm-start IK 生成机器人关节计划 | **是** | 直接证明“对象/末端轨迹 → 机器人轨迹”可用于大规模规划数据，但 `theta0`、机器人模型、碰撞和连续 IK 仍是每个 embodiment 的条件；它主要是 motion-planning 数据，不是完整语言 VLA。见 [项目页](https://arjung128.github.io/mpao/)。 |
| **RoboCasa + MimicGen** | RoboCasa 在仿真厨房提供 human 与自动生成数据；MimicGen 用少量 source demonstrations 把 object-centric subtask 迁移到新对象、场景和初始状态；MimicGen 从约 200 个真实/源 demo 生成 50K+ trajectories across 18 tasks | **是** | 适合把“接近、抓取、开门/抽屉、后续操作”切成可组合 skill；但它是 source-demo 扩增，不等于完全无真实先验。见 [RoboCasa 数据说明](https://github.com/robocasa/robocasa/blob/main/docs/datasets/using_datasets.md)、[MimicGen 论文](https://arxiv.org/abs/2310.17596)。 |
| **ManiSkill / SAPIEN、Where2Act、FlowBot3D** | PartNet-Mobility articulated assets + 物理仿真；通过 scripted/探索式交互得到 RGB-D、点云、actionability、push/pull trajectory 和 robot state | **是** | 适合学习铰链/滑轨几何、可交互点和物体状态转移；多数工作目标是 affordance/planning，不是面向自然语言的长时域 VLA，因此需补充语言、导航和 event labels。见 [Where2Act](https://arxiv.org/abs/2101.02692)、[FlowBot3D](https://arxiv.org/abs/2205.04382)。 |
| **GraspVLA / SynGrasp-1B** | 大规模光照/物体/位姿随机化，完全仿真生成 grasp action 数据，训练 VLA 预训练 | 否（主要是 grasp） | 证明 synthetic action data 可以作为 VLA foundation pretraining；但抓取的几何接触比门/抽屉的长时域受约束动力学简单，不能直接替代 articulated interaction 数据。见 [论文](https://arxiv.org/abs/2505.03233)。 |
| **SONIC / BONES-SEED** | 人体 mocap 经 retargeting 得到 G1 robot motion，再在 Isaac Lab/MuJoCo 做 physics-based motion tracking；官方数据包含约 142K motions、约 288 小时 | 不是门/抽屉专门数据 | 是“运动 retarget + 仿真训练”的先例，而不是 object-endpoint 到任意机器人的通用 VLA 数据。其 VLA demo 仍通过 VR/teleoperation 接口采集；低层 SONIC policy 需要 G1-specific motion representation。见 [training data](https://nvlabs.github.io/GEAR-SONIC/user_guide/training_data.html)。 |
| **NVIDIA PhysicalAI Robotics Manipulation Kitchen** | 自动生成的 LeRobot 数据，包含 open/close cabinet、dishwasher、fridge、drawer 等任务，动作模态含双臂、夹爪、头/躯干和轮子 | **是** | 是公开的 synthetic articulated-kitchen 数据格式先例；但其 action contract 与具体平台绑定，不能仅凭“同为 34D”就视为跨机器人通用。见 [数据集卡](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen)。 |

MolmoBot 是目前与本项目最值得对照的先例：它不是把仿真中的门关节直接 teleport 到目标值，而是为门/抽屉生成阶段化的末端 waypoint，经过 IK/碰撞规划后在 MuJoCo 中执行，并把多种 action representation 一起落盘。其报告明确指出，RB-Y1 的 base 是 3-DoF holonomic planar joint，移动操控模型使用 joint delta 和 base velocity；同时保存 commanded joint、EE twist/pose 和 base/object/task state。这说明“末端轨迹作为中间监督 + 机器人关节作为执行动作”是可行的，但也说明最终训练 action 不能脱离目标机器人和执行器。

### 9.5 针对 MolmoSpaces 交互导航数据的建议

建议把对象轨迹提升为一种**可复用的 task-space interaction layer**，并为每个 robot 生成独立的执行层：

```text
共享层（robot-independent）
  articulation_type, axis, joint_scalar(t), handle/contact frame,
  intended_effect, target_object, phase, navigation/graph delta

机器人层（robot-specific）
  base/torso/arm q(t), dq(t), controller target,
  EE target/readback, gripper, contact force,
  IK/collision/limit feasibility and physics outcome

VLA view
  image + language + proprio
    -> mode / object / primitive / base + EE or joint-delta action chunk
```

可按下面顺序实施：

1. **先用几何生成，不要先训练 Sonic。**从 door/drawer 的 joint scalar 和 articulation geometry 生成 handle pose；固定一个或多个 grasp/contact frame，得到 EE target trajectory。
2. **每个机器人单独求 whole-body trajectory。**至少为 RB-Y1、目标真实机器人各求一次 IK/trajectory optimization，并记录失败原因；不可达的轨迹不要用零填充成“成功样本”。
3. **用物理 rollout 过滤。**加入关节限位、碰撞、摩擦、闩锁/导轨阻尼、控制延迟和动作噪声；检查实际 object joint progress、contact、base drift 和最终 navigation effect。
4. **同时导出三种 view。**`object/EE task-space` 供跨机器人表示和辅助监督，`joint+base` 供当前 embodiment 的 VLA/behavior cloning，`raw readback/force` 供低层控制和失败分析。
5. **对交互导航保留非交互对照。**同一场景/目标生成 `required`、`beneficial`、`unnecessary` 及失败/错误交互，避免模型仅学习“看见门就开”。
6. **把 SONIC 放在 executor 层。**如果目标是 G1，先把 door/drawer task-space trajectory retarget 为 G1-compatible reference，再研究 VLA → SONIC latent；如果目标是 RB-Y1，不应直接复用 G1 的 SONIC joint/token interface。

最终建议可以概括为：**对象末端轨迹适合做跨机器人共享的约束/中间标签，不适合单独作为完整 action；IK/retargeting 能生成候选关节轨迹，但必须依赖具体机器人并经过 whole-body、碰撞和物理验证；仿真生成 VLA 数据已有成功先例，尤其是 MolmoBot，但其成功关键是大规模场景/动力学随机化、真实执行语义和严格质量过滤，而不是单纯把一条几何曲线转换成关节数组。**

## 10. 本节新增来源

- [Adaptive Mobile Manipulation for Articulated Objects In the Open World](https://arxiv.org/abs/2401.14403)
- [Predicting Motion Plans for Articulating Everyday Objects / ArtObjSim](https://arjung128.github.io/mpao/) / [论文 PDF](https://arxiv.org/pdf/2303.01484.pdf)
- [Opening Cabinets and Drawers in the Real World](https://embodied-ai.org/papers/2024/12_Opening_Cabinets_and_Drawer.pdf)
- [Opening Articulated Objects in the Real World](https://arxiv.org/abs/2402.17767)
- [MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation](https://arxiv.org/abs/2603.16861) / [MolmoBot-Data](https://huggingface.co/datasets/allenai/molmobot-data)
- [RoboCasa dataset documentation](https://github.com/robocasa/robocasa/blob/main/docs/datasets/using_datasets.md)
- [MimicGen](https://arxiv.org/abs/2310.17596)
- [Where2Act](https://arxiv.org/abs/2101.02692) / [FlowBot3D](https://arxiv.org/abs/2205.04382)
- [GraspVLA](https://arxiv.org/abs/2505.03233)
- [SONIC](https://arxiv.org/html/2511.07820v4) / [motion reference](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/references/motion_reference.md) / [VLA workflow](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/docs/source/tutorials/vla_workflow.md)
- [NVIDIA PhysicalAI Robotics Manipulation Kitchen](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen)
