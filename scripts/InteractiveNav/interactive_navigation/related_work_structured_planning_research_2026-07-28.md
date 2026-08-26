# 结构化表示、动态交互图、TAMP 与 VLM/LLM 规划：primary-source 核验

**调研日期：** 2026-07-28
**用途：** 为 `interactive-nav-paper/Sections/2_related_work.tex` 和 `Sections/4_method.tex` 提供可核验的候选文献。
**来源原则：** 优先使用论文原文、出版社开放页面、会议论文集或作者项目页；仅将二手索引页用于交叉核对书目信息，不据此单独支撑技术结论。

## 先给结论

当前 `main.bib` 已有 Armeni et al. (2019) 3D Scene Graph、Hydra (2022)、ConceptGraphs (2023) 和 HOV-SG (2024)。这些可作为静态/层次化/开放词汇表示的已有锚点，不建议重复添加。

本文最需要补上的不是更多“语义地图”论文，而是以下三条连续证据链：

1. **可行动和动态表示。** Rosinol et al. 的 3D Dynamic Scene Graphs 已把 traversability 与时空关系放进图中，但主要针对人和动态实体，未把门、抽屉、冰箱等铰接状态作为会改变导航边和目标可见性的显式转移。
2. **功能部件、容器和交互后更新。** FunGraph、MomaGraph、Pandora、MoMa-SG 分别推进了部件级 affordance、状态感知关系、铰接对象和容器内容建模；它们是本文“container-mediated access”与“state-conditioned graph”最接近的对照，必须在投稿截止日前核对版本日期。
3. **图到可执行规划。** 经典 TAMP（HTAMP、FFRob、PDDLStream、LGP）提供离散动作与连续可行性之间的接口；SayPlan、Optimal Scene Graph Planning、DELTA 等展示了 LLM 如何从图或语言产生高层计划，但通常不以 ObjectGoal 导航中的“是否必须交互/交互后继续导航”作为独立评估维度。

## 25 条核心候选

### A. 结构化、动态和可行动场景图（12 条）

| ID | 准确书目信息与 primary source | 核心贡献（据原文概括） | 建议放置与本文差异 |
|---|---|---|---|
| A1 | Antoni Rosinol, Arjun Gupta, Marcus Abate, Jingnan Shi, Luca Carlone. **“3D Dynamic Scene Graphs: Actionable Spatial Perception with Places, Objects, and Humans.”** *Robotics: Science and Systems (RSS)*, 2020. [论文](https://www.roboticsproceedings.org/rss16/p079.pdf), [arXiv:2002.06289](https://arxiv.org/abs/2002.06289), DOI `10.15607/RSS.2020.XVI.079` | 提出分层 DSG；边可表达 places/rooms 的 connectivity、时空关系和 traversability，并由 SPIN 从视觉惯性数据增量构建。 | **Related Work—structured representations** 的必引。指出其 dynamic/actionable 图仍主要描述人、物和拓扑，未显式表示铰接对象状态转移及其对目标可见性的影响。 |
| A2 | Shun-Cheng Wu, Johanna Wald, Keisuke Tateno, Nassir Navab, Federico Tombari. **“SceneGraphFusion: Incremental 3D Scene Graph Prediction from RGB-D Sequences.”** *CVPR*, 2021, 7515–7525. [CVPR open access](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_SceneGraphFusion_Incremental_3D_Scene_Graph_Prediction_From_RGB-D_Sequences_CVPR_2021_paper.pdf) | 用 GNN 融合 RGB-D 子图，增量合并实例并处理部分/缺失图数据，强调在线、全局一致的图更新。 | 放在“incremental graph construction”段。它解决感知层的增量融合，不定义“执行交互后哪些导航关系失效/恢复”。 |
| A3 | Zachary Ravichandran, Lisa Peng, Nathan Hughes, J. Daniel Griffith, Luca Carlone. **“Hierarchical Representations and Explicit Memory: Learning Effective Navigation Policies on 3D Scene Graphs using Graph Neural Networks.”** *ICRA*, 2022, 9272–9279. [arXiv:2108.01176](https://arxiv.org/abs/2108.01176), [DOI](https://doi.org/10.1109/ICRA46639.2022.9812179) | 将层次化 3D 场景图编码为 agent-centric 特征，保留轨迹记忆并学习 Object Search 导航策略。 | 放在“scene graph for navigation”段。与本文相同点是图驱动导航，不同点是其图和策略不包含显式 interaction necessity、door/container effects 或交互后重规划。 |
| A4 | Dominic Maggio, Yun Chang, Nathan Hughes, Matthew Trang, J. Daniel Griffith, Carlyn Dougherty, Eric Cristofalo, Lukas Schmid, Luca Carlone. **“Clio: Real-time Task-Driven Open-Set 3D Scene Graphs.”** *IEEE Robotics and Automation Letters*, 9(10):8921–8928, 2024. [作者代码页/书目信息](https://github.com/MIT-SPARK/Clio), [arXiv:2404.13696](https://arxiv.org/abs/2404.13696), DOI `10.1109/LRA.2024.3451395` | 用 Information Bottleneck 和增量聚类，按自然语言任务选择应保留的对象/区域粒度，并在线构建紧凑开放集图。 | 放在“task-driven representation”段。它按任务压缩图，但没有把动作的后果建模为 connectivity/visibility 的状态转移。 |
| A5 | Sebastian Koch, Narunas Vaskevicius, Mirco Colosi, Pedro Hermosilla, Timo Ropinski. **“Open3DSG: Open-Vocabulary 3D Scene Graphs from Point Clouds with Queryable Objects and Open-Set Relationships.”** *CVPR*, 2024. [CVPR open access](https://openaccess.thecvf.com/content/CVPR2024/papers/Koch_Open3DSG_Open-Vocabulary_3D_Scene_Graphs_from_Point_Clouds_with_Queryable_CVPR_2024_paper.pdf), [arXiv:2402.12259](https://arxiv.org/abs/2402.12259) | 从点云预测开放词汇对象和开放集关系，并允许用语言查询对象/关系。 | 放在“open-vocabulary graph”段。它增强关系查询和 affordance 推理，但关系仍是观测/静态语义关系，不是执行动作后的图变换。 |
| A6 | Hang Yin, Xiuwei Xu, Zhenyu Wu, Jie Zhou, Jiwen Lu. **“SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation.”** *NeurIPS*, 2024. [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/098491b37deebbe6c007e69815729e09-Paper-Conference.pdf), [arXiv:2410.08189](https://arxiv.org/abs/2410.08189) | 在线构建对象—组—房间层次图，用层次化 prompt 和重感知机制做 zero-shot ObjectNav。 | 放在“LLM + graph navigation”段。与本文共享 ObjectGoal/图推理背景，但 SG-Nav 的动作空间是移动和探索，没有开门/开容器的状态改变。 |
| A7 | Muhammad Fadhil Ginting, Sung-Kyun Kim, David D. Fan, Matteo Palieri, Mykel J. Kochenderfer, Ali-akbar Agha-Mohammadi. **“SEEK: Semantic Reasoning for Object Goal Navigation in Real World Inspection Tasks.”** *RSS*, 2024. [RSS paper](https://www.roboticsproceedings.org/rss20/p024.pdf), [arXiv:2405.09822](https://arxiv.org/abs/2405.09822) | 维护 Dynamic Scene Graph 与 Relational Semantic Network，用先验语义概率指导 ObjectGoal 搜索，并在真实机器人验证。 | 放在“semantic priors for ObjectGoal”段。它优化搜索顺序，不把交互对象作为改变可达性/可见性的 action-conditioned edge。 |
| A8 | Daniel Honerkamp, Martin Büchner, Fabien Despinoy, Tim Welschehold, Abhinav Valada. **“Language-Grounded Dynamic Scene Graphs for Interactive Object Search with Mobile Manipulation.”** *IEEE Robotics and Automation Letters*, 9(10):8298–8305, 2024. [arXiv:2403.08605](https://arxiv.org/abs/2403.08605), [DOI](https://doi.org/10.1109/LRA.2024.3441495) | MoMa-LLM 将开放词汇动态场景图与对象中心动作空间交织，边探索边更新图，在交互搜索中用 LLM 规划。 | **最接近的导航/图/LLM 对照之一。** 需明确其目标是 interactive object search，本文则把 channel reachability、container visibility、required/unnecessary episode labels 和 ObjectGoal terminal success 分开评估。 |
| A9 | Dennis Rotondi, Fabio Scaparro, Hermann Blum, Kai O. Arras. **“FunGraph: Functionality Aware 3D Scene Graphs for Language-Prompted Scene Interaction.”** arXiv:2503.07909, 2025. [论文](https://arxiv.org/abs/2503.07909) | 将 handles、knobs、buttons 等 affordance-relevant parts 作为图节点，加入 intra-object 关系并支持功能部件查询。 | 放在“functional/part-level scene graphs”段。它解决“找哪个可操作部件”，本文还需解决“该交互是否改变导航任务、何时值得执行”。 |
| A10 | Yuanchen Ju, Yongyuan Liang, Yen-Jen Wang, Nandiraju Gireesh, Yuanliang Ju, Seungjae Lee, Qiao Gu, Elvis Hsieh, Furong Huang, Koushil Sreenath. **“MomaGraph: State-Aware Unified Scene Graphs with Vision-Language Model for Embodied Task Planning.”** arXiv:2512.16909, 2025. [论文](https://arxiv.org/abs/2512.16909) | 将 spatial-functional relations、part-level elements 和状态更新统一到 task-specific graph，并用 VLM 进行 Graph-then-Plan；包含 precondition/effect reasoning。 | **最接近本文“状态感知图”表述的近期工作。** 但其重点是 household manipulation/task planning，尚未以导航终点为主线比较 channel、container、mixed 和 unnecessary interaction。提交时需确认该版本是否属于允许的文献截止日期。 |
| A11 | Alan Yu, Yun Chang, Christopher Xie, Luca Carlone. **“Pandora: Articulated 3D Scene Graphs from Egocentric Vision.”** arXiv:2603.28732, 2026. [论文](https://arxiv.org/abs/2603.28732) | 从人类第一视角数据恢复铰接部件、物体动态和 object-container 关系，并将图用于移动机器人检索被遮蔽目标。 | **与 container-mediated access 高度接近。** 应明确本文不是重新估计关节运动或构建 articulated graph，而是研究已知交互状态如何进入 ObjectGoal 导航评估。属 2026 预印本，需核对截稿日期。 |
| A12 | Martin Büchner, Adrian Röfer, Tim Engelbracht, Tim Welschehold, Zuria Bauer, Hermann Blum, Marc Pollefeys, Abhinav Valada. **“Articulated 3D Scene Graphs for Open-World Mobile Manipulation.”** arXiv:2602.16356, 2026. [论文](https://arxiv.org/abs/2602.16356), [项目页](https://momasg.cs.uni-freiburg.de) | MoMa-SG 从 RGB-D 交互片段估计 revolute/prismatic articulation，将容器与 contained objects 纳入语义—运动图，并在 HSR/Spot 上执行开关/取物。 | **必须做最新 closest-work 核对。** 它侧重语义—运动建模和底层开合；本文贡献应定位为 benchmark/task-level 的 required-vs-unnecessary interaction、导航可达性与目标可见性评估，而非 articulation estimation。属 2026 预印本。 |

### B. TAMP、图编辑和可执行规划（7 条）

| ID | 准确书目信息与 primary source | 核心贡献（据原文概括） | 建议放置与本文差异 |
|---|---|---|---|
| B1 | Leslie Pack Kaelbling, Tomás Lozano-Pérez. **“Hierarchical Task and Motion Planning in the Now.”** *ICRA*, 2011, 1470–1477. [MIT PDF](https://people.csail.mit.edu/tlp/pdf/2011/hpnICRA11Final.pdf), [DOI](https://doi.org/10.1109/ICRA.2011.5980391) | 用激进的层次化、由上而下承诺，把离散任务选择和连续几何规划结合，并减少对远期动作效果投影的需求。 | TAMP 背景段的基础引用。本文可说明交互图是对 TAMP 状态/动作接口的导航特化，而不是重新提出通用 TAMP。 |
| B2 | Marc Toussaint. **“Logic-Geometric Programming: An Optimization-Based Approach to Combined Task and Motion Planning.”** *IJCAI*, 2015, 1930–1936. [IJCAI paper](https://www.ijcai.org/Proceedings/15/Papers/274.pdf) | 将符号 action sequence 作为非线性轨迹优化约束，利用几何代价指导离散搜索。 | 用于说明 interaction cost、几何可行性和动作序列可以统一；本文只在导航层使用交互代价，不承担低层轨迹优化。 |
| B3 | Caelan Reed Garrett, Tomás Lozano-Pérez, Leslie Pack Kaelbling. **“FFRob: Leveraging Symbolic Planning for Efficient Task and Motion Planning.”** *The International Journal of Robotics Research*, 37(1):104–136, 2018. [IJRR](https://journals.sagepub.com/doi/10.1177/0278364917739114), [arXiv:1608.01335](https://arxiv.org/abs/1608.01335) | 以 Extended Action Specification、启发式符号搜索和可条件化 roadmap 处理几何/运动约束；实验包括 rearrangement 与 NAMO。 | 适合放在“classical TAMP and movable obstacles”段。相比本文，它把交互作为 TAMP 中的操作变量，而不研究 episode-level interaction necessity 标签。 |
| B4 | Caelan Reed Garrett, Tomás Lozano-Pérez, Leslie Pack Kaelbling. **“PDDLStream: Integrating Symbolic Planners and Blackbox Samplers via Optimistic Adaptive Planning.”** *ICAPS*, 2020, 440–448. [AAAI/ICAPS record](https://doi.org/10.1609/icaps.v30i1.6739), [arXiv:1802.08705](https://arxiv.org/abs/1802.08705) | 以 streams 声明黑盒采样器，把 IK、碰撞、可见性等连续约束接入 PDDL，并用 optimistic adaptive planning 平衡搜索和采样。 | 方法部分可作为“existing policies/planners execute selected interaction” 的技术背景；不要把本文描述成 PDDLStream 的新算法。 |
| B5 | Ziyuan Jiao, Yida Niu, Zeyu Zhang, Song-Chun Zhu, Yixin Zhu, Hangxin Liu. **“Sequential Manipulation Planning on Scene Graph.”** *IROS*, 2022. [作者页](https://yzhu.io/publication/tamp2022iros/), [arXiv:2207.04364](https://arxiv.org/abs/2207.04364), DOI `10.1109/IROS47612.2022.9981735` | 提出 contact graph+，用 Graph Editing Distance 生成动作编辑，再施加时序可行性约束，处理开柜门、摆放和遮挡依赖。 | **直接支撑“交互是图编辑/状态转移”这一论点。** 但其目标是 manipulation arrangement，本文把编辑结果投射到 connectivity/visibility/accessibility 并评估导航终点。 |
| B6 | Aaron Ray, Christopher Bradley, Luca Carlone, Nicholas Roy. **“Task and Motion Planning in Hierarchical 3D Scene Graphs.”** *ISRR*, 2024（预印本版本 arXiv:2403.08094）。[论文](https://arxiv.org/abs/2403.08094), [ISRR PDF](https://groups.csail.mit.edu/rrg/papers/cbradley_isrr_2024.pdf) | 从层次化 3D 图构建稀疏 TAMP 问题，在规划期间增量加入相关对象，减少大场景中无关元素的计算。 | **最适合连接本文 graph 与 planner。** 本文可以借鉴“task-relevant subgraph / incremental grounding”，但需强调自身交互图的状态转移和 benchmark diagnosis。 |
| B7 | Zhirui Dai, Arash Asgharivaskasi, Thai Duong, Shusen Lin, Maria-Elizabeth Tzes, George J. Pappas, Nikolay Atanasov. **“Optimal Scene Graph Planning with Large Language Model Guidance.”** *ICRA*, 2024, 14062–14069. [项目页与 BibTeX](https://existentialrobotics.org/LLM-Scene-Graph-LTL-Planning/), [arXiv:2309.09182](https://arxiv.org/abs/2309.09182), DOI `10.1109/ICRA57147.2024.10610599` | 将自然语言转为 LTL automaton，在层次化 scene graph 上用 LLM heuristic 加速，同时用一致的 LTL heuristic 保持最优性。 | 可作为“LLM guidance 不等于无约束执行”的强对照。本文的差异是交互必要性和交互后导航状态，而不是 LTL 最优规划本身。 |

### C. VLM/LLM 驱动的机器人规划与闭环（6 条）

| ID | 准确书目信息与 primary source | 核心贡献（据原文概括） | 建议放置与本文差异 |
|---|---|---|---|
| C1 | Michael Ahn et al. **“Do As I Can, Not As I Say: Grounding Language in Robotic Affordances.”** *CoRL*, 2022, 287–318. [PMLR](https://mlanthology.org/corl/2022/ichter2022corl-say/), [arXiv:2204.01691](https://arxiv.org/abs/2204.01691) | 用 LLM 提供高层语义知识，再以 skill value/affordance 模型约束机器人可执行性，完成长时域移动操作。 | 放在“LLM chooses among existing skills”段。本文同样保留已有 interaction policy，但额外要求判断 interaction necessity、candidate role 和状态更新。 |
| C2 | Wenlong Huang et al. **“Inner Monologue: Embodied Reasoning through Planning with Language Models.”** *CoRL*, 2022, 1769–1782. [PMLR](https://proceedings.mlr.press/v205/huang23c/huang23c.pdf), [arXiv:2207.05608](https://arxiv.org/abs/2207.05608) | 将 success detector、scene description 和人类反馈作为语言反馈注入 LLM，形成闭环重规划。 | **支撑本文 outcome verification/replanning 的先例。** 区别在于本文把验证结果写回结构化交互图，并用图的 connectivity/visibility 变化驱动导航重规划。 |
| C3 | Krishan Rana, Jesse Haviland, Sourav Garg, Jad Abou-Chakra, Ian Reid, Niko Sünderhauf. **“SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning.”** *CoRL*, 2023. [项目页与 BibTeX](https://sayplan.github.io/), [arXiv:2307.06135](https://arxiv.org/abs/2307.06135) | 利用层次图做 semantic subgraph search，将高层 LLM 计划与经典路径规划结合，并通过 scene-graph simulator 迭代修正不可执行动作。 | **最适合放在 closest LLM+scene-graph work。** 本文可明确沿用其“图检索 + 迭代验证”思想，但研究对象是 ObjectGoal navigation 中的 channel/container state change，而非通用 household task plan。 |
| C4 | Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, Animesh Garg. **“ProgPrompt: Generating Situated Robot Task Plans using Large Language Models.”** *ICRA*, 2023, 11523–11530. [项目页](https://progprompt.github.io/), [arXiv:2209.11302](https://arxiv.org/abs/2209.11302), DOI `10.1109/ICRA48891.2023.10161317` | 用程序化 prompt、动作/对象 schema、assertions 和 recovery actions 约束计划生成，避免自由文本动作不可执行。 | 用于说明 schema-constrained decision 的先例。本文可将 decision output 进一步限制为 interaction candidate、necessity/abstain 和 bounded rationale。 |
| C5 | Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, Li Fei-Fei. **“VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models.”** *CoRL*, 2023, 540–562. [PMLR/论文](https://mlanthology.org/corl/2023/huang2023corl-voxposer/), [arXiv:2307.05973](https://arxiv.org/abs/2307.05973) | LLM 推断 affordance/约束，VLM 负责 grounding，组合 3D value maps，再用 model-based planner 生成闭环轨迹。 | 放在“低层 grounding/执行”段。它说明本文无需重做 controller；本文贡献在 interaction-level decision and graph update，而非轨迹生成。 |
| C6 | Yuchen Liu, Luigi Palmieri, Sebastian Koch, Ilche Georgievski, Marco Aiello. **“DELTA: Decomposed Efficient Long-Term Robot Task Planning using Large Language Models.”** *ICRA*, 2025, 10995–11001. [项目页](https://delta-llm.github.io/), [arXiv:2404.03275](https://arxiv.org/abs/2404.03275), DOI `10.1109/ICRA55743.2025.11127838` | 用 scene graph 生成形式化 domain/problem，再由 LLM 自回归分解长任务子目标，并交给自动规划器求解。 | 作为近期 LLM+formal planner 对照。本文的 mixed door→container 链可被视为更窄但可诊断的状态转移任务；不要声称 DELTA 没有重规划，需按其版本准确描述。 |

## 建议如何写进当前 Related Work

### 建议新增的论证顺序

1. **先定义表示谱系。** 用现有 Armeni/Hydra/ConceptGraphs/HOV-SG，加 A1–A5，说明从静态层次图、在线增量图、开放词汇图到 task-driven graph 的演进。
2. **再引出“可行动但非交互状态”的缺口。** A1 的 traversability 和 A2 的增量更新是前置工作；A9–A12 说明功能部件、状态、铰接和容器关系正在被显式化，但多数目标是 manipulation 或 retrieval。
3. **然后讨论图到规划。** B1–B7 形成“经典 TAMP → 图编辑 → 大场景 TAMP → LLM heuristic”的连续链，避免把所有工作笼统归为“LLM planning”。
4. **最后放最近的 LLM/闭环工作。** C1–C6 说明 affordance grounding、反馈重规划、scene-graph subgraph search 和形式化 planner 已有基础；本文的缺口是把交互是否必要以及交互后的导航图改变作为可测变量。

### 建议在 Related Work 中明确的三句定位

- 既有动态/可行动场景图通常编码拓扑、时空关系或对象 affordance，但不把开合动作表示为会改变导航 connectivity、target visibility 或 accessibility 的显式状态转移。
- 既有 TAMP/LLM 系统能够选择和执行操作，却多以 manipulation/household completion 为终点；交互是否必要、是否误触发，通常不是独立的 episode-level 变量。
- 本文将 channel 与 container effects 放入同一导航任务接口，保留现有低层技能，并通过状态更新后重规划和 required/unnecessary 控制评估交互决策质量。

## 与本文主贡献的边界提醒

最接近的四篇不应被笼统称为“没有交互图”：

- **MoMa-LLM (2024)** 已有动态开放词汇图、对象中心动作和交互搜索；差异应落在任务终点、交互必要性标签、channel/container mixed split 和独立导航指标。
- **MomaGraph (2025)** 已明确使用 state-aware unified graph、precondition/effect reasoning 和 Graph-then-Plan；本文应避免把“state-aware graph”本身宣称为全新概念，而强调导航状态变化与诊断 benchmark。
- **Pandora (2026)** 与 **MoMa-SG (2026)** 已处理铰接对象、容器关系和隐藏目标；本文不应声称首次把容器或 articulated objects 放入图，而应突出 task-level evaluation 和 channel/container composition。
- **SayPlan / DELTA / Optimal Scene Graph Planning** 已展示图、LLM、形式化规划和重规划的组合；本文的方法贡献应限定在交互导航决策闭环及其可验证的状态接口。

## 备用候选（不计入上面的 25 条核心清单）

若需要补足“开放词汇地图”而不扩展动态交互段，可优先选择：

- Boyuan Chen et al., **“Open-vocabulary Queryable Scene Representations for Real World Planning” (NLMap)**, arXiv:2209.09874. [论文](https://arxiv.org/abs/2209.09874)。重点是 VLM 构建可查询表示、LLM 根据对象可用性生成计划。
- Chenguang Huang et al., **“Visual Language Maps for Robot Navigation”**, *ICRA* 2023, 10608–10615. [论文](https://arxiv.org/abs/2210.05714)。重点是把视觉语言特征直接融合到 3D 地图并支持语言到导航目标序列。
- Jacky Liang et al., **“Code as Policies: Language Model Programs for Embodied Control”**, arXiv:2209.07753. [论文](https://arxiv.org/abs/2209.07753)。重点是 LLM 生成可调用感知/API 的机器人程序；更偏执行层，若篇幅有限可不放 Related Work。

## 使用注意

- A10–A12 和 C6 的版本较新；最终引用前应按论文提交截止日期冻结版本，并确认是否已有正式会议/期刊版本。
- 不建议在 Related Work 中逐条堆砌所有 25 篇。正文可选 12–16 篇，剩余文献放表格、补充材料或 Method 背景。
- 对 A8、A10–A12 的比较应使用中性措辞（“focuses on…”, “does not evaluate…”），避免无证据的“首次”“唯一”“完全没有”。
