# AGENTS 工作说明

## 项目概览

本项目是 MolmoSpaces / MolmoBot 相关代码库，面向机器人操作与导航场景，包含资产与场景转换、抓取生成、远程操作、数据生成、基准评测，以及 MuJoCo、Isaac、ManiSkill 等模拟器相关支持。

主要参考文档：

- `README.md`：项目介绍、安装方式、主要入口。
- `docs/`：资产、环境变量、数据、评测等更详细说明。
- 各子目录下的 `README.md`：模块级说明。
- `scripts/InteractiveNav/interactive_navigation/`：交互式导航课题文档、讨论记录、阶段规划与文献调研。

## 当前核心高层文档

当前交互导航相关的高层信息，应优先维护并收敛到以下 4 个核心文档：

1. `TODO.md`：项目子任务拆解、阶段规划与开发推进
2. `readme_pi.md`：项目整体立意、研究目标、系统概述与文档索引
3. `AGENTS.md`：Agent 协作约定、工作边界与文档维护规范
4. `test.md`：开发测试命令汇总、流程测试方式与参数配置

维护原则：

- 子任务拆解与当前阶段判断，以 `TODO.md` 为准。
- 项目整体立意、系统主线与文档入口，以 `readme_pi.md` 为准。
- Agent 的工作边界、协作约定和文档维护规范，以 `AGENTS.md` 为准。
- 命令、测试流程和调试入口，以 `test.md` 为准。
- `scripts/InteractiveNav/interactive_navigation/` 下的历史方案与讨论，后续应被逐步吸收进上述 4 个文档，避免继续作为唯一高层入口依赖。

## 当前重点课题：交互式导航

交互式导航课题的核心目标是在传统语义地图导航或 SLAM 导航基础上，增加环境交互能力，使机器人能够通过开门、开冰箱、开衣柜、开抽屉等交互行为完成更具体的导航与探索任务。

当前研究定位：

- 核心贡献聚焦于**交互图的构建与感知**，不是底层 manipulation 控制。
- 当前第一阶段应把**交互性作为导航空间中的一等变量**建模，重点放在交互如何改变可达性、可见性和路径代价。
- 当前主类优先聚焦于：
  - **通道属性**：改变空间连通性与可达性，例如 door、sliding door、gate-like barrier、可移动阻挡物
  - **容器属性**：改变目标可见性、可取性与内部可访问性，例如 fridge、cabinet、drawer
- 设备、开关、灯光、窗帘等更适合作为后续扩展，而不是第一阶段主线。
- 将交互对象、交互状态、交互代价和状态转移链接到场景拓扑图或语义图中。
- 操作执行阶段优先使用 oracle interaction、已有 open/close policy 或 MolmoSpaces 内置任务能力。
- 评估重点是是否正确识别需要交互、是否维护交互状态，以及交互后导航成功率和路径效率是否提升。

交互式导航相关文档：

- `scripts/InteractiveNav/interactive_navigation/agent_init.md`：课题初始说明。
- `scripts/InteractiveNav/interactive_navigation/plan.md`：整体研究规划与阶段目标。
- `scripts/InteractiveNav/interactive_navigation/discussion.md`：课题讨论记录。
- `scripts/InteractiveNav/interactive_navigation/survey_2026-04-11.md`：文献调研与研究空白。

该课题的差异化表达应保持清晰：不是“做一个更强的开门控制器”，而是“把交互性作为导航图中的状态变化因素来建模，使规划能够主动利用交互改变可达性、可见性与效率”。

## 交互式导航技术路线

优先采用 MolmoSpaces 作为实验平台，结合其大规模多房间场景、可操作对象、MuJoCo 物理仿真、`nav_to_obj` 导航任务、`DoorOpeningTask` / open-close 任务与 benchmark/evaluation 框架。

当前建议阶段：

1. 基础搭建：跑通 `nav_to_obj` 和 door opening/open-close 相关能力，确认场景、对象、关节、门状态、导航目标的可读接口。
2. 双主线收敛：
   - MolmoSpaces 主线：基于 GT / oracle 构建交互导航任务、表示、规划与 benchmark
   - ROS 模块化主线：基于 detector-only 构建交互图和可复用工程基线
3. 交互图表示：设计能表达状态转移、交互触发边和交互代价的导航表示。
4. 规划验证：比较 `pure nav`、`nav + oracle open`、`interactive graph planner` 与 ROS 模块化基线。
5. 后续扩展：在 door-centric 最小闭环稳定后，再扩展到 container 类对象、数据采集、Agent 架构与端到端基线。

优先研究问题：

- 交互对象 ground truth 从哪里来：对象类别、joint 类型、关节范围、task sampler 还是 benchmark spec？
- 门的 open/closed/locked 状态如何定义和读取？
- 交互代价如何建模：固定代价、基于对象类型、基于预计时间，还是基于失败概率？
- 交互图如何和已有 nav map / scene graph / ProcTHOR room graph 对齐？
- 第一阶段如何对齐 MolmoSpaces 主线和 ROS 模块化主线的公共表示接口？

## 工作原则

- 优先遵循仓库已有代码风格、目录结构和工具链。
- 修改范围应尽量贴近用户请求，避免无关重构。
- 不随意修改大规模数据、下载资产、生成结果或 benchmark 文件。
- 不主动运行耗时的数据生成、仿真、渲染或评测任务，除非用户明确要求。
- 不提交密钥、认证文件、本地缓存、机器相关路径或大型二进制产物。
- 遇到用户已有改动时，不要回滚；应在现有改动基础上继续工作。

## 环境与安装

推荐 Python 版本为 3.11。

基础环境：

```bash
conda create -n mlspaces python=3.11
conda activate mlspaces
pip install -e ".[mujoco]"
```

可选安装项根据任务选择：

```bash
pip install -e ".[dev]"
pip install -e ".[grasp]"
pip install -e ".[housegen]"
pip install -e ".[curobo]"
```

如果任务只涉及文档、配置或静态代码阅读，不应为了验证而安装大型依赖。

## 常见目录

- `molmo_spaces/`：核心项目代码。
- `molmo_spaces/evaluation/`：benchmark 与评测逻辑。
- `molmo_spaces/grasp_generation/`：抓取生成相关流程。
- `molmo_spaces/housegen/`：房屋与场景生成工具。
- `scripts/datagen/`：脚本化 planner、数据生成与 benchmark 创建相关脚本。
- `scripts/InteractiveNav/`：交互式导航与 ROS/nav_to_obj 联调相关脚本。
- `docs/`：项目文档与图片资源。
- `molmo_spaces_isaac/`：Isaac 相关支持。
- `molmo_spaces_maniskill/`：ManiSkill 相关支持。
- `Interactive-Nav-SG-nav/`：交互式导航相关代码。

与交互式导航直接相关的仓库入口：

- `molmo_spaces/configs/base_nav_to_obj_config.py`：导航到目标对象任务基础配置。
- `molmo_spaces/data_generation/config/nav_to_obj_configs.py`：`nav_to_obj` 数据生成配置。
- `molmo_spaces/tasks/nav_task.py`、`molmo_spaces/tasks/nav_task_sampler.py`：导航任务与采样。
- `molmo_spaces/tasks/opening_tasks.py`、`molmo_spaces/tasks/opening_task_samplers.py`：开门/open-close 类任务与采样。
- `molmo_spaces/policy/solvers/opening_solver.py`：RBY1 door opening planner。
- `molmo_spaces/policy/solvers/object_manipulation/`：open/close planner policy。
- `molmo_spaces/evaluation/benchmark_schema.py`、`molmo_spaces/evaluation/eval_main.py`：benchmark JSON schema 与评估入口。
- `molmo_spaces/utils/scene_maps.py`：场景地图与 door path 相关工具。

## 搜索与阅读代码

- 搜索文件优先使用 `rg --files`。
- 搜索文本优先使用 `rg`。
- 阅读代码前先确认相关模块已有 README、配置文件和测试文件。
- 处理 JSON、YAML、XML、MJCF、USD 等结构化文件时，优先使用结构化解析方式，避免脆弱的字符串替换。

## 验证方式

修改代码后，优先运行与改动范围最小匹配的检查。

常见命令示例：

```bash
pytest
```

交互式导航与 `nav_to_obj` 调试可参考：

```bash
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1
```

评估流程优先参考：

```bash
python molmo_spaces/evaluation/eval_main.py <POLICY_CONFIG> --benchmark_dir <BENCHMARK_DIR>
```

如果只修改某个模块，应优先运行相关测试文件或更小范围的测试。若因为依赖、资产、GPU、模拟器或网络限制无法运行测试，需要在最终说明中明确写出。

## `interactive-nav/full` 集成与 ROS 端到端回归

当任务要求从 `codex/exp-setting` 创建或更新 `interactive-nav/full`、合入最新 `main` 并证明交互导航性能没有下降时，按以下流程执行。不要因为 `/opt/ros/noetic/setup.bash` 不存在就判定本机没有 ROS：本机使用 Conda ROS Noetic，前缀为 `/home/ldl/conda_envs/ros-noetic`；MolmoSpaces 仿真 Python 环境则是独立的 `/home/ldl/conda_envs/mlspaces`。

### 分支与提交纪律

1. 开始前记录基线 SHA，并执行 `git fetch origin main`。确认目标分支包含最新 `origin/main`，不能只依据可能过期的本地 `main`。
2. 工作树已有用户改动时，不切换或覆盖该工作树；为 `interactive-nav/full` 使用独立 worktree。
3. 合并提交、每一项兼容性修复、测试调整和文档更新分别提交。不要把多个不相关修复压进一个提交，也不要提交 `build/`、`devel/`、缓存、日志、视频或 benchmark 输出。
4. 每个小提交后先运行与该修改直接相关的最小测试；全部修改完成后再运行静态回归和 ROS 端到端回归。

可参考以下分支检查命令：

```bash
git fetch origin main
git status --short --branch
git log --oneline --decorate --graph -12
git merge-base --is-ancestor origin/main HEAD
```

最后一条命令退出码必须为 `0`。提交建议保持单一意图，例如：

```text
merge: preserve interactive navigation history on latest main
feat(interactive-nav): restore simulator and navigation integration
fix(interactive-nav): <one concrete compatibility fix>
test(interactive-nav): cover <the repaired behavior>
docs: document ROS regression workflow
```

### 准备 Conda ROS 和当前 worktree 的 catkin 产物

每个新 worktree 都必须生成自己的 `Interactive-Nav-SG-nav/build` 和 `devel`；不能复用另一个绝对路径下生成的 `devel/setup.bash`。首次准备命令如下：

```bash
cd "$(git rev-parse --show-toplevel)"
export TMPDIR=/home/ldl/tmp/interactive-nav-full-build
export XDG_CACHE_HOME=/home/ldl/.cache/interactive-nav-full-build
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

source /home/ldl/conda_envs/ros-noetic/setup.bash
cd Interactive-Nav-SG-nav
catkin_make \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=/home/ldl/conda_envs/ros-noetic/bin/python
source devel/setup.bash

command -v roscore
command -v roslaunch
rospack find nav_pkg
rospack find semantic_decision_py_pkg
```

预期 `roscore`、`roslaunch` 位于 `/home/ldl/conda_envs/ros-noetic/bin/`，两个 `rospack find` 指向当前 worktree。若 `/home/ldl/conda_envs/ros-noetic/setup.bash` 本身不存在，才属于 ROS 环境缺失；安装或下载 ROS 需要用户另行授权。若 Conda ROS 存在但 catkin 构建失败，应报告首个真实编译/依赖错误，不能笼统写成“缺少 `/opt/ros/noetic`”。

### 单场景 ROS smoke

从仓库根目录运行。脚本会先激活 MolmoSpaces Python 环境，再加载当前 worktree 的 `ROS_SETUP`；临时目录、缓存和输出必须放在 `/home/ldl`：

```bash
cd "$(git rev-parse --show-toplevel)"
RUN=/home/ldl/outputs/interactive-nav/full_smoke_h1_$(date +%Y%m%d_%H%M%S)
TMPDIR=/home/ldl/tmp/interactive-nav-full-smoke \
XDG_CACHE_HOME=/home/ldl/.cache/interactive-nav-full-smoke \
HF_HOME=/home/ldl/.cache/interactive-nav-full-smoke/hf \
TORCH_HOME=/home/ldl/.cache/interactive-nav-full-smoke/torch \
CONDA_SH=/home/ldl/miniconda3/etc/profile.d/conda.sh \
CONDA_ENV=/home/ldl/conda_envs/mlspaces \
PYTHON_BIN=/home/ldl/conda_envs/mlspaces/bin/python \
ROS_SETUP="$PWD/Interactive-Nav-SG-nav/devel/setup.bash" \
ROS_MASTER_URI=http://127.0.0.1:15101 \
METHOD=full_mllm_exploration \
HOUSE_IND=1 SCENE_SEED=1 ROUTE_ID=full_smoke_h1 \
USE_FIXED_ROUTE=false RUNTIME_TARGET_MODE=none \
TASK_HORIZON=100 SIM_TIMEOUT_S=1800 \
MAPPING_SCAN_SOURCE=organized_depth POINTCLOUD_STRIDE=1 \
INITIAL_DOOR_STATE=closed FORCE_CLOSE_CONTAINERS=true \
ENABLE_RECORDING=true CLEAN_INTERMEDIATE=false \
bash scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
  "$RUN" full_smoke_h1
```

运行前必须确认所选 `ROS_MASTER_URI` 端口没有其他 master。清理时只终止本次启动的进程组，不使用全局 `pkill roscore`。成功后至少检查：

- `$RUN/semantic_exploration_result.json`
- `$RUN/force_interaction_events.json`
- `$RUN/debug/events.jsonl`
- `$RUN/offline_video_summary.json`
- `$RUN/videos/overview_6panel.mp4`

### “无性能下降”回归门槛

不能用“脚本退出 0”代替算法性能验收。应在合并前基线 SHA 和 `interactive-nav/full` 上使用相同模型服务、配置、house、seed、horizon、录制开关与机器资源，至少运行 H1/H3/H4/H7/H8/H10 六场各 1000 step。每场使用不同 ROS master 端口和独立输出目录；资源不足时分批并行，但基线与候选必须采用相同并发度。

回归报告至少比较：

- 完成 step、是否异常早停、崩溃、TF 卡死、长期无 subgoal 或 actionlib 竞态；
- coverage / mapped-free；
- 物理交互成功数与失败数、container 成功数、drawer scan 成功数；
- interaction approach/M1/anchor exhausted、navigation stagnation 与 timeout 次数；
- recorder 的 exact-step 对齐、missing/drop/write failure；
- 每 step 墙钟耗时以及关键 mapping/navigation p50、p95。

候选分支必须满足：没有新增崩溃或永久卡死；视频/状态 exact-step 对齐且无丢帧；交互总成功数和容器成功数不低于基线；失败数不高于基线；平均 coverage 不出现可复现下降。单场 coverage 绝对下降超过 `0.02`、交互结果变差或耗时显著上升时，不得宣称“无性能下降”，必须归因、修复并按相同条件重跑。MLLM 存在随机性时，至少重复异常场景，不能用一次有利波动覆盖退化。

如果完整 ROS 回归受阻，允许先提交已经通过最小测试的独立修复，但最终状态必须明确写成 `blocked`，列出已经通过的测试、缺失的具体环境文件或首个构建错误，以及环境恢复后的完整命令；在端到端结果产生前不能声称性能门槛通过。

## 文档与注释

- 面向用户的安装、使用和功能说明放在 `README.md` 或 `docs/`。
- 面向 Agent 的工作约定放在本文件。
- 注释应简短，只解释不明显的意图或复杂逻辑。
- 不为显而易见的赋值、调用或控制流添加冗余注释。

## 安全边界

以下操作需要用户明确确认：

- 删除文件或目录。
- 重新生成、覆盖或清理数据集与资产。
- 运行长时间仿真、渲染、benchmark 或数据生成任务。
- 安装大型依赖或需要网络下载的依赖。
- 修改认证、密钥、token、个人本地配置等敏感文件。

## 提交前检查

完成任务前应尽量确认：

- 改动是否符合用户请求。
- 是否误改了无关文件。
- 是否有必要更新 README 或 docs。
- 是否运行了合理范围内的验证命令。
- 如果无法验证，是否已经说明原因。
