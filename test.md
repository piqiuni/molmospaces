# 交互导航开发测试手册

最后更新：2026-09-23

### 2026-09-23：22 日基线八项选择性迁移

分支 `codex/migrate-22-selected` 基于 `d3f27870b`，迁移来源 `8fb9460ae`。
范围、保留的模型超时、未迁入项与性能验收边界见
[迁移报告](docs/migration_22_selected_20260923.md)。
本轮 706 项定向与相邻回归通过，没有运行 ROS 长时仿真。

复现此次测试集合：

```bash
cd /home/ldl/molmospaces-exp-setting
export TMPDIR=/home/ldl/tmp/migrate22-tests
export XDG_CACHE_HOME=/home/ldl/.cache/migrate22-tests
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
export PYTHONPATH=/home/ldl/conda_envs/ros-noetic/lib/python3.11/site-packages:$PWD/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts:$PWD/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts:$PWD/Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts
/home/ldl/conda_envs/mlspaces/bin/python -m pytest -q -p no:cacheprovider \
  Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/tests/test_{attribute_inference_request_state,room_attribute_inference,interaction_graph_store}.py \
  Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/tests/test_{behavior_candidates,mission_completion,semantic_behavior_executor_static,behavior_execution,rule_decision_navigation_recovery,target_distance_verification}.py \
  scripts/InteractiveNav/test_{run_benchmark_eval,run_interactive_nav_v3_ros_eval_batch,timeout_monitoring,scene_distractor_filter}.py \
  mlspaces_tests/data_generation/test_interactive_nav_v3_{metrics,benchmark_evaluation,goal_status,round_summary,benchmark_cli,public_evaluator_contract,evaluation}.py
```

监控脚本应从仓库根目录以模块方式启动，避免 evaluation 目录的 `types.py`
遮蔽标准库。以下 `RUN` 和 `TASK_ID` 需要替换为实际运行目录与云任务 ID。
collector 只读取已有产物，不启动模型/仿真；周期监控会查询云任务状态。

```bash
RUN=/home/ldl/outputs/interactive-nav/your-evaluation-run
/home/ldl/conda_envs/mlspaces/bin/python -m scripts.InteractiveNav.evaluation.collect_timeout_performance "$RUN"
TASK_ID=your-cloud-task-id
/home/ldl/conda_envs/mlspaces/bin/python -m scripts.InteractiveNav.evaluation.monitor_mllm_timeouts \
  --task-id "$TASK_ID" --evaluation-dir "$RUN" \
  --qwen-dir "$RUN/qwen-service" --interval 30 --window 120
```

### 2026-09-17: 独立模块消融入口

新增 `scripts/InteractiveNav/run_benchmark_ablation.py`，提供 `full`、
`no_interaction_graph`、`no_task_decision`、`no_outcome_update` 四组。
原算法、原 launch、原评测器及默认配置保持不变；适配通过运行目录中的 launch-prefix 生效。
实验定义、边界和命令见 [消融说明](scripts/InteractiveNav/ablations/README.md)。
其中图消融是高层关系图推理消融，底层候选可行性仍共用原图。

```bash
/home/ldl/conda_envs/mlspaces/bin/python \
  /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/run_benchmark_ablation.py \
  --variant no_task_decision --workers 1 --episode-indices 10 --max-steps 20 --dry-run
```

`--dry-run` 不创建产物，不启动 ROS/模型/仿真。
本轮 28 项新增消融测试及 100 项原有相关回归通过，共 128 项。
检查日志位于 `/home/ldl/outputs/interactive-nav/ablation-check-20260917/`；
已验证三组生成 launch 均可由实际 ROS loader 解析，并且只替换指定的节点。
这里只验证接口和模块行为，未运行长时仿真或 benchmark。

### 2026-09-17: 完整方法实验基线

Git 标签 `codex/experiment-baseline-20260917` 固定本轮完整方法与评测器代码，
作为后续模块级消融的共同起点。提交前 HEAD 为 `301f0aff8`。
这是代码与配置基线，本次未重新运行仿真或完整 benchmark，不代表新增性能验收结果。

- 方法：`full_mllm_object_goal`；M1/M2/M3 分别为
  `dynamic_mllm / mllm_score / mllm_skill_verified`。
- 评测器：`interactive_nav_v3_benchmark_eval_v17`；动态预算公式 v2，基础 200，
  路径超过 3 m 后每米 40，必要通道交互 200，必要容器交互 250，
  容器每个记录关节 50，向上取整到 50，launcher 硬上限 2000。
- 默认 launcher：10 worker，episode 10–12、1010–1012、2010–2013，
  完整录制，本机模型入口 8010。观察轮次倍率保持当前值 1；因此可能先达到观察轮次
  上限，不能把每场运行解释为一定执行满动态动作预算。后续变更须对所有比较组一致生效。
- restricted-GT 最大距离 8 m；抽屉 action/M1 最小站距 0.70 m；
  保留当前容器交互锚点完成判定、drawer scan 完成记忆、静态开门通行与目标冷却修复。
  容器锚点判定与旧版单纯目标中心距离判定不同，消融组必须共用本基线评测器。
- Manifest：`interactive_nav_v3_procthor10k_val_release_v1_2` 的 `benchmark/benchmark.json`，
  3000 episode、639 house，SHA-256：
  `4021ca2bebd9c875ccc4df70c746d9ed7f2376d13247fa1b7e98f2e9690b219f`。
- 默认 `benchmark_batch.json` SHA-256：
  `d535b883a8f0c8758eb9d555a687ae6a366e969c5e9082d7e5b00366775b27b0`。
- 模型权重、认证、数据集、运行产物及包含个人云端队列/挂载的资源 YAML 不进入本提交。
  Git 标签不固定外部模型服务；每轮仍需记录模型版本、推理参数和实际生效配置。

提交前验证共 408 项通过：候选生成、执行状态机、决策恢复、交互图、launcher、
批量调度器、benchmark 预算/指标、restricted-GT 与目标完成契约。
其中 33 项决策恢复测试需要先加载 `/home/ldl/conda_envs/ros-noetic/setup.bash`；
其余测试使用 `mlspaces` Python，并将 `semantic_decision_py_pkg/scripts`、
`semantic_mapping_py_pkg/scripts`、`semantic_mllm_py_pkg/scripts` 加入 `PYTHONPATH`。
另通过修改文件的 Python AST、Shell 语法、JSON 解析和 `git diff --check`。
本机检查日志在 `/home/ldl/outputs/interactive-nav/baseline-check-20260917/`，不纳入 Git。

### 2026-09-16: 双卡自定义任务评测 smoke

入口为 `scripts/InteractiveNav/run_v3_custom_task_smoke.sh`，内部使用当前本机统一入口
`python scripts/InteractiveNav/run_benchmark_eval.py`。提交配置为
`scripts/InteractiveNav/configs/custom_task/v3_qwen_10worker_10scene_100step.yaml`
（本机云端资源配置，含队列及挂载信息，不纳入 Git）。
配置使用一个 `ml.pni2.7xlarge` 双卡实例：GPU 0/1 分别启动一个 Qwen 35B FP8
后端（端口 8000/8001），本机 8010 负载均衡器供 10 个隔离的 ROS/MuJoCo
worker 使用。Qwen 与仿真共享同一实例，启动方式与本机双服务一致。
测试选择 10 个 episode，覆盖 channel、container、mixed，固定 100 applied steps；
使用 `--no-recording`，用于验证云端调度、模型服务、ROS 与仿真链路，不作为性能回归。

```bash
cd /home/ldl
volc ml_task submit \
  -c molmospaces-exp-setting/scripts/InteractiveNav/configs/custom_task/v3_qwen_10worker_10scene_100step.yaml \
  -n interactive_nav_v3_qwen_10w10s100_$(date +%Y%m%d_%H%M%S)
```

提交后用返回的 task ID 监控；不要用 CLI `--set TaskRoleSpecs[0]...` 修改副本数，
当前 CLI 会把数组覆盖解析成无效规格。

```bash
volc ml_task get -i TASK_ID --output json
volc ml_task instance list -i TASK_ID --output json
volc ml_task logs -i TASK_ID
```

Python 3.10 编译头从 Ubuntu 开发包解压到
`/home/ldl/.cache/python3.10-dev`，不修改系统 Python 或 Qwen 虚拟环境。
容器镜像未预装 GLVND 的 EGL loader，因此 `libegl1`/`libglvnd0` 运行库解压到
`/home/ldl/.cache/egl-runtime`，入口在启动 Qwen 前执行一次 MuJoCo EGL context
预检；NVIDIA 驱动库仍由双卡实例注入，不复制本机驱动。
云端容器还必须显式设置 `MLSPACES_CACHE_DIR=/home/ldl/molmo-spaces-resources` 和
`MLSPACES_ASSETS_DIR=/home/ldl/molmospaces/assets`。否则资源管理器会在容器默认
home 下重新初始化资源；首个 worker 持锁联网，其余 worker 会长期阻塞在 `.lock`。
同时设置 `NLTK_DATA=/home/ldl/nltk_data`，复用已有 WordNet，禁止 worker 向
`/root/nltk_data` 下载语料。
共享输出目录为
`/home/ldl/outputs/interactive-nav/custom-task-TASK_ID/evaluation`。重点检查
`resource_telemetry.csv`、`summary.csv`、各 episode 的 `eval.log`，以及
`task-state/evaluation.exit_code`。任一 episode、Qwen、环境预检或 batch runner
失败都会令自定义任务失败，避免平台显示 Success 时掩盖评测未实际运行。

### 2026-09-15: 到达判定与无动作恢复回归

- 目标导航候选与最终图验证都检查对象距离；导航锚点到达不能替代目标成功距离。目标视点在成功半径内生成，受阻时检查其他方向，保留当前可见且已到达目标的无移动验证路径。
- `require_current_visibility=false` 时，可靠历史观测与当前对象距离共同决定到达，允许抽屉关闭后接近已观测目标；这与 evaluator 的公共观测证据账本一致。显式开启实时可见性要求时仍检查当前帧；历史可见不能绕过距离阈值。
- 无有效 frontier 但仍有未解决交互对象时，允许有界恢复扫描。无候选等待保留 120 观察步上限，并增加 30 秒空闲上限；扫描执行时间不计入空闲墙钟时间，避免与 bridge 的 60 秒无新动作保护竞态。
- 门 M1 在完整入镜前提下，按同一对象的相邻源帧累计稳定观测；不再以旋转扫描中包围框 IoU 作为连续观测的必要条件。重复帧不累加、间断帧重置，精确 RGB 配对与后续状态确认仍保留。
- 三轮端到端复测使用 `run_interactive_nav_v3_ros_eval_batch.py`、动态预算与完整录制。正式评估以 `applied_action_step_count` 为预算单位，不能用录像帧数比较预算；任务失败与执行进程失败分开统计。复测与并发结论须以最终批次报告为准，不能仅凭以下单测宣称全部场景成功。

最小回归（从仓库根目录运行，复用已有 Conda 环境）：

```bash
TMPDIR=/home/ldl/tmp XDG_CACHE_HOME=/home/ldl/.cache PYTHONDONTWRITEBYTECODE=1 \
/home/ldl/conda_envs/mlspaces/bin/python - <<'PY'
import sys
from pathlib import Path
root = Path.cwd() / "Interactive-Nav-SG-nav"
sys.path.extend(str(p) for p in (root / "src").glob("*/scripts"))
sys.path.extend([
    "/home/ldl/conda_envs/ros-noetic/lib/python3.11/site-packages",
    str(root / "devel/lib/python3.11/site-packages"),
])
import pytest
decision = root / "src/semantic_decision_py_pkg/tests"
tests = [decision / name for name in (
    "test_behavior_candidates.py", "test_behavior_execution.py", "test_mission_completion.py",
    "test_rule_decision_navigation_recovery.py", "test_target_distance_verification.py",
)]
tests.append(root / "src/semantic_mapping_py_pkg/tests/test_attribute_inference_request_state.py")
raise SystemExit(pytest.main([*(str(p) for p in tests), "-q", "-p", "no:cacheprovider"]))
PY
```

### 2026-09-15: 轻量录制配置与动态预算

- 配置入口：`scripts/InteractiveNav/configs/evaluation/benchmark_eval.conf`，详细说明见同目录 `README.md`。单场命令只需输出目录与 episode index；可用第三个参数传入 Bash 覆盖配置。
- 默认动态预算、上限 2000；关闭 `events.jsonl` 与离线六宫格 PNG，保存精简且 gzip 压缩的 `step_boundaries.jsonl.gz`。保留完整图 6 数据与终止摘要，旧 JSONL 仍可重绘。
- 每场保存 `config/effective_config.env`、四份算法 YAML、ROS 日志配置；不复制凭据内容。Python ROS 日志默认 WARN、每文件 5 MiB 加 2 份轮转，控制台日志不是严格总量配额。
- M2 使用五步逻辑提示词与独立 `selection_reasoning_effort=low`，1536 token 总上限、30 秒超时；超时仅重试 1 次，退避 1 秒（最大等待约 61 秒）。`SEMANTIC_M2_ENDPOINT` / `SEMANTIC_M2_MODEL_NAME` 可把 M2 切到更强的 OpenAI-compatible 文本模型，不改变视觉 M1/M3；未设置时沿用共享模型。仅消费最终 JSON；需区分请求推理模式与后端实际返回 reasoning token。
- 历史对照（实际 applied action）：Container 动态 650、成功 202；Mixed 动态 1000、首次失败 1000、复跑成功 685；旧 Channel 动态 650、历史失败 1000。这不是动态预算下的新仿真结果。

```bash
bash scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_test.zsh \
  /home/ldl/outputs/interactive-nav/my_eval 0
```

### 2026-09-15: Benchmark v16 感知与门 M1 接口修复

- 默认评测数据改为 `interactive_nav_v3_procthor10k_val_release_v1_2`，三类各 1000 条。旧 v1.1 的 2968 条结果和质量门控签名不沿用。
- Benchmark 感知移除额外的 512 像素平方 bbox 面积门槛；保留像素、短边、距离和投影可见范围检查。图像里可见不等于已通过这些过滤，不应把没有发布解释为物理不可见。
- `doorway` / `doorframe` / `door_leaf` 统一公开类别为 `door`，保留 `obj_…` 身份；帧头携带同一采集时刻的相机 `observation_pose_xyyaw`，使门的多视点 M1 门控能够正常请求。
- 最小回归：`python -m pytest mlspaces_tests/data_generation/test_v3_semantic_minimal_bridge.py mlspaces_tests/data_generation/test_ros_object_goal_evaluator_adapter.py mlspaces_tests/data_generation/test_restricted_gt_perception.py mlspaces_tests/data_generation/test_benchmark_current_contract.py`。
- 本次未重新运行仿真；此前 v15 视频不能作为 v16 端到端验收。整体 AABB 包含也不能作为真实容器内腔关系的证据。

### 2026-09-15: 跳过等待后的十场并行 500-step 回归

- 当前可提交的完整性能记录：[2026-09-15 性能检查点](docs/interactive_nav_performance_20260915.md)。提交前扩大回归 1218 + 13 = 1231 项通过；没有重新启动另一轮仿真。
- H1-H10、seed=house、10 worker、500-step 上限，原生线程各 1、录制开启；8 场达到上限，H2/H3 在 371/286 步因 `no_eligible_candidates_after_bounded_recovery` 提前结束，共 4657 步。整批墙钟 725.060 秒；平均 0.751 秒/step，P95 1.047 秒。
- 相比最近十场 R5：平均 step 耗时下降 2.1%，覆盖率 72.15% -> 72.20%；门调用/物理成功 10/4 -> 11/6，另有 1 次静态通道确认，门失败 6 -> 4；容器调用/成功均为 8/8，无待返回。两轮还相差横移修复与机器人锁定缓存，不是跳过等待的单变量 A/B。
- `navigation_hold` 207 步，ready 后平均等待 0.619 ms、最大 0.924 ms；无未 ready 放行、无误计动作超时。M1 237/237 无请求错误或超时。GT 分母、批处理参数与 R5 十场均一致，源码运行期间无变化。
- 十个视频 exact-step 对齐，全部 4657 帧完整解码通过。日志审计未发现运行期崩溃；H2/H3/H4/H6/H8/H9/H10 仍有关闭期内存释放告警。本轮专用端口 16201-16210 已释放。
- 不能宣称逐场无退化：H3 覆盖率 45.73% -> 27.82%、无交互且提前结束，该场未触发 navigation_hold；H10 的 17.84 个百分点提升几乎抵消了 H3 的下降。H8 容器成功 1 -> 0。结果、逐场耗时、差异说明及全部视频见 [本轮报告](/home/ldl/molmospaces-exp-setting/outputs/navigation_hold_500/report.md)；原始批次为 `/home/ldl/outputs/interactive-nav/batch_timeopt_navigation_hold_500_20260915_020704`。

### 2026-09-15: 交互期间跳过导航指令等待

- 仿真主循环仅在交互控制器 `should_pause_navigation()` 为真时，调用 `RosBridgePolicy.get_action(..., hold_navigation=True)`。原传感器发布和本帧地图 ready 等待完成后返回保持动作，记录 `action_source=navigation_hold`，不计入连续动作超时；未 ready 时保留原导航等待路径，其他调用者默认行为不变。
- 保留原 fresh-command gate、step sync、录制 ACK、`task.step`、交互 before/after 回调和躯干视角目标；未修改物理子步、交互步数、距离、容差、地图发布频率或其他优化项。
- 验证 142 项通过：`mlspaces_tests/test_ros_bridge_navigation_hold.py`、`test_nav_ros_scene_timeout.py`、`test_ros_bridge_policy_twist.py`、`test_force_interaction_bridge.py`、`test_force_robot_lock_cache.py`、`test_force_contact_lookup.py`。新测试覆盖延迟 ready、ready 失败、bootstrap、ROS shutdown、普通导航超时，以及保持 20 步后第 21 步恢复导航，逐步检查传感器/录制与躯干指令保留。
- 测试使用下述组件验证相同的 `/home/ldl` 临时/缓存目录、Conda Python 与 ROS `PYTHONPATH`，通过 `python -m pytest -q -p no:cacheprovider` 运行这六个文件。本次没有启动 500-step ROS 仿真，不据此宣称整场性能提升。

### 2026-09-15: 后续耗时优化的组件验证

- 本次只修改交互期间的机器人锁定回调，缓存同一次交互的 base/group views、零速度模板和控制数组形状；模型、data 或 robot view 更换时退回原查询路径。保留全部物理子步、锁定回调、`mj_forward`、相机更新以及 base ctrl setter，不调整交互距离或容差。
- 定向测试 99 项通过。新增真实小型 MuJoCo 力驱动轨迹逐元素对比、yaw 跨界与缓存失效测试；交互状态机同时覆盖 3/20 步，第 20 步之前不返回完成，物理子步计数不变。该次组件验证时尚未实现“交互时跳过导航动作等待”，不能把该测试当作这一优化的端到端验收。
- H1 静态组件微基准：锁定回调 9.301 -> 9.271 ms/次，约 0.3%，收益很小；其中 `mj_forward` 约 9.048 ms。渲染场景更新复用约节省 0.1 ms/三通道渲染，去重 RGB readback 约节省 0.8-1.0 ms；三个视角中的一个连原始基线 RGB 重复渲染都不完全一致，因此未修改生产渲染路径。
- H1 静态 GT 的位打包压缩样本约 17.1 KB，解压逐位一致；新地图算法与旧评测相差 6648 个可导航像素，旧算法重放精确复现旧分母 1789027。正式缓存必须固定算法、资产和机器人半径版本，不可直接改变历史覆盖率口径。
- 本次未跑 500-step ROS 回归、未做全量 GT 扫描。完整命令、实测数据、并行扫描/报错方案及六项建议见 [后续优化分析](/home/ldl/molmospaces-exp-setting/outputs/speed_followup_20260915/report.md)。

### 2026-09-14: 横移修复后 H1 单 worker 500-step 实测

- H1、seed=1、1 worker、原生数值线程 1、500 step 已完成；运行期间源码哈希未变。总墙钟 757.865 秒，step 总计 506.351 秒，平均 1.013 秒/step，P50 0.805 秒，P95 2.691 秒；视频 500 帧 exact-step 对齐且抽样可解码。
- 覆盖率 43.99%。门指令 2 次，成功 1 次，另一次在 step 497 发出、500 步结束时未返回；容器 1 次成功，为抽屉扫描。M1 28 次全部有效返回。关闭阶段仍有一次内存释放告警。
- 1579 条 stamped 速度指令全部 vy=0，无横移拒绝保护触发；实际位姿估计侧向速度峰值仍有 0.133 m/s，不能宣称物理侧向位移完全消失。
- 对照上轮 10-worker 的 H1：平均 0.915 秒/step，覆盖率 49.67%，无成功物理交互。本轮普通步均值约快 8.9%，但更多交互前置控制增加约 83.1 秒，使完整均值慢 10.6%。两轮横移修复、路线及交互不同，不是并发数单变量 A/B，不代表无质量退化。
- [视频、逐阶段耗时、残留问题与复现命令](/home/ldl/molmospaces-exp-setting/outputs/h1_lateral_fix_500/report.md)；[原始输出](/home/ldl/outputs/interactive-nav/batch_h1_lateral_fix_1worker_500_20260914_213404)。

### 2026-09-14: 非全向导航横移修复

- DWA 保持 `min_vel_y=max_vel_y=0`，将 `acc_lim_y` 从 0 改为 2.5，`vy_samples` 从 0 改为 1。横向制动窗口现在覆盖 `max_vel_trans=0.5` / `controller_frequency=5` 的一个控制周期，避免把非零 odom 横向速度重新采样为导航指令。
- 当前十场仿真入口 `scripts/InteractiveNav/run_nav_ros_sim.py` 默认 `--allow_lateral_cmd_vel false`；残留非法横移指令整步保持当前位姿，记录 `action_source=lateral_cmd_vel_rejected` 和累计拒绝告警，不仅截断横向分量后继续执行原前进/转向分量。明确使用全向规划器时可以设为 true；通用 `RosBridgePolicy` 默认仍兼容全向调用者。
- 保留真实 odom 横向速度、前向/旋转速度参数、3.0 线速度增益、0.2 秒控制周期、交互与到位条件。没有修改底盘物理或强制清零实际速度。
- 验证：相关 Python 回归 204 项通过；本机 ROS DWA 动态窗口七组输入均只产生 vy=0，旧 acc_lim_y=0 负对照能复现失败；H1 的 2057 条原始消息转换回放中，937 条横移指令被拒绝、1120 条正常指令保持原样。这不是场景重跑，尚未验证修复后十场运动与覆盖率。
- 扩展回归需将 `/home/ldl/conda_envs/ros-noetic/lib/python3.11/site-packages` 加入 `PYTHONPATH`，与五轮评测原有测试环境一致；否则既有 executor 测试的 ROS stub 缺少 `rospy.numpy_msg`。命令与结果见 [修复验证](/home/ldl/molmospaces-exp-setting/outputs/nav_lateral_fix/report.md)。
- 前后视频均为 15 fps，每仿真步 0.2 秒，没有改变播放倍率。R0->R5 的 H1 `timeout_noop` 从 139 减至 32，轨迹长度从 11.98 增至 21.93 m；视频运动更连贯还包含轨迹变化和原有异常横移，不能仅用计算加速解释，也不是提高了速度上限。

### 2026-09-14: 五轮时间优化评测

- 检查点为 `c0386aa4e`（时间优化前）。已完成额外同代码 R0 对照及五轮 H1-H10、10 worker、500-step 上限评测；优化改动尚未提交 git。
- 最近完整评测 R5（横移修复前）：实际 4840 step，9 场满 500，H2 在 340 step 提前结束；加权平均 0.768 秒/step、P95 1.083 秒、整批墙钟 844.173 秒。R0 为 1.551 秒/step，因此平均降低 50.5%。覆盖率 72.15%，门调用/成功 10/4，容器 8/8，无待返回命令。
- 不代表全面无退化：H1 本轮门成功和容器调用均为零；六次门失败全部为 `non_articulated` 拒绝。关闭期的内存释放告警仍存在。R4 首次数组真假判断异常已修复并完整重跑，失败现场保留。
- 五项实现分别是原生线程限制、深度射线/掩码复用、ROS NumPy 地图传输、地图预处理按掩码复用、交互接触拓扑查询缓存。未改变分辨率、物理步数、交互距离、安全距离和到达条件。
- 批处理新增 `--native-threads-per-worker`，默认 `1`，`0` 保留库默认；显式设置的 `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` / `MKL_NUM_THREADS` 优先，并记录在场景结果的 `native_thread_environment` 中。
- 最终相关测试 1180+13=1193 项通过；两处同名 `test_force_interaction_bridge.py` 分开运行。六个有效批次的 60 个视频均 exact-step 对齐，运行期间源码未变；测试不等于全仓库验收。
- [完整报告、历史对照与复现命令](/home/ldl/molmospaces-exp-setting/outputs/time_optimization_5rounds/report.md)；[逐场数据及最新十个视频](/home/ldl/molmospaces-exp-setting/outputs/time_optimization_5rounds/report_tables.md)。最新输出：`/home/ldl/outputs/interactive-nav/batch_timeopt_r5_contacts_500_20260914_192049`。

### 2026-09-14: 到位净空、M1 请求归属与深度投影回归

- `candidate.navigation_clearance_enabled`：full MLLM 配置启用候选发布前净空检查；过滤发生在生成索引对齐的 capture/action 列表之前，不修改已执行中的锚点编号。
- 当前候选筛选、替代探索点搜索及门/容器执行净空只检查机器人半径 + 安全余量 + 栅格半对角线，不再叠加到位容差；进入位置容差后不再检查 XY 容差，只检查实际位置的碰撞净空。探索模块原有的 0.40 m 足迹筛选、各类站位档位与到达容差保持不变。
- 去掉到位容差后的定向回归 434 项通过，覆盖净空、候选生成、导航执行、到位锁定、门前姿态和探索恢复；该修改已作为上方时间优化的 R0 对照重跑。下方保留修改前的历史批次。
- `executor.drawer_fallback_bbox_wait_task_steps: 8`：M1 失败后的抽屉 bbox 延迟可原地等待新帧，30 秒为失活保护；超出预算可重试，不永久排除对象。
- 当前物体表面距离：抽屉按后续要求改为 `0.50/0.85/1.00 m`，冰箱 `1.15/1.35/1.55 m`，其他容器 `0.50 m`，门交互 `0.95/1.20/1.45 m`，门复观测 `0.85/1.10/1.35 m`。
- 重点测试：`test_navigation_clearance.py`、`test_behavior_candidates.py`、`test_semantic_behavior_executor_static.py`、`test_attribute_inference_request_state.py`、`mlspaces_tests/test_depth_continuity_equivalence.py`。
- 原始 H10 地图回放和投影微基准：`outputs/analysis_runtime500_20260914/verify_clearance_runtime_fixes.py`。结果与命令约束见同目录 `clearance_runtime_fixes.md`；不启动仿真，不改原始 batch。
- 深度投影优化保留图像尺寸、深度阈值、物理步数和录制精度。微基准的阶段加速不等于 10 worker 的整场加速。

优化前实测：`/home/ldl/outputs/interactive-nav/batch_minimalfix_h1_h10_500_20260914_155635`，最小导航修复后 H1-H10、10 worker、500-step 上限，总墙钟 1286.160 秒（含启动、仿真、清理、离线视频和分析）。实际 4529 step，实测加权平均 1.576 秒/step，P95 2.544 秒（不含启动和后处理）；H2/H3/H5 分别在 421/151/457 step 因有界恢复后仍无可执行候选而提前结束，其他 7 场达到 500 step。门交互调用/成功为 5/3，容器为 6/6；平均覆盖率 66.54%。10 个视频 exact-step 对齐且可解码，无悬而未决的交互命令，运行期间源文件未变化。启动前相关测试 623 项通过。

本轮不代表性能验收通过：相对上一轮，平均覆盖率从 68.60% 降至 66.54%，H7/H8 分别从 99.60%/36.22% 降至 83.40%/20.52%，门成功数从 6 降至 3，H3 仍有导航停滞。日志审计未发现关闭前崩溃；各场关闭阶段存在内存释放错误，H7/H9 另有向已关闭 ROS topic 发布的异常。明细、视频入口和原始审计见最新目录的 `results_500.md`、`results_500.json`、`interaction_audit_500.json`、`step_timing_summary.csv` 和 `step_phase_timing.csv`。复现入口：`outputs/analysis_minimalfix_500_20260914/run_timed_batch.py`；汇总入口：同目录 `report_batch.py RUN_DIR`。

上一轮实测：`/home/ldl/outputs/interactive-nav/batch_clearance100_h1_h10_500_20260914_112848`，H1-H10、10 worker、500-step 上限，总墙钟 1172.989 秒。实际 4018 step，加权平均 1.428 秒/step，P95 2.194 秒；H2/H3/H4/H5/H10 分别提前结束于 319/106/217/430/446 step，因此不算完整的 5000-step 回归。10 个视频 exact-step 对齐，运行中源文件未变化。明细为该目录的 `step_timing_summary.csv`、`step_phase_timing.csv`、`step_timing_report.md` 和 `results_500.json`。原始计时保留在各场 `sim/step_timing.jsonl`。

上一轮距离/净空定向测试 94 项通过；额外的 `test_nav_goal_orientation_config.py` 有 3 项既有配置契约不匹配（orientation 0 对旧断言 3、DWA path bias 30 对旧断言 18、V3 的冰箱角度/净空开关与探索配置不同），未为此次运行改动这些无关参数。复现入口：`outputs/analysis_clearance100_500_20260914/run_timed_batch.py`；耗时汇总入口：同目录 `report_step_timing.py RUN_DIR`。

## 1. 文档定位

本文件用于维护交互导航相关的：

1. 开发测试命令
2. 流程测试方式
3. 常用参数配置
4. 调试入口与输出位置

使用原则：

- 面向项目整体立意、研究目标和文档入口，请看 `readme_pi.md`
- 面向子任务拆解和阶段推进，请看 `TODO.md`
- 面向 Agent 协作约定与维护边界，请看 `AGENTS.md`
- 面向具体命令、运行方式、调试方式和测试流程，请看本文件

---

## 2. 使用说明

本文件按“从轻到重”的方式组织测试入口：

1. 环境与路径检查
2. MolmoSpaces 侧最小运行
3. ROS 联调与导航系统启动
4. 语义交互图与检测测试
5. 地图、探索与手动控制辅助调试

如果只是改动文档、配置或高层设计，不需要跑重型测试。  
如果只是改动某个局部模块，优先使用最小相关命令，而不是整套系统全启动。

---

## 3. 环境与路径

### 2026-09-14 执行与感知回归

- 原始仿真入口启用 `publish_odom_twist=True, odom_twist_source="step_delta"`，按仿真控制 dt 计算实际位姿差分，同一步重复发布保持同一速度。离散位置控制动作结束时 qvel 接近零，不能把该瞬时停车速度当作上一控制步的运动，也不能使用墙钟 step 耗时估计速度。其他入口的瞬时速度模式保持兼容。
- 已发出的物理命令持有执行所有权直到匹配的 backend result。后台每秒在 `interaction_action_feedback` 发布带 command ID 的 RUNNING 进度，不携带私有资产名。
- `interaction_execution_stall_timeout_s=60`、`interaction_execution_wall_cap_s=600` 为失联/硬上限保护；drawer scan 仍受任务步预算约束。超限发布 EXPLORATION_STALLED 并保留所有权，结束本场，不在后台仍忙时派发新目标。
- M1 targeted refresh 区分 waiting_for_view / pending / in_flight；无有效视图最多等 `interaction_observation_view_wait_task_steps=6`，已入队请求仍使用原观测预算。过期未入队请求被撤销，晚到状态不能把 in_flight 降为 waiting_for_view。
- 门共识的补充视角不再受普通物体成功缓存抑制；仍保留不同视角、完整画面和三次状态确认。门框检测以可见范围而非实心面积衡量投影覆盖，保留连通分量、像素数量和短边阈值。
- 交互导航偏离同一 global path 超过 0.35 m 时，每隔至少 20 个任务步重规划一次，最多 3 次；进入位置容差后不触发这类重规划。local inflation 保持 0.45 m。
- 导航无步进兜底从 5 s 调整为 30 s，任务步超时预算不变；房间颜色在线/离线共享稳定 ID 配色，不再每 8 个 ID 重复。

按下节 PYTHONPATH 配置运行两个 ROS Python 包的完整 tests，并加上
`mlspaces_tests/test_force_interaction_bridge.py`、
`mlspaces_tests/test_semantic_video_offline.py`、
`mlspaces_tests/test_interactive_nav_runtime_regressions.py`、
`mlspaces_tests/test_organized_depth_scan.py`。

本次端到端为 H1-H10、seed=house、10 worker、horizon=500、录制开启，
每场独立 master 端口 16001-16010；输出使用新的 runtimefix 批次目录，
不能用 500 步结果直接与上一轮 1000 步的最终覆盖率作等价比较。

### 2026-09-13 批次故障回归

轻量测试覆盖真实 rospy 序列化、M1 配帧与入队、房间分裂/合并确认、
交互失败兜底、视角去重、导航客户端恢复、局部净空及视频因果时序。
无需启动仿真或 ROS master；使用已有 mlspaces 的 pytest 加载 Conda ROS 库，
避免因 rospy 缺失而跳过关键测试。

```bash
cd /home/ldl/molmospaces-exp-setting
export TMPDIR=/home/ldl/.cache/interactive-nav-regression/tmp
export XDG_CACHE_HOME=/home/ldl/.cache/interactive-nav-regression
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:$PWD/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts:$PWD/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts:$PWD/Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts:$PWD/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts:/home/ldl/conda_envs/ros-noetic/lib/python3.11/site-packages"
/home/ldl/conda_envs/mlspaces/bin/python -m pytest -q -p no:cacheprovider \
  Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/tests \
  Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/tests \
  mlspaces_tests/test_semantic_video_offline.py
```

执行真实仿真前，应先检查 M1 `image_pairing.stage=paired` 以及对象请求
`enqueued/started` 计数。`interaction_observation_timeout` 不等于 HTTP 请求超时；
`header.seq` 不等于 task step。动作兜底不会伪造 M1 ready 或 open 状态，
仍须通过物理位姿检查；抽屉扫描仍要求同帧有效的公开检测框。

长时间仿真需另行确认。单元测试和保存候选回放通过不代表物理交互成功率已恢复，
也不能证明 move_base 原生崩溃的内部原因已消除。

## 3.1 常用环境

- `mlspaces`：MolmoSpaces 与主导航流程
- `yolo_world`：本地 YOLOE / 检测测试相关

## 3.2 常用资源路径

- 资产目录：`/home/user/ldl/molmospaces/assets`
- ROS 输出目录：`/home/user/ldl/molmospaces/assets/datagen/nav_to_obj_ros_sim_v1`
- GT 导出目录：`/home/user/ldl/molmospaces/scripts/InteractiveNav/output`
- RViz 配置：`/home/user/ldl/molmospaces/nav_rviz.rviz`

## 3.3 资产环境变量

```bash
export MLSPACES_CACHE_DIR=/home/user/.cache/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/home/user/ldl/molmospaces/assets
```

检查当前解析到的目录：

```bash
python -c "from molmo_spaces.molmo_spaces_constants import DATA_CACHE_DIR, ASSETS_DIR; print(DATA_CACHE_DIR); print(ASSETS_DIR)"
```

---

## 4. MolmoSpaces 侧最小测试

## 4.1 `nav_to_obj` 数据流最小测试

适用场景：

- 检查 MolmoSpaces 环境是否可运行
- 检查 `nav_to_obj` 主流程是否正常
- 作为交互导航主线的最小基础测试

```bash
conda activate mlspaces
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1
```

指定场景数据集：

```bash
conda activate mlspaces
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1 --scene_dataset procthor-10k
```

## 4.2 场景排序测试

适用场景：

- 初步分析导航场景
- 生成可参考的 scene ranking / map 结果

```bash
conda activate mlspaces
python scripts/datagen/rank_nav_scenes.py --scene_dataset procthor-10k --data_split train --save_maps
```

## 4.3 抓取/预取场景资源

适用场景：

- 提前准备一批场景资源，从cache软链接到本地资产目录
- 避免运行时逐个下载

```bash
export MLSPACES_CACHE_DIR=/home/user/.cache/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/home/user/ldl/molmospaces/assets

source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlspaces && for i in $(seq 0 100); do python scripts/datagen/fetch_assets.py scene procthor-10k $i --split train --variant ceiling || true; done

source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlspaces && for i in $(seq 0 1000); do python scripts/datagen/fetch_assets.py scene procthor-10k $i --split val --variant ceiling || true; done

```

注意：该命令较重，不应在普通文档修改后顺手执行。

---

## 4.4 开关门 GT path获取


测试
```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MLSPACES_CACHE_DIR=/tmp/molmo-spaces-cache-proxy MLSPACES_ASSETS_DIR=/tmp/molmo-spaces-assets-proxy /home/user/miniconda3/envs/mlspaces/bin/python scripts/InteractiveNav/explore_molmo_interactions.py door-path-study --house_ind 10 --variant ceiling --target_types fridge --close_doors_on_path 0 --output_json scripts/InteractiveNav/output/door-path-study_procthor-10k_train_10_fridge
```

## 4.5 mixed 数据粗筛与 V3 精采集

适用场景：

- 从现有 `container_rough_catalog_v1` 二次生产 GT path 穿门的 crossing rough 候选
- 将单门关闭阻断容器交互目标、门前几何 path-backoff 提示仍可达记录为 `mixed_required_verified` 子集标签，而不是 rough 输入门槛
- 直接生成带真实 joint/readback、路径、可见性、成功证据和最小性证据的 `interactive_nav_v3`

正式 mixed rough 输入范围固定为 container rough 中全部有 strict pair 的 house 和全部 strict pair；禁止使用 door-only `door-required house` 结果预筛。当前全集为 712 house、2603 pair。rough 的门前点只是导航级几何提示，不是 manipulation-validated 开门位姿。

三类 rough smoke（crossing-only、required、无穿门路径）：

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-mixed-rough
python scripts/InteractiveNav/collect_mixed_rough_catalog.py \
  --house_indices 0,101,103 \
  --workers 3 \
  --output_dir scripts/InteractiveNav/output/mixed_rough_catalog_crossing_smoke_v1
```

全量 all-strict rough 扫描：

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-mixed-rough-all-crossing
python scripts/InteractiveNav/collect_mixed_rough_catalog.py \
  --workers 8 \
  --resume \
  --output_dir scripts/InteractiveNav/output/mixed_rough_catalog_all_crossing_v1
```

全量结果必须满足 `summary.json` 中：

- `selection_scope=all_container_rough_houses_with_strict_pairs`
- `candidate_selection=all_open_gt_path_crosses_interactive_door`
- `mixed_required_role=verified_subset_annotation_not_rough_input_gate`
- `door_required_house_prefilter_used=false`
- `expected_strict_house_count=712`
- `expected_strict_pair_count=2603`
- `pair_coverage_complete=true`

2026-07-15 crossing/required 拆分后的全量实测结果：

- 完成 712/712 个 strict-pair house、解析 2603/2603 个 strict pair，失败 0，`pair_coverage_complete=true`
- crossing rough：1609 pair、覆盖 456 house，pair 命中率 61.81%；真正没有任何穿门 pair 的 house 为 256 个
- required 子集：917 pair、覆盖 262 house；与旧 required-only catalog 的 917 个 `case_id` 完全一致
- crossing-only：692 pair；另有 194 个 house 有 crossing pair、但没有任何 rough required pair
- 非 crossing pair：762 对存在开放路径但不穿 interactive door，232 对所有 source start 均无开放路径
- 共检查 6087 个 source start：5513 条开放路径，其中 2895 条穿门；穿门路径中 1511 条通过 required 验证、1384 条为 crossing-only
- crossing 容器类别：Fridge 821，Dresser 788；最短 crossing 路径中位数 9.61 m、均值 10.40 m、范围 0.48--30.05 m
- required 子集路径统计保持原结果：中位数 11.19 m、均值 12.03 m
- 旧 443-house door-required 集合中只有 369 个 house 含 container crossing，另有 87 个 crossing house 位于旧集合外；旧集合只能作为历史对照

### 4.5.1 crossing-only rough 标注俯视图

用于直接检查 `1609 - 917 = 692` 个 crossing-only pair：左图为全开状态的 rough GT planner path，右图为关闭所穿门后的重新规划路径；同时标出起点、门前几何提示、粗容器目标、容器/目标/所穿门以及全部 scene object AABB。

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-mixed-rough-viz
python scripts/InteractiveNav/visualize_mixed_rough_catalog.py \
  scripts/InteractiveNav/output/mixed_rough_catalog_all_crossing_v1/mixed_rough_catalog.json \
  --candidate_type door_crossing_only \
  --max_samples 24 \
  --workers 4 \
  --output_dir scripts/InteractiveNav/output/mixed_rough_catalog_all_crossing_v1/crossing_only_visualizations
```

默认按 Fridge/Dresser、路径长度和穿门数量分层抽样，并优先使用不同 house。输出包括逐样本双面板 PNG、含完整路径点和 Oxxx 对照的 JSON、`contact_sheet_*.png`、`index.html` 与 `manifest.json`。

需要检查某个粗门前几何提示的机器人第一视角时，可指定单个 `case_id` 并增加 `--render_first_person`；输出会同时给出该门全开/关闭的 `head_camera` 对比图：

```bash
python scripts/InteractiveNav/visualize_mixed_rough_catalog.py \
  --case_id <MIXED_ROUGH_CASE_ID> \
  --max_samples 1 \
  --workers 1 \
  --render_first_person \
  --output_dir scripts/InteractiveNav/output/mixed_rough_first_person
```

2026-07-16 的 24-house 图册重放结果：24/24 成功出图、全开路径长度均与 catalog 一致；其中 house 351/candidate 427 的闭门可达性与 catalog 不一致，已在图和 sidecar 中标记 `CATALOG/REPLAY MISMATCH`，应作为粗地图边界不稳定样本复扫，而不是直接进入 fine 生产。

10 条 V3 fine smoke：

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-mixed-fine
python scripts/InteractiveNav/build_mixed_interaction_benchmark.py \
  --mixed_rough_catalog scripts/InteractiveNav/output/mixed_rough_catalog_crossing_smoke_v1/mixed_rough_catalog.json \
  --source_variants_per_pair 2 \
  --max_samples 10 \
  --max_samples_per_house 8 \
  --px_per_m 50 \
  --max_poses_per_joint 2 \
  --output_dir scripts/InteractiveNav/output/mixed_interaction_v3_smoke10
```

### 4.5.2 Full 逐步采集与 smoke

`mode: full` 不生成错误动作负轨迹，也不生成 `open_gt_control`。它按指定
executor 执行连续导航和交互，并将每个 step 的第一视角图像、动作、qpos/qvel、
phase、segment、reward、terminal 和 info 写入 `trajectory.h5`；同时为每个相机
写出同一 step 序列的 MP4。force 模式的交互 step 使用
`force effort -> mujoco.mj_step() -> joint readback`，不会直接写 qpos。

统一 full smoke 配置：

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-interactive-full
python scripts/InteractiveNav/collect_interactive_nav.py \
  --config scripts/InteractiveNav/configs/collection/procthor10k_train_full_3domain_smoke.yaml \
  --stage full
```

full 训练资格要求：

- channel：`nav_to_door`、`force_open_door`、`nav_to_target`、`terminal_observation`
- container：`nav_to_container`、`force_open_container`、`terminal_observation`
- mixed：上述两组交互 segment 和 `terminal_observation`
- H5 数组（图像、动作、状态）长度完全对齐，且至少有一个 `force_joint` step 和
  一个 terminal step
- 失败 rollout 仅保留在 `full/runs` 供诊断，不计入训练数据；汇总见
  `full/summary.json`

full 导航使用平滑后的 waypoint 作为 RBY1 holonomic base 的绝对 position target，
由 MuJoCo actuator 逐步跟踪。导航和 force 阶段会保持初始 head、左右臂、夹爪和
torso 的 qpos/qvel/position target；`lock_base_during_force: true` 还会在每个 force
step 前后恢复 base 的 x/y/yaw pose，避免交互把机器人推离操作位或让机械臂下落，
并在 force metadata 中记录锁定组和最大漂移。

full 数据时间基准默认是 `collection_hz=5`，即 `dt=0.2s`；H5 训练 step 与 MP4
视频帧使用同一频率。内部 MuJoCo/controller 可以使用更高频率，但只在每个 0.2s
采样边界保存一次。按人类步行速度 1.4m/s 的 60%（0.84m/s）计算，每个导航
采样间隔对应约 0.168m；当前场景实际导航距离约 4.666m，对应约 28 个导航 step。
冰箱门 2s 连续打开对应 10 个交互 step，目标开度从 0% 递增到100%。任务
`success_threshold`（当前 0.8）只记录成功事件，不提前结束2s交互轨迹。

复用预计算 rough 时可配置：

```yaml
rough:
  container_catalog: /path/to/container/rough_catalog.json
  mixed_catalog: /path/to/mixed/mixed_rough_catalog.json
  generate_if_missing: false
```

统一入口相关测试：

```bash
python -m pytest -q \
  mlspaces_tests/data_generation/test_collect_interactive_nav.py \
  mlspaces_tests/data_generation/test_container_interaction_benchmark.py \
  mlspaces_tests/data_generation/test_mixed_interaction_benchmark.py
```

主要输出位于配置的 `output.root`：

- `scene_manifest.json`：版本化 train house 清单
- `seeds/benchmark.json`：真实 train scene 生成的 NavToObj seed episode
- `raw/{channel,container,mixed}/benchmark.json`：三类 fine 原始 V3 数据
- `balanced/benchmark.json`：最终严格均衡 benchmark
- `balanced/audit.json`、`balanced/structure_report.md`：字段、类别、recipe、house 与占位值审计

full step 级采集使用同一入口；`full.max_episodes` 表示每个
`full.domains` 中各采集的 episode 数。先运行单个 mixed smoke：

```bash
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-interactive-full
python scripts/InteractiveNav/collect_interactive_nav.py \
  --config scripts/InteractiveNav/configs/collection/procthor10k_train_full_smoke.yaml \
  --stage full
```

三类各采一个 smoke 可改用
`scripts/InteractiveNav/configs/collection/procthor10k_train_full_3domain_smoke.yaml`。

full 输出位于 `output.root/full/`：每个 run 包含 `trajectory.h5`、`manifest.json`、
交互结果和日志。只有 `returncode=0`、H5 对齐校验通过且 `success=true` 的 run 才会
进入 `valid_trajectory_count`；导航或操作失败不会被当成 rollout 负轨迹。

### 4.5.3 PointGoal / InstructionGoal 数据生成 demo

从现有 InteractiveNav V3 channel benchmark 采一个 interaction-aware PointGoal：

```bash
conda activate mlspaces
export MUJOCO_GL=egl
python scripts/InteractiveNav/generate_point_goal_v3.py \
  <V3_BENCHMARK_OR_DIR> \
  --episode-index 0 \
  --interaction-aware \
  --px-per-m 50 \
  --output-dir scripts/InteractiveNav/output/point_goal_v3_smoke
```

直接从原始 scene split 采一个普通可达 PointGoal：

```bash
python scripts/InteractiveNav/generate_point_goal_v3.py \
  --source-mode scene_split \
  --house-ind 7 \
  --output-dir scripts/InteractiveNav/output/point_goal_raw_scene_smoke
```

基于 V3 oracle plan 规则生成 hidden/partial/explicit 三条 InstructionGoal：

```bash
python scripts/InteractiveNav/generate_instruction_goal_v3.py \
  <V3_BENCHMARK_OR_DIR> \
  --mode rule \
  --output-dir scripts/InteractiveNav/output/instruction_goal_rule_smoke
```

如果已有 full rollout，可按 segment 抽取首/中/尾关键帧并调用 VLM；`--graph-json`
可选传入 unified graph JSON，生成器只保留 GT path 周围指定半径及必需交互实体：

```bash
python scripts/InteractiveNav/generate_instruction_goal_v3.py \
  <V3_BENCHMARK_OR_DIR> \
  --mode vlm \
  --trajectory-h5 <FULL_RUN>/trajectory.h5 \
  --graph-json <GRAPH_JSON> \
  --graph-radius-m 1.0 \
  --model-mode http \
  --endpoint <OPENAI_COMPATIBLE_ENDPOINT> \
  --model <MODEL_NAME> \
  --output-dir scripts/InteractiveNav/output/instruction_goal_vlm_smoke
```

只验证 full H5 关键帧和 V3 封装、不访问外部 API 时，将上面的
`--model-mode http ...` 替换为 `--model-mode mock`。

不访问外部模型的轻量单元测试与 V3 示例校验：

```bash
conda activate mlspaces
python -m pytest -q mlspaces_tests/data_generation/test_interactive_nav_task_generation.py
python scripts/InteractiveNav/dataset_definition/v3/validate_examples.py
```

相关单元测试：

```bash
conda activate mlspaces
python -m pytest \
  mlspaces_tests/data_generation/test_mixed_interaction_benchmark.py \
  mlspaces_tests/data_generation/test_container_interaction_benchmark.py \
  mlspaces_tests/data_generation/test_visualize_mixed_interaction_benchmark.py \
  mlspaces_tests/data_generation/test_capture_mixed_gt_storyboard.py \
  -q
```

主要输出：

- rough：`mixed_rough_catalog.json`、`summary.json`、`shards/*/run.log`；支持按成功 house 断点续扫
- fine：`benchmark.json`、`valid.json`、`rejected.json`、`summary.json`
- fine builder 接收 crossing rough 后，会在真实容器交互位姿重新验证 required；crossing-only rough 不会被直接视为正式 mixed episode
- fine episode 必须通过 `validate_mixed_v3_episode`，且 `minimal_plan_verified=true`

## 4.6 mixed V3 标注俯视图

适用场景：

- 重放 mixed V3 中冻结的物体位姿、门/容器关节初态
- 对比 required door 关闭的初态与开门后的 GT 导航状态
- 标出起点、朝向、门前位姿、容器交互位姿、目标、交互对象及全部 scene object AABB
- 输出逐 episode PNG、Oxxx 物体标注 JSON、联系图和 HTML 图册

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-mixed-viz
python scripts/InteractiveNav/visualize_mixed_interaction_benchmark.py \
  scripts/InteractiveNav/output/mixed_interaction_v3_smoke10/benchmark.json \
  --output_dir scripts/InteractiveNav/output/mixed_interaction_v3_smoke10/visualizations \
  --px_per_m 100
```

仅展示部分 episode：

```bash
python scripts/InteractiveNav/visualize_mixed_interaction_benchmark.py \
  scripts/InteractiveNav/output/mixed_interaction_v3_smoke10/benchmark.json \
  --episode_indices 0,8 \
  --max_episodes 2
```

默认会校验：初态到容器交互位姿不可达、门前位姿可达、开门后 GT path 可达，以及重算长度满足 `0.35 m` 绝对误差或 `10%` 相对误差。中断后可显式加 `--reuse_existing` 复用已有 PNG/JSON。

## 4.7 mixed GT 五步 storyboard

用于从 mixed V3 benchmark 自动选择一个“单 required door + 单 Fridge + 单容器交互”的代表场景，并以独立 step 重放以下五个 GT 状态：起点（门关、冰箱关）、门前（门关）、同一门前位姿（门开）、冰箱前（冰箱关）、同一冰箱位姿（冰箱开）。每一步都先重放 episode 冻结初态，再应用目标状态并保存实际关节读回。

```bash
conda activate mlspaces
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-mixed-story
python scripts/InteractiveNav/capture_mixed_gt_storyboard.py
```

默认相机固定在机器人右肩后上方，并朝机器人前方观察。输出目录包含五张原始 RGB、五份 step sidecar、`story_steps.json`、`storyboard.png`、`pathpoints_topdown.png` 与汇总 `manifest.json`。可固定 episode 或调整镜头：

```bash
python scripts/InteractiveNav/capture_mixed_gt_storyboard.py \
  --episode_index 1 \
  --camera_behind_m 0.72 \
  --camera_right_m 0.34 \
  --camera_height_m 1.72 \
  --camera_lookahead_m 1.75
```

脚本以 `StoryStep` 和逐步 `apply state -> place robot -> place camera -> capture` 循环组织；后续连续视频采集可在相邻 step 之间插值机器人 pathpoint，并将门/冰箱关节状态替换为连续轨迹。



## 5. 导航仿真与 ROS 联调

## 5.1 启动导航仿真

适用场景：

- 运行 `run_nav_ros_sim.py`
- 联通 MolmoSpaces 与 ROS 导航桥接

```bash
conda activate mlspaces
python scripts/InteractiveNav/run_nav_ros_sim.py \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --target_types Apple \
  --task_horizon 3000 \
  --policy_dt_ms 200 \
  --ctrl_dt_ms 2 \
  --action_timeout_s 0 \
  --require_fresh_cmd_vel true \
  --require_move_base_active_for_cmd_vel true
```

`action_timeout_s=0` 表示仿真阻塞等待当前观测之后产生的新 ROS 动作；等待期间会周期性重发当前观测，因此 ROS 启动和首个规划阶段不会推进空动作 step。每次 `task.step()` 固定推进 `policy_dt_ms` 指定的仿真时间（上例为 `200ms`，即 5Hz），不按墙钟 sleep；`ctrl_dt_ms` 只控制单个 policy step 内的低层控制更新频率。

对于 `/cmd_vel_stamped`，bridge 用当前 Twist 和 `policy_dt_ms` 生成下一个 base **位置目标**（平移/转角均按 `v × dt`）；RBY1 的 position-PD 跟踪仍可能留下物理残差，所以实际位移必须以录制的 pose/trajectory 为准，不能承诺严格等于 `v × dt`。墙钟等待不会延长单个仿真 step，但标准 `move_base` 的 latest-Twist 会在 step 边界采样，ROS 调度不同仍可能改变被选中的命令和实际轨迹。

## 5.2 带 timing 日志的仿真测试

适用场景：

- 观察运行耗时
- 定位性能瓶颈

```bash
conda activate mlspaces
python scripts/InteractiveNav/run_nav_ros_sim.py \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --target_types Apple \
  --timing_log_every_n_frames 20 \
  --sim_timing_log_every_n_steps 20 \
  --step_log_every_n_steps 10
```

`RosBridgePolicy timing` 分解观测发布与动作等待；`SimLoop timing` 分解 policy、MuJoCo physics、sensor polling 和完整循环耗时。ROS 探索默认不渲染未使用的腕部 RGB/depth，相机需求调试时可传 `--include_wrist_cameras true` 恢复。

默认 `ctrl_dt_ms=2` 保持原有低层控制动态。已验证 `--ctrl_dt_ms 10` 可进一步减少 physics 耗时，但会改变控制更新频率，应作为性能实验参数单独验证，不作为探索基线默认值。

## 5.3 ROS 导航系统一键启动

适用场景：

- 启动 ROS 导航联调整套流程
- 配合 RViz、语义地图、探索包调试

```bash
conda activate mlspaces
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch
```

## 5.3.1 多 ROS Master 并行运行

并行调度器为每个 worker 创建独立的 `ROS_MASTER_URI`、`ROS_HOME`、ROS 日志目录、仿真输出目录和进程组。不同 worker 可以继续使用相同的话题名与 TF frame。

先检查分片、端口、GPU 和实际启动命令，不启动 ROS：

```bash
conda activate mlspaces
python scripts/InteractiveNav/run_parallel_ros_episodes.py \
  --house-inds 4 7 10 \
  --num-workers 2 \
  --base-master-port 11411 \
  --output-dir /home/user/ldl/molmospaces/outputs/parallel_ros_dryrun \
  --exploration-only \
  --start-explore-py \
  --task-horizon 50 \
  --dry-run
```

运行两个短场景 worker：

```bash
conda activate mlspaces
python scripts/InteractiveNav/run_parallel_ros_episodes.py \
  --house-inds 4 7 \
  --num-workers 2 \
  --base-master-port 11411 \
  --output-dir /home/user/ldl/molmospaces/outputs/parallel_ros_houses_04_07 \
  --exploration-only \
  --start-explore-py \
  --task-horizon 50 \
  --worker-timeout-s 600 \
  --resource-interval-s 2
```

如果有多个 GPU，可按 worker 指定；GPU 数量少于 worker 时会轮询复用：

```bash
--gpu-ids 0 1
```

主要输出：

- `plan.json`：worker 分片、ROS Master、GPU、命令和 RViz 连接方式
- `worker_NNN/roscore.log`：独立 Master 日志
- `worker_NNN/roslaunch.log`：导航系统和仿真日志
- `worker_NNN/worker.log`：调度器生命周期日志
- `worker_NNN/episodes.jsonl`：house/episode 的 running 与最终状态事件
- `worker_NNN/resources.jsonl`：CPU、RSS、主机内存、swap、GPU 利用率与显存采样
- `worker_NNN/status.json`：worker 退出状态与耗时
- `summary.json`：全部 worker 状态和资源统计

观察指定 worker：

```bash
ROS_MASTER_URI=http://127.0.0.1:11411 \
ROS_HOME=/tmp/molmospaces_ros/worker_000 \
rviz
```

调度器收到 `Ctrl-C` 后会依次向自己创建的 `roslaunch` 和 `roscore` 进程组发送 `SIGINT`、`SIGTERM` 和最终的 `SIGKILL`。不要使用全局 `pkill roscore` 清理，以免影响其他 ROS 任务。

单场景超时与失败重试：

```bash
python scripts/InteractiveNav/run_parallel_ros_episodes.py \
  --house-inds 0 1 2 3 \
  --num-workers 4 \
  --output-dir /home/user/ldl/molmospaces/outputs/parallel_ros_retry_example \
  --data-split val \
  --exploration-only \
  --start-explore-py \
  --task-horizon 850 \
  --scene-timeout-s 900 \
  --max-scene-attempts 2 \
  --max-consecutive-action-timeouts 12
```

- `scene-timeout-s`：单次场景 attempt 的墙钟上限。
- `max-scene-attempts`：场景执行异常或超时后的最大尝试次数。
- `max-consecutive-action-timeouts`：连续没有新 ROS 动作时提前判定本次 attempt 失败。启用场景超时且未显式设置 `action_timeout_s` 时，单次动作等待自动限制为最多5秒。
- 超时会抛出场景执行异常，当前 attempt 不保存为完成轨迹；pipeline 执行清理和 ROS reset 后进入下一次 attempt。正常跑满 horizon 的场景不会额外重试。

并行运行时为每个 worker 记录探索四宫格、第一视角、外部相机、地图轨迹和调试信息：

```bash
python scripts/InteractiveNav/run_parallel_ros_episodes.py \
  --house-inds 0 1 2 3 \
  --num-workers 4 \
  --output-dir /home/user/ldl/molmospaces/outputs/parallel_ros_debug_4w \
  --data-split val \
  --exploration-only \
  --start-explore-py \
  --task-horizon 300 \
  --record-debug
```

每个 worker 的 debug 输出位于：

```text
worker_NNN/debug/
```

该通用入口的产物取决于 `--recorder-extra-args` 与 recorder 模式；常见文件包括 `summary.json`、`final_map_trajectory.png` 和 `recorder.log`。不要假设每次都会生成 `first_person.mp4` 或 `composite_frames/`；当前语义交互主线使用下一节的 PNG+JSON 离线流程。

`--recorder-extra-args` 可继续传递 `record_explore_debug.py` 参数。调度器结束仿真后会先等待 recorder 排空异步写盘，再关闭 ROS Master；是否编码 MP4 由 recorder 的 `--runtime-video-encode` / `--offline-video-only` 显式决定。

## 5.3.2 语义交互导航与并行批测

完整交互导航优先通过统一封装脚本启动，不建议直接调用 `run_nav_ros_sim.py`：

```text
scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh
```

虽然脚本名包含 `house7`，但可通过 `HOUSE_IND` 运行其他场景。脚本会统一启动仿真、实时 GT 感知、动态语义交互图、`explore_py`、语义候选与规则或 MLLM 决策、行为执行器、`move_base`、力交互 backend，以及可选的六联图录制。运行链路为：

```text
run_nav_ros_sim.py
→ realtime GT observation
→ dynamic semantic interaction graph
→ explore_py navigation frontiers
→ semantic navigation/interaction candidates
→ semantic rule / MLLM decision
→ behavior executor
→ move_base navigation / force interaction
→ interaction_result updates graph
```

### 5.3.2.0 当前 step 同步、OCC readiness 与录像契约

`run_house7_semantic_exploration_ros_test.zsh` 及其批处理入口默认采用按 simulator step 对齐的运行方式。一个 step 的关键顺序是：

1. Bridge 以同一个 stamp 发布 odom/TF、RGB、depth/pointcloud 与可选 realtime-GT。
2. Bridge 等待 `/semantic_decision/step_ready`，再打开本 step 的 `/molmo_spaces/fresh_cmd_gate`，只接收该观察之后到达的动作。
3. 动作执行后发布 `/molmo_spaces/step_sync`；recorder 接收该标记、提交本 step 的原始快照，并通过 `/molmo_spaces/step_capture_ack` 允许仿真进入下一 step。

`step_ready` 当前聚合 `semantic_mapping` 与 `explore_py` 的 ready 消息。默认快速契约仍只等待 OCC；需要因果验收时，加载 `readiness_strict_room_graph.yaml`（以及可选的 `readiness_strict_decision.yaml`），要求观测 N 的 OCC 完成 room segmentation、unified graph 更新并完成 ROS 发布后才可 ready。严格聚合以共同的非零 `stamp` 作为 source token（中间点云节点可能重写 `header.seq`；无 stamp 时才回退到 seq），并拒绝 source 不一致；不会用不同模块的最新步号取最小值伪造同一观测。每行 `sim/step_timing.jsonl` 的 `step_ready` 保存 satisfied/timeout、期望 source、三阶段 source 与 graph revision。封装脚本默认设置 `MAP_WARMUP_SKIP_FRAMES=0`、`STEP_READY_WARMUP_SKIP_FRAMES=0`、启用 ready/capture-ack barrier，常规 timeout 为 `2s`，首次 map bootstrap 默认 `10s`；严格 smoke 应将二者提高到至少 `15s`。直接调用 `run_nav_ros_sim.py` 时，barrier 默认关闭，必须显式传参才能得到同样的契约。

严格 smoke 示例（脚本当前为 Bash 入口）：

```bash
METHOD=interactive_rule TASK_HORIZON=20 \
SEMANTIC_MAPPING_OVERRIDE=scripts/InteractiveNav/configs/semantic_decision/readiness_strict_room_graph.yaml \
SEMANTIC_DECISION_OVERRIDE=scripts/InteractiveNav/configs/semantic_decision/readiness_strict_decision.yaml \
STEP_READY_BARRIER_ENABLED=true STEP_READY_TIMEOUT_S=15 \
STEP_READY_BOOTSTRAP_TIMEOUT_S=15 SKIP_COVERAGE=true \
bash scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
  outputs/readiness_strict_smoke house7_force_route_01
```

性能验收读取同一 `roslaunch.log`：`SemanticMappingTiming` 给出
`room_worker_{overlay,segment,crop,graph_update,total}`、topology/graph cache hit 及
commit-triggered publish 耗时；`[struct_mapping] pipeline_timing` 给出 source ingress age、
filter、TF、projection、`addScan`、`updateMap` 与 map publish。不要再把 timer 轮询等待算作
room segmentation：严格模式下 room commit 会立即发布 source-pinned core bundle，`room_commit_publish_total`
应为数毫秒，且每行 `SemanticMappingCausalReady` 的 raw/room stamp 必须相同。

`pipeline_timing` 的 `filter` 是 callback 内的点云高度过滤，不是 `tf::MessageFilter` 等待。
`scan_filter_tolerance_sec=0.03` 会额外要求 `stamp + 30 ms` 的未来 TF；在 bridge 的
250 ms TF keepalive 下可造成约 200 ms source-age。入口脚本保留历史默认值 0.03，但可在
strict latency smoke 显式传 `SCAN_FILTER_TOLERANCE_SEC=0.0`。2026-08-06 的同条件 50-step
测试中，ingress 两个 20-callback window 从 212.88/200.33 ms 降至 54.58/60.53 ms，50/50
ready 且没有 transform、unsynced、stale 或 projection drop。该开关不改变点云、GMapping
算法或 source-N readiness；在作为默认值前仍须用 recorder-enabled 固定 seed 回归 OCC/costmap。

`tf_keepalive_period_s` 保持默认 0.25 s：它只在 simulator 被同步阻塞时重发 cached
odom/base/lidar TF，不发点云。tolerance=0 后没有提高它的性能必要；若将它提高到 20–50 Hz，
会放大 cached-pose 的“当前时间”重发与 ROS 调度负担。只有未来出现长阻塞期间的真实 TF stale
timeout，才应以 0.25 vs 0.10 s 的固定 seed A/B 检验，而不是直接设为 0.05 s。

NavToObjTask 会在一次 `get_and_cache_all_step_information()` 内将 visibility/reward 缓存为
同一物理状态的结果，供 reward/info/success 复用，并在 `finally` 清空；不跨 simulator step
或外部调用复用。它避免常见失败路径的约 4 次 head-camera segmentation render。2026-08-06
的 strict 50-step smoke 中，`task_sensor_polling` 为 33.09 ms（无该缓存的相邻 0-tolerance
smoke 为 94.47 ms）；应结合固定 seed recorder 回归确认导航轨迹与视图完全一致。

对 `semantic_source=realtime_gt`，Bridge 会在 policy step `N+1` 将发布 GT 时，才请求
`NavToObjTask` 在产生状态 `N+1` 的 metric transaction 内保留一个私有 head-camera
segmentation snapshot。该 ndarray 不进入 observation、ROS payload、recorder manifest 或视频；
取用前比较 model/data identity、qpos、sim time 与相机位姿，任何不一致都退回 fresh render。
交互后的 `force=True` 则先失效快照，必定 fresh render。2026-08-06 strict 50-step smoke
（`GT_STEP_INTERVAL=3`）中 17 个 GT 发布有 16 个 snapshot hit，活跃 GT 阶段由 83.30 ms
降至 61.62 ms；`task_sensor_polling` 只从 33.08 到 34.69 ms。由于不同 run 的 timeout 数
不同，应同时看正常 `cmd_vel` 子集：full step 为 304.27 → 275.29 ms。不得改回“每 step
捕获”模式：该模式会把 task sensor 平均升至约 48 ms，抵消收益。

同一 realtime-GT launch 还将 `semantic_map/subscribe_pointcloud=false`：legacy
`scene_attribute` 流不存在时，semantic mapping 不再订阅/反序列化 `/registered_scan`；detector
和 offline-GT 路径保持订阅。smoke 日志必须出现 `cloud=<disabled>`，且若误发 legacy scene
消息会节流告警而不是静默使用过期点云。Bridge 的 `step_timing.jsonl` 新增
`policy_action_wait_after_ready_ms`、`policy_fresh_cmd_after_gate_ms`、
`policy_fresh_action_after_gate_ms` 与 `policy_realtime_gt_snapshot_hit_ms`，用于区分 readiness、
move_base/DWA 产出命令和 GT 渲染三个阶段。

`DWAPlannerROS.publish_traj_pc` 与 `publish_cost_grid_pc` 默认关闭：它们是无人订阅的纯 debug
点云，不参与局部轨迹选择，也不被离线六面板读取。调试 DWA 内部候选轨迹时可显式重新开启；完整
50-step A/B 结果见下一段。

local costmap 10 Hz 不是默认优化项。2026-08-06 在同一 fixed route、snapshot/cloud 优化和
DWA debug 点云关闭条件下，5 Hz control 的 `full_step_ms=355.15`、`timeout_noop=11/50`；仅将
`local_costmap/update_frequency` 提到 10 Hz 后变为 `369.44 ms`、`20/50`。warning 数从 72 到
121 不能直接比较（deadline 从 200 降至 100 ms），但 10 Hz 组实际 loop 中位仍为 222.1 ms，
76 次超过 200 ms，且出现一次 1.86 s 尾部延迟，未达到 10 Hz 能力。虽然正常 `cmd_vel` 步的
fresh-command callback 较快（36.77 → 6.44 ms），整体控制序列更容易 timeout，故保留 5 Hz。
测试 override 为 `scripts/InteractiveNav/configs/semantic_decision/local_costmap_10hz_smoke.yaml`，
仅覆盖 local update，不改 DWA controller 或 local costmap publish frequency。

`ENABLE_COSTMAP_LATENCY_PROBE=true` 启动轻量 metadata-only probe：它不保存 PointCloud2 或
OccupancyGrid payload，只记录 `/registered_scan`、`/filtered_pointcloud`、local costmap、local
plan 和 `/cmd_vel` 的 stamp/receipt。2026-08-06 的 5 Hz、50-step probe 中，raw pointcloud
header→probe receipt 为 p50 32.67 ms，raw→filtered receipt 为 p50 1.11 ms；filtered receipt
→下一张 local costmap publish 为 p50 105.96 ms、p95 291.36 ms、max 595.81 ms。由于本仿真
ROS stamp 与 wall-clock 同 epoch，filtered header→下一 costmap header 为 p50 170.32 ms、p95
352.10 ms、max 760.51 ms。local costmap 实际 publish period 为 p50 223.16 ms、p95 386.88 ms。
这些是“第一张后续地图发布”的表观端到端延迟，不是 `Costmap2DROS::updateMap()` 内部纯计算时长，
也不能证明每张地图只消费一个点云；精确函数级耗时仍需 navigation overlay 打点。probe 增加两个
PointCloud2 元数据订阅，故只用于诊断，不用于无扰动性能主表。

1000-step house7 完整录像（`interactive_rule`、realtime-GT、strict readiness、local costmap
5 Hz、`SCAN_FILTER_TOLERANCE_SEC=0.0`）结果保存在
`/home/ldl/tmp/house7_interactive_rule_1000step_6panel_20260806_001`：
`step_timing.jsonl` 连续 0–999 共 1000 行，raw recorder 的 step boundary accepted/persisted
均为 1000、queue peak 6/64、drop/error 均为 0；离线对齐的 sim/state/input/output/exact
均为 1000，missing indexes 为空。视频为 15 fps、1440×540、66.67 s，离线渲染耗时 158.59 s，
离线分析耗时 15.89 s。

该次运行全 step 平均 `full=527.92 ms`、`policy=421.45 ms`、`task=106.42 ms`
（physics=63.78 ms、sensor=40.47 ms），1000 step 的累计 loop 约 527.92 s，对应 200 s
simulated time 的 RTF 约 0.379。`step_ready` 为 1000/1000 satisfied 且 source aligned；
action 为 `cmd_vel=400`、`timeout_noop=600`。0–399 step 尚有正常命令，约 500 step 后连续
timeout，使后半段每 step 约 600 ms；这是当前导航执行状态，不是录像截断。realtime-GT active
334 帧，其中 333 帧 snapshot hit，active GT 平均 52.49 ms。

`timeout_noop` 是 Bridge 在本 step 打开 fresh-command gate 后、等待
`action_timeout_s` 仍未收到新 `/cmd_vel` 时输出的安全 hold-pose action；它本身不是完成条件。
对已知的终端交互失败，semantic decision 现有更窄的正常退出契约：仅当 `INTERACT` 的全部
approach pose 都已被 executor 确认 `make_plan_unreachable`，且此后连续 20 个不同的
`exploration_context.observation_step` 没有可执行替代候选（至少 3 次独立观测确认）时，发布
`EXPLORATION_STALLED` / `no_executable_candidates_after_terminal_interaction_no_plan`。
启动 SCAN、活动行为或任何可执行候选都会清除该 streak；`RosCompletionMonitor` 因而通过正常
loop break 收尾 recorder、drain 和离线 MP4，而不是抛 action-timeout 异常中断录像。原 1000-step
事件回放中，step 711 的终端交互失败会在 observation step 732 请求退出，而非空转到 step 1000。

OCC pipeline 的 1000 个 pointcloud callback 全部 processed、无 stale/TF/projection drop；
但 struct-mapping 的 `updateMap` 窗口均值从早期约 4 ms 逐步升到末段约 142 ms，应与
local costmap 分开看。该次 5 Hz costmap warning 共 421 次（warning loop p50 249.8 ms、
p95 381.2 ms、max 448.3 ms）；10 Hz 残留 warning 单独归为 shutdown/global context，不并入
local 5 Hz。

运行时 recorder 只保存可重放源数据，不在线合成 panel 或编码 MP4：

- `sim_step_frames/manifest.jsonl` 与逐 step 相机 PNG；
- `debug/raw/step_boundaries.jsonl`：step、位姿、轨迹、plan、语义选择/执行状态、TF 与每类地图 receipt；
- `debug/raw/map_manifest.jsonl` 和 `debug/raw/maps/*.png`：raw/planning OCC、room segmentation、global/local costmap 及 global costmap 增量；
- `debug/summary.json`：`raw_recording_stats`，含 source/accepted/persisted/drop/failure 计数。

recorder 的 raw writer 默认以有界本地队列和 `block` 溢出策略持久化 PNG+JSON；没有 ROS bag，也不应在回调中编码视频。排空成功后，runner 调用离线构建器，以原始相机帧、step boundary 和地图 receipt 重建原有六面板风格：

```bash
python scripts/InteractiveNav/build_semantic_video_offline.py \
  --scene-dir <OUTPUT_DIR> \
  --debug-dir <OUTPUT_DIR>/debug \
  --fps 5 \
  --state-alignment exact \
  --output-stem overview_6panel
```

产物为 `videos/overview_6panel.mp4`、`videos/offline_composite_frames/`、`offline_video_summary.json` 和 `offline_render_alignment.jsonl`。验收时要求 sim manifest、raw step boundary 与离线帧数一致，`exact_step_match_count` 等于输出帧数，且 `raw_recording_stats` 无 drop/write failure。`offline_render_alignment.jsonl` 还保留每帧实际选择的 receipt 与 source stamp；如需审计组件时间关系，应检查它和 `component_post_observation_receipt_count`，不要只凭 MP4 目测推断因果顺序。

规则 oracle 的容器正面接近仅用于规则评测：
`house7_channel_container_interaction_rule.yaml` 的
`runtime.rule_oracle_gt_interaction_axis=true` 使 simulator 从**当前** container articulation joint
推导世界坐标正面轴；不加载 House7 pose/AABB 缓存，门仍使用通用 portal 几何候选。入口脚本只允许
`METHOD=interactive_rule` 开启该开关，MLLM/detector lane 会直接拒绝。`full_mllm_house7_mapping.yaml` 与
`house7_interaction_geometry.yaml` 已删除。

规则运行后的全图视觉 QA 使用（本地模型只接收一张带匿名框的完整 RGB；所有 GT 名称、AABB、joint
axis 和规则选择都只写在 held-out JSON）：

```bash
/home/ldl/miniconda3/bin/conda run -p /home/ldl/miniconda3/envs/navwam \
  python scripts/InteractiveNav/run_rule_mllm_image_qa.py \
  --run-dir <RULE_RUN_DIR> --cases-per-stage 6 --temperature 0
```

它默认导出 `mllm_image_qa/{annotated_frames,cases.jsonl,responses.jsonl,summary.json}`（可用
`--output-dir` 改名），按 Module 1
视觉属性/正面证据、Module 2 匿名候选排序、Module 3 视觉交互门控抽题。只接受
`image_step == observation_capture_step` 的图像-框配对；某个 rollout 若没有足够的新鲜决策帧，会少于
请求数量而不会用延迟框补齐。相同 object+全图 bbox 的静止重复帧会去重；`temperature=0` 用于离线记录可复现。

六联图的显示参数也随每个 raw step 的 `visualization_config` 持久化，而非一次性后处理：默认图 3（room + interaction）使用 `1.5x` 世界坐标缩放，图 5（semantic XY）使用 `1.8x`，并只标注当前交互目标和 room。`run_house7_semantic_exploration_ros_test.zsh` 可通过 `VIDEO_ROOM_PANEL_SCALE`、`VIDEO_SEMANTIC_XY_PANEL_SCALE` 与 `VIDEO_SEMANTIC_XY_LABEL_MODE` 覆盖；缩放在同一 world transform 中完成，因此 OCC、图节点、边和机器人保持对齐。

实时 restricted-GT 的门公开 ID 与展示名统一为 `door_<ordinal>`（例如 `door_0001`）；`doorframe`、
`doorway` 与 door leaf 都归一到同一个 generic door。公开图、录像和 MLLM compact context 只出现
这个 generic ID，不能出现 subtype、MuJoCo source/body name 或 `gt_*`。图中的内部 `type=portal`
仅是拓扑/候选分类，规则 lane 仍将未知 `door_<ordinal>` 作为一次可验证的交互假设；首次 action
feedback 才能将状态写为 `open`、`static_open` 或 `blocked`。MLLM lane 对 unknown door 仍须先获得
视觉 attribute patch，不能从 restricted-GT 几何推断开闭状态。私有名称只在 simulator process
内由动作 bridge 解析。

2026-08-06 的 House7 door-only rule 1000-step 验收使用
`house7_door_interaction_rule.yaml`（只保留 portal 交互候选，不改变默认通用策略）。它在 716/1000
step 以 `navigation_and_interaction_frontiers_exhausted` 正常退出：`door_0001`（step 73）和
`door_0002`（step 204）均由 `closed → open`、`articulated`、`SUCCEEDED`；`door_0003` 与
`door_0004` 分别通过动作反馈写为 `static_open`，而不是误称为可操作门体。公开 final graph
仅有 `door_0001`–`door_0004`，不含 `doorframe`、`doorway`、`gt_portal` 或 `portal_ref`。
recorder 的 step boundary accepted/persisted 均为 716、drop/error/write failure 均为空；离线视频
716 帧、exact match 716、缺失 step 为空（15 fps H.264）。产物在
`/home/ldl/tmp/house7_interactive_rule_doorx_1000step_6panel_20260806_001`。

`static_open` 是动作 bridge 对“固定开口、无可操作门体”的同步终态：行为 executor 收到该反馈后立即发布终态，绝不等待 `verification_timeout_s`。它保留 `traversable=true`，供后续真实 OCC/frontier 自然穿越；但不创建 synthetic child room、不触发 post-open OCC/room refresh，也不生成强制的门后 `traverse` subgoal。

2026-08-06 static-fix 1000-step horizon 回归输出在
`/home/ldl/tmp/house7_interactive_rule_staticfix_1000step_6panel_20260806_155451`。它在 393/1000
step 依既有无候选早停规则以 `navigation_and_interaction_frontiers_exhausted` 正常结束；`door_0003`
于 step 340 以 `static_open`、0 sim/task/physics step 直接 `SUCCEEDED`，无 `VERIFYING`、
`verification_timeout` 或 `traverse:door_0003:*`。393 张 raw step 与离线视频逐步 exact 对齐，所有
raw step 均持久化图 3=1.5、图 5=1.8、`interaction_target_only`。该路线在早停前未观测到
`door_0004`，因此本次覆盖一个 static 门的端到端回归；不应把它误称为四门全覆盖。

交互接近的 pose gate 与 `verification_timeout_s` 分离：`move_base` 完成后，executor 只轮询新鲜的
evaluator/simulator step 位姿，最多由 `interaction_approach_pose_poll_max_attempts` 限制（当前为 5），
不使用墙钟秒数截止。每个样本都按同一距离/朝向契约验证实际位姿；失败则依次重试候选中持久化的
fallback approach pose（当前最多 3 个）。选中的有效 fallback 会写入
`effective_goal_xyyaw`，并作为 bridge 的 `approach_goal_xyyaw` 和离线六联图的箭头/目标位置。若全部
pose poll 或 fallback 耗尽，决策层立即排除该 pose-sensitive candidate 并重选，而真实对象/动作失败
仍走既有 cooldown。终态同时发布 `semantic_selection.active=false`，renderer 因而不会把旧 target
复活到后续空闲帧。

2026-08-06 的 House7 通道+容器集成回归使用
`scripts/InteractiveNav/configs/semantic_decision/house7_channel_container_interaction_rule.yaml`，请求
1500 step，实际在 446 step 正常早停：`door_0001`、`door_0002` 为 articulated `SUCCEEDED`，
`door_0003`、`door_0004` 为 `static_open` `SUCCEEDED`，`refrigerator` 容器交互也为 `SUCCEEDED`。
door_0003 的实际 fallback 是 `[6.8591, 4.9671, -1.5708]`，已从接近阶段起显示在录像中；本轮没有
`interaction_pose_invalid`。最后一个 chest-of-drawers 的 3 个 approach pose 均为 `empty_plan`，
故以 `no_executable_candidates_after_terminal_interaction_no_plan` 正常收尾，而非录像或 pose gate
故障。raw recorder 的 step boundary accepted/persisted 均为 446、drop/write failure 为 0；离线视频
446 帧且 exact match 446。产物为
`/home/ldl/tmp/house7_channel_container_rule_1500step_20260806_run3`。

### 5.3.2.0.1 初始扫描与语义决策的关系

语义控制探索默认 `external_behavior_control=true`，并关闭 ExplorePy 的 legacy `initial_spin_enabled`。启动观测由 `semantic decision -> semantic executor -> bridge` 的必选、不可中断 `SCAN` 接管：map ready 前不发布候选；ready 后只发布稳定的 `SCAN`，成功前绝不放行 `EXPLORE / NAVIGATE / INTERACT`。这不是 MLLM 未返回：`interactive_rule`、`container_exploration` / `semantic_interaction_exploration` 默认是 rule backend，只有 `full_mllm_*` METHOD 才调用 MLLM。

SCAN 按 `step_index` 原子配对 `RGB(N)` 与 fresh-command gate `N`，并只在对应 `step_sync` 确认后消费下一命令。默认目标 `2π`、`1.25 rad/s`、`dt=0.2 s`、逻辑控制时间上限 `15 s`、硬上限 `40 step`；`timeout_s` 只累计已确认的 control step × `dt`，墙钟只检测命令已发出但 step-sync 不再推进的故障，不能因机器负载改变已执行轨迹。2026-08-03 的 100-step smoke 已在 28/40 个确认控制 step、`6.406 rad` 达到 `SCAN SUCCEEDED`，随后进入 `INTERACT` 和 `NAVIGATE`。调试检查 `debug/cmd_vel.csv`、`debug/events.jsonl` 的 `SCAN` feedback 与 `step_gate` 诊断。

`external_behavior_control=true` 下，`INTERACT` 由 semantic behavior executor 直接驱动 `move_base`，不必写入 ExplorePy 的 `current_subgoal` / `active_goal`。视频与离线分析应优先读取 `semantic_selection.goal_xyyaw`、`semantic_execution_state` 与 `move_base` plan，而不是将空的 ExplorePy subgoal 视为决策失败。

### 5.3.2.1 METHOD 总览

当前交接基线为分支 `codex/semantic-decision`、提交 `c9fe3c8`。最关键的任务区别是：

- **交互探索没有 object goal**。系统在 `EXPLORE / INTERACT` 候选中决策，完成条件是导航 frontier 与交互 frontier 耗尽，或达到停滞恢复条件。
- **交互目标导航必须额外提供目标来源、目标完成条件和目标优先级**。目标一旦被稳定感知，`NAVIGATE` 目标候选应抢占普通探索；完成条件是到达目标并通过配置的可见性验证。
- 两类任务使用相同 ROS 入口和底层感知、建图、导航、交互执行链路，区别主要来自 `METHOD`、决策配置和运行时目标上下文。

入口脚本当前支持以下全部 METHOD：

| METHOD | 任务与 mission | 决策模块 | 目标来源与完成条件 | 默认配置及用途 |
|---|---|---|---|---|
| `interactive_rule` | 规则交互探索；默认 `semantic_interaction_exploration` | Module 1 `dynamic_rule`；Module 2 `rule_cost`；Module 3 `rule_verified` | 无目标；按交互/导航 frontier 完成 | 不覆盖默认决策配置，适合最小规则链路 smoke test；默认不强制关闭容器 |
| `semantic_interaction_exploration` | 完整规则交互探索 | 三模块均为规则实现 | `target.enabled=false`；导航和交互候选耗尽后完成 | 自动使用 `interactive_exploration.yaml`，并强制关闭容器 |
| `container_exploration` | 与 `semantic_interaction_exploration` 相同的规则交互探索 | 三模块均为规则实现 | 无目标；按 frontier 完成 | 面向批处理脚本保留的别名，自动使用 `interactive_exploration.yaml` |
| `full_mllm_exploration` | 全 MLLM 交互探索 | `dynamic_mllm / mllm_score / mllm_skill_verified` | 无目标；LLM 在具体 `EXPLORE / INTERACT` 候选中选择 | 自动使用 `full_mllm_interactive_exploration.yaml`、`full_mllm_mapping.yaml`，并启用属性推理 |
| `full_mllm_object_goal` | 全 MLLM 交互目标导航 | `dynamic_mllm / mllm_score / mllm_skill_verified` | 默认运行时注入 `random_far_container_object`，只向决策侧公开目标物体类别；目标到达并通过可见性验证后完成 | 自动使用 `full_mllm_object_goal_runtime.yaml`、`full_mllm_mapping.yaml`，并启用属性推理，不需要外部配置覆盖 |
| `full_mllm_object_goal_apple` | House 7 固定 apple 的全 MLLM 交互目标导航 | `dynamic_mllm / mllm_score / mllm_skill_verified` | 固定目标仅为 `apple`，默认禁止运行时目标覆盖，不公开冰箱或交互要求 | 自动使用 `full_mllm_object_goal_apple.yaml` 和通用 `full_mllm_mapping.yaml`；不得加载 House 7 专用交互位姿标定 |
| `frontier_only` | 纯导航 frontier 对照 | 不启动 semantic decision | 无目标；以 frontier completion 为准 | 不执行语义候选决策和交互，用于 pure-nav baseline |
| `semantic_interaction_object_goal` | 规则交互目标导航；`semantic_interaction_object_goal` | 默认 `dynamic_rule / rule_cost / rule_verified` | 默认运行时注入 `random_far_container_object`，决策侧只收到目标物体类别；目标到达并通过可见性验证后完成 | 自动使用 `object_goal_runtime.yaml`，强制关闭容器；可通过显式 override 切换为全 MLLM |
| `object_goal_runtime` | 与 `semantic_interaction_object_goal` 相同的运行时目标导航 | 默认三模块规则实现 | 默认 `random_far_container_object`，公开上下文隐藏容器关系 | 面向批处理脚本保留的别名，自动使用 `object_goal_runtime.yaml` |
| `object_goal_rule` | 配置文件固定目标的规则导航 | 三模块均为规则实现 | 默认目标为 fridge/refrigerator；可用 `SEMANTIC_DECISION_OVERRIDE` 替换为 apple 等目标 | 自动使用 `object_goal_fridge.yaml`，不加载场景专用交互几何；严格目标完成配置应显式包含 `mission.mode=semantic_interaction_object_goal` |
| `object_goal_model_mock` | 固定 fridge 目标的模型接口测试 | Module 2 使用 `model.mode=mock`，不请求真实模型 | 固定配置目标 | 自动使用 `object_goal_fridge_model_mock.yaml`，用于验证模型协议、选择和回退链路，不用于真实 MLLM 指标 |

补充约定：

- `semantic_interaction_object_goal`、`object_goal_runtime` 和 `full_mllm_object_goal` 会在未设置 `RUNTIME_TARGET_MODE` 时自动改为 `random_far_container_object`；其余 METHOD 默认是 `none`。
- 固定目标配置必须显式设置 `RUNTIME_TARGET_MODE=none`，否则运行时目标可能覆盖配置目标。
- `full_mllm_object_goal_apple` 已在 METHOD 内固定 `RUNTIME_TARGET_MODE=none`，因此 House 7 apple 回归不需要额外设置目标模式或 YAML override。
- `random_far_container_object` 会用完整容器关系构造任务，但默认通过 `public_object_goal_context()` 只向决策/LLM 发布目标物体类别，并把 `require_interaction` 置为 `false`。容器类别、source name 和真实交互要求保存在 `target_selection.json` 的 `private_target_context` 中，用于任务构造和离线评测；只有显式启用 `reveal_container_context` 时才公开。
- 每次真实模型运行的延时、token、错误与 TPS 写入 `<output_dir>/mllm_metrics.jsonl`。
- 多场景脚本 `run_semantic_interaction_exploration_batch.py` 接受 `container_exploration`、`object_goal_runtime`、`full_mllm_exploration` 和 `full_mllm_object_goal`。容器目标队列默认使用 `full_mllm_object_goal`。

### 5.3.2.2 全 LLM 交互探索

该模式不创建 object goal，模块 1、2、3 全部使用 MLLM。下面是当前仓库可直接执行的单场景命令（使用本机路径、纯离线视频录制和本地 Qwen；`13500` 需确认未被占用）：

```bash
cd /home/ldl/molmospaces-exp-setting
RUN=/home/ldl/outputs/interactive-nav/manual_house0000_mllm_$(date +%Y%m%d_%H%M%S)
mkdir -p /home/ldl/tmp/interactive_nav_manual /home/ldl/.cache/interactive_nav_manual

TMPDIR=/home/ldl/tmp/interactive_nav_manual \
XDG_CACHE_HOME=/home/ldl/.cache/interactive_nav_manual \
CONDA_ENV=/home/ldl/conda_envs/mlspaces \
PYTHON_BIN=/home/ldl/conda_envs/mlspaces/bin/python \
SEMANTIC_MODEL_ENV_FILE=/home/ldl/molmospaces-exp-setting/.env \
MLLM_DECISION_TIMEOUT_S=30 \
SEMANTIC_ATTRIBUTE_REQUEST_TIMEOUT_S=30 \
ROS_MASTER_URI=http://127.0.0.1:13500 \
METHOD=full_mllm_exploration \
HOUSE_IND=0 SCENE_SEED=0 ROUTE_ID=house_0000 \
USE_FIXED_ROUTE=false ROUTE_NAV_CONFIG='' RUNTIME_TARGET_MODE=none \
TASK_HORIZON=500 SIM_TIMEOUT_S=3600 \
MAPPING_SCAN_SOURCE=organized_depth POINTCLOUD_STRIDE=1 \
INITIAL_DOOR_STATE=closed FORCE_CLOSE_CONTAINERS=true \
ENABLE_RECORDING=true \
bash scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh "$RUN" house_0000
```

脚本结束后会自动完成离线合成；视频为
`$RUN/videos/overview_6panel.mp4`，原始逐步图片在 `$RUN/sim_step_frames/`，链路日志和
`semantic_exploration_result.json` 在 `$RUN/` 与 `$RUN/debug/` 下。要换场景只需同时修改
`HOUSE_IND`、`SCENE_SEED`、第二个位置参数和未占用的 `ROS_MASTER_URI` 端口。

自动设置：

- 决策配置：`scripts/InteractiveNav/configs/semantic_decision/full_mllm_interactive_exploration.yaml`
- 映射配置：`scripts/InteractiveNav/configs/semantic_decision/full_mllm_mapping.yaml`
- `ENABLE_ATTRIBUTE_INFERENCE=true`
- Module 1 `dynamic_mllm`、Module 2 `mllm_score`、Module 3 `mllm_skill_verified`

注意：`METHOD=semantic_interaction_exploration` 是规则版本，不是全 MLLM 版本。

### 5.3.2.3 全 LLM 运行时交互目标导航

直接使用内置的 `full_mllm_object_goal`。该 METHOD 会自动选择全 MLLM 决策和映射配置、启用属性推理，并在未指定目标模式时采样远距离容器内物体：

```bash
ROS_MASTER_URI=http://127.0.0.1:12836 \
SEMANTIC_MODEL_ENV_FILE=/home/user/ldl/molmospaces/.env \
METHOD=full_mllm_object_goal \
HOUSE_IND=7 \
SCENE_SEED=7 \
USE_FIXED_ROUTE=false \
TASK_HORIZON=1000 \
INITIAL_DOOR_STATE=closed \
ENABLE_RECORDING=true \
SIM_TIMEOUT_S=1800 \
zsh scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
outputs/house7_full_mllm_object_goal
```

`full_mllm_object_goal_runtime.yaml` 相比全 LLM 探索配置增加或改变：

- `mission.mode=semantic_interaction_object_goal`
- 生成并保留 target `NAVIGATE` 候选
- `target_relevance_weight=3.0`
- top-K 配额为 1 个 `NAVIGATE`、3 个 `INTERACT`、4 个 `EXPLORE`
- 使用目标到达、距离容差和可见性作为完成验证
- 目标稳定出现后允许抢占正在执行的普通探索

### 5.3.2.4 规则模式示例

单场景完整规则交互探索：

```bash
cd /home/user/ldl/molmospaces-semantic-decision
conda activate mlspaces

ROS_MASTER_URI=http://127.0.0.1:11521 \
HOUSE_IND=7 \
SCENE_SEED=7 \
METHOD=semantic_interaction_exploration \
USE_FIXED_ROUTE=false \
TASK_HORIZON=1000 \
ENABLE_RECORDING=true \
INITIAL_DOOR_STATE=closed \
  zsh scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
  outputs/house7_semantic_interaction_exploration
```

单场景规则运行时 Obj-goal：

```bash
ROS_MASTER_URI=http://127.0.0.1:11522 \
HOUSE_IND=7 \
SCENE_SEED=7 \
METHOD=semantic_interaction_object_goal \
USE_FIXED_ROUTE=false \
TASK_HORIZON=1000 \
ENABLE_RECORDING=true \
INITIAL_DOOR_STATE=closed \
  zsh scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
  outputs/house7_object_goal_runtime
```

固定冰箱目标使用 `METHOD=object_goal_rule`；模型接口 mock 使用 `METHOD=object_goal_model_mock`；纯 frontier 对照使用 `METHOD=frontier_only`。常用配置位于：

```text
scripts/InteractiveNav/configs/semantic_decision/interactive_exploration.yaml
scripts/InteractiveNav/configs/semantic_decision/full_mllm_interactive_exploration.yaml
scripts/InteractiveNav/configs/semantic_decision/full_mllm_object_goal_runtime.yaml
scripts/InteractiveNav/configs/semantic_decision/full_mllm_object_goal_apple.yaml
scripts/InteractiveNav/configs/semantic_decision/full_mllm_mapping.yaml
scripts/InteractiveNav/configs/semantic_decision/semantic_controlled_explore.yaml
scripts/InteractiveNav/configs/semantic_decision/semantic_interaction_nav.yaml
scripts/InteractiveNav/configs/semantic_decision/object_goal_runtime.yaml
scripts/InteractiveNav/configs/semantic_decision/object_goal_fridge.yaml
scripts/InteractiveNav/configs/semantic_decision/object_goal_fridge_model_mock.yaml
```

### 5.3.3 冻结 V3 单 episode 可视化评测

当前 ROS benchmark 使用 `interactive_nav_v3_benchmark_eval_v15`。复用原始
`AtomicForceInteractionController` 的 smooth 状态机：转换 5 步、每抽屉打开观察
3 步、关闭并恢复视角；这些步骤真实调用 task.step、发布受限观察并进入录制，
计入动作预算和观察步号。`drawer_scan` 扫描选定容器全部抽屉，不要求 M1 region；
`drawer_open` 仍按 region 选择。单场和 batch 均默认 15 fps。门候选采用原始
unknown-state 逻辑，M1 distinct-view 到达容差为 0.15 m / 0.20 rad。
锚点阻挡确认最少连续 3 次不同地图更新，中途恢复 clear 则恢复同一锚点导航。
每次 smooth 交互的私有阶段、实际打开比例和录制步号保存在
`eval/smooth_interactions/*.json`；必须结合视频的 open/observe/close 验收，不能
只看后端 success。v15 与历史 fast 运行的步数、视频和性能结果不可混算。

上一版 evaluator 协议为 `interactive_nav_v3_benchmark_eval_v14`：受限 GT 和同状态 RGB
使用相同源时间戳（包括宏内开/关观测），交互前置条件复用原始执行器的容差与选面检查，
公开结果保留姿态失败恢复类型，V3 候选开启导航净空检查。不要把 v13/v14 结果合并计分。
执行器通过 fresh-command window 的源时间戳将 RGB 绑定到明确的任务步号，
不再把 ROS 自动改写的 `Image.header.seq` 当作任务步号；额外宏观测不会推进旋转预算。
同步 force 宏在真实执行进度点发送 command-scoped `RUNNING` 反馈，避免长抽屉扫描
被新版执行器误判为后端未确认；完成后停止反馈，仍保留进度停滞和总时长保护。
交互仍是历史 benchmark 的同步 fast 宏，不是原始 smooth 逐 task-step 交互；宏内 RGB
不额外增加 evaluator 动作步或六联图 step marker。

三类录制 smoke 使用各 domain 文件的 episode 0、固定 200 applied-step 上限，
分别选择空闲 ROS 端口和全新输出目录。例如 channel：

```bash
TMPDIR=/home/ldl/tmp/v3-smoke \
XDG_CACHE_HOME=/home/ldl/.cache/v3-smoke \
HF_HOME=/home/ldl/.cache/v3-smoke/hf \
TORCH_HOME=/home/ldl/.cache/v3-smoke/torch \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
BENCHMARK=/home/ldl/molmospaces/scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_val_release_v1_1/benchmark/channel.json \
ROS_MASTER_URI=http://127.0.0.1:16451 \
MAX_STEPS=200 MIN_STEPS=200 STEP_BUDGET_MODE=fixed FAST_EVAL=false \
bash scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_test.zsh \
  /home/ldl/outputs/interactive-nav/v3_smoke_channel_new 0
```

container/mixed 分别替换 domain 文件、端口和输出目录。验收同时检查 M1 status 的
`image_pairing.stage=paired`、`reason=matched`，episode result、真实交互结果及
`offline_video_summary.json` 的帧数/对齐状态；不能把 wrapper 退出 0 等同于任务成功。
GT 先于对应 RGB 到达时可出现短暂等待计数，需区分随后配对成功与持续配帧失败。

冻结 benchmark 的 ROS object-goal 评测使用专用单 episode 入口。它会自行启动独立 ROS master 和 ROS 算法栈；默认还会启动 recorder，并强制输出六联图视频与俯视结果图。不要把多个 episode 放进同一次调用，以免把不同 episode 的 ROS 轨迹混入同一份 recorder 产物。

```bash
ROS_MASTER_URI=http://127.0.0.1:11311 \
MAX_STEPS=1000 \
VIDEO_FPS=15 \
bash scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_test.zsh \
  outputs/v3_container_episode_1000 1000
```

该入口固定为 `METHOD=full_mllm_object_goal`：`object_goal_v3_full_mllm.yaml` 必须包含
`dynamic_mllm / mllm_score / mllm_skill_verified`。其中 `POLICY=ros_object_goal_rule`
只是受限 GT、ROS 和 opaque interaction 的评测适配器名，不是规则语义方法，不能替换为
其他 policy。需要替换 MLLM 配置时，可传 `SEMANTIC_DECISION_OVERRIDE` 与
`SEMANTIC_MAPPING_OVERRIDE`，但脚本会校验三模块仍为完整 MLLM。

V3 入口将 M1（object attribute）输出上限独立默认设为
`SEMANTIC_ATTRIBUTE_MAX_OUTPUT_TOKENS=384`，可用同名环境变量显式覆盖；该值只传给
`semantic_attribute_max_output_tokens`，不改动 M2 的 `model.max_tokens`、M3 的
`skill_max_output_tokens`/`verification_max_output_tokens`，也不改动
`full_mllm_mapping.yaml` 中 room MLLM 的 `room_mllm.max_output_tokens`。

对抽屉目标，MLLM 输出的是高层 `drawer_scan` 宏动作和可见抽屉的归一化区域；V3 对外
仍只发送 opaque `open(object_id)`，由可信评测侧私有执行“低头 → 从上到下逐格打开 →
观察/更新受限感知 → 关闭 → 下一格 → 恢复视角”。扫描期间已验证的开度和目标可见性会
被私有地计入评测，因此关闭抽屉不会被误判为交互或目标失败。空/非法视觉区域会直接
失败，不会退化成扫描全部 simulator drawer；每个 force substep 都锁定机身与上肢，异常时
会先尝试关闭已触及抽屉并恢复视角。

完成后必须检查：

- `debug/videos/overview_6panel.mp4`：ROS 相机、OCC、房间/交互、全局/局部代价图、语义图与拓扑图六联视频。该 V3 入口当前仍显式使用 legacy `--runtime-video-encode`；它不等同于上文 raw-only 离线管线，若排查录像完整性应优先迁移/复用 raw-only wrapper，而不是把两种输出目录混用。
- `eval/episodes/<episode>/episode_topdown.png`：场景底图、真实探索轨迹、起点、GT target、GT/实际交互及 oracle 路径。
- `eval/episodes/<episode>/episode_result.json`：冻结 V3 的正式评测结果。

该入口默认不重复缓存 head-camera 视频，以避免同一 episode 同时保存两份大视频。若需要保留它作为 force interaction 的补充第一视角，额外传 `RECORD_HEAD_CAMERA=true`。

马上进行并行回归、只关心正式结果和耗时定位时，可启用 `FAST_EVAL=true`。该模式只关闭 recorder、逐 step 图片、离线视频和俯视渲染，不改变 episode 顺序、timeout、最大 step 或成功判定。每个实例必须使用不同的输出目录和 ROS master 端口；脚本在端口已有 ROS master 时会直接拒绝运行，避免误接到其他实例。

```bash
# 原生 nav_to_obj；为每个实例替换 IDX、PORT 和输出目录。
IDX=0
PORT=12101
ROS_MASTER_URI=http://127.0.0.1:${PORT} \
FAST_EVAL=true EPISODE_IDX=${IDX} \
zsh scripts/InteractiveNav/run_native_nav_to_obj_eval.zsh \
  outputs/native_nav_fast_${IDX}

# 冻结 InteractiveNav V3；BENCHMARK 必须指向实际存在的 benchmark.json。
IDX=0
PORT=12102
BENCHMARK=/absolute/path/to/benchmark.json \
ROS_MASTER_URI=http://127.0.0.1:${PORT} \
FAST_EVAL=true \
bash scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_test.zsh \
  outputs/interactive_nav_v3_fast_${IDX} ${IDX}
```

V3 三类 benchmark 的批量成功率测试必须显式传入 benchmark 文件；`--no-recording`
是 `--fast-eval` 的别名，只保留 `episode_result.json` 和资源遥测，不启动 recorder、
不等待 `step_capture_ack`，因此不需要 MP4。两个本地 Qwen 副本按逻辑 worker 轮询，
每个 episode 的 dotenv 会复制 `.env` 后覆盖 `SEMANTIC_MODEL_ENDPOINT`：

```bash
V3_ROOT=/absolute/path/to/v3_release/benchmark
COMMON=(
  --workers 10
  --base-master-port 12600
  --episode-indices $(seq 0 49)
  --max-steps 2000
  --step-budget-mode fixed
  # 先保守设置；完成小批 p95 后再按实际时长收紧。
  --scene-timeout-s 7200
  --model-endpoints http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1
  --semantic-model-env-file "$PWD/.env"
  --mujoco-egl-devices 0 1
  --no-recording
)
/home/ldl/conda_envs/mlspaces/bin/python \
  scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_batch.py \
  --benchmark "$V3_ROOT/channel.json" \
  --output-dir /home/ldl/outputs/v3_channel_50 \
  "${COMMON[@]}"
/home/ldl/conda_envs/mlspaces/bin/python \
  scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_batch.py \
  --benchmark "$V3_ROOT/container.json" \
  --output-dir /home/ldl/outputs/v3_container_50 \
  "${COMMON[@]}"
/home/ldl/conda_envs/mlspaces/bin/python \
  scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_batch.py \
  --benchmark "$V3_ROOT/mixed.json" \
  --output-dir /home/ldl/outputs/v3_mixed_50 \
  "${COMMON[@]}"
```

若三类要同时跑，不要把上述三个命令都设为 10 worker（那会启动 30 个 simulator）。
当前双卡开发机可先采用总计 10 个 worker：channel=4（端口基址 12600）、container=3
（12610）、mixed=3（12620）；mixed 的 endpoint 顺序写成 `8001 8000`，其余两个使用
`8000 8001`，可得到约 5:5 的模型请求分配。三批输出目录各自保留
`resource_telemetry.csv` 和 `aggregate_metrics.json`。

`--step-budget-mode fixed` 才表示每个 episode 使用 2,000 个 applied-action step；
`dynamic` 会按路径和交互复杂度缩短上限。`--resume` 只复用带有当前
planned-invocation 签名的完整结果；benchmark、评测器/ROS launch、预算、超时、模型
环境或运行时解释器发生变化时会安全地重新执行旧 episode。历史目录若没有该签名也
不会被静默当作当前协议结果。

每轮结束后可生成包含 incomplete、no-fresh/applied、M1 错误和交互失败原因的诊断：

```bash
/home/ldl/conda_envs/mlspaces/bin/python \
  scripts/InteractiveNav/evaluation/v3_round_summary.py \
  /home/ldl/outputs/<v3-round> --format table \
  --json-output /home/ldl/outputs/<v3-round>/round_diagnostic.json
```

若已有运行中的任意批次 PID，需要额外记录整批 CPU/RAM/每卡显存和 GPU 利用率，可
并行启动：

```bash
/home/ldl/conda_envs/mlspaces/bin/python \
  scripts/InteractiveNav/monitor_batch_resources.py \
  --pid "$BATCH_PID" \
  --output-dir /home/ldl/outputs/<batch-name> \
  --interval-s 2
```

原生入口把逐 step 分项耗时写到 `debug/step_timing.jsonl`，并生成 `debug/step_timing_summary.json`；V3 入口把汇总写入 `episode_result.json` 的 `result.timing_summary`，逐决策分项位于 `trace[].timing_ms`。优先比较 `policy_act`、`base_action`、`step_precheck`、ROS action wait、传感器和 task step 的 p50/p95，以定位未知 step 耗时。

多场景交互实验使用专用批处理脚本。每个 worker 是独立子进程，拥有独立 `ROS_MASTER_URI`；场景按 round-robin 分片，worker 内部串行运行分到的场景：

```bash
python scripts/InteractiveNav/run_semantic_interaction_exploration_batch.py \
  --output-dir outputs/interaction_explore_houses_0_9 \
  --house-inds 0 1 2 3 4 5 6 7 8 9 \
  --workers 2 \
  --base-master-port 12420 \
  --task-horizon 1000 \
  --method container_exploration \
  --scene-timeout-s 1500
```

当前批处理脚本中的 `container_exploration` 对应完整语义交互探索配置；运行时 Obj-goal 使用：

```bash
python scripts/InteractiveNav/run_semantic_interaction_exploration_batch.py \
  --output-dir outputs/object_goal_houses_0_9 \
  --house-inds 0 1 2 3 4 5 6 7 8 9 \
  --workers 2 \
  --base-master-port 12520 \
  --task-horizon 1000 \
  --method object_goal_runtime \
  --scene-timeout-s 1500
```

容器内物体交互导航队列使用专用脚本。脚本先顺序扫描场景，发现严格位于可交互容器内部的物体后立即入队；扫描满 5 个场景后才启动 5 个独立 ROS worker。公开目标上下文只包含目标物体，不包含其容器身份或 `require_interaction=true`：

```bash
python scripts/InteractiveNav/run_container_goal_queue.py \
  --output-dir outputs/container_object_goal_queue \
  --house-start 0 \
  --house-count 10 \
  --workers 5 \
  --warmup-scenes 5 \
  --task-horizon 1000 \
  --scene-timeout-s 1500 \
  --gt-step-interval 5 \
  --gt-max-distance-m 6.0 \
  --gt-min-visible-pixels 16 \
  --env-file .env \
  --allow-failures
```

每个任务默认启用 `semantic_interaction_object_goal`、模块 1/2/3 的 MLLM 配置、关闭初始门和容器、快速原子交互及六联图录像。批次根目录保存 `scan_results.json`、`queue_state.json`、`aggregate_metrics.json` 和 `summary.csv`；每个任务目录保存目标选择、语义结果、MLLM 指标和 `videos/overview_6panel.mp4`。

例如 10 个场景和 2 个 worker 的分配为：

```text
worker 0: House 0, 2, 4, 6, 8
worker 1: House 1, 3, 5, 7, 9
```

批量实验可通过环境变量控制录制。完整六联图实验使用 `ENABLE_RECORDING=true`，并由统一 wrapper 进入 `--offline-video-only`：运行期只保存 `sim_step_frames/` 的原始相机 PNG/manifest，以及 `debug/raw/` 下的无损地图 PNG、JSONL metadata 与 step boundary；不会写 `debug/videos/*_frames/`、运行期 composite PNG 或运行期 MP4。每个 step 的 graph、语义候选/选择/执行状态、轨迹和 plan 都写入 step boundary 或普通 debug JSON/CSV，因此之后可替换 renderer 重新构建任意 panel。

runner 会先用 `wait_for_recorder_drain.py` 验证 raw receipt、step boundary 和 sim frame 完整性，再调用离线构建器输出 `videos/overview_6panel.mp4` 与 `videos/offline_composite_frames/`。正确性以 `offline_video_summary.json`（frame count、exact match、codec）和 `offline_render_alignment.jsonl`（每 step 的 receipt 选择）为准；raw 路径不采用 `latest` panel 补帧。要保存全部帧，请保持 raw queue 溢出策略为 `block`，并检查 `debug/summary.json.raw_recording_stats` 的 accepted/persisted 计数相等且 drop/write failure 为零。当前六联图默认最多绘制 `SEMANTIC_VIDEO_MAX_OBJECT_NODES=64` 个非房间交互节点；完整图数据仍保存于 debug/语义结果中。

原始 PNG+JSON 已具备重绘任意 panel 的全部输入；当前 raw-only 构建器已验证并支持完整 `overview` 六联图。`--panel` 仍是 legacy 已保存 panel-frame 路径，不能用于没有 `video_frames.csv` 的 raw-only 产物；为避免误导，不要用它验证当前主线。将 raw renderer 扩展为单 panel 导出是后续工作，届时可直接复用相同 raw 数据而无需重跑仿真。

只统计成功率、覆盖率和耗时的快速实验使用：

```bash
ENABLE_RECORDING=false \
python scripts/InteractiveNav/run_semantic_interaction_exploration_batch.py \
  --output-dir outputs/fast_houses_0_9 \
  --house-inds 0 1 2 3 4 5 6 7 8 9 \
  --workers 2 \
  --task-horizon 1000 \
  --method container_exploration
```

批处理常用参数：

- `--resume`：复用已有有效结果，跳过已完成场景。
- `--allow-failures`：允许部分场景失败而不让批处理整体返回非零状态。
- `--dry-run`：只输出分片、端口和命令，不启动仿真。
- `--memory-sample-interval-s`：资源采样间隔，默认 `2s`。
- `--base-master-port`：worker 端口起点，第 `N` 个 worker 使用 `base + N`。

主要输出包括每个 `house_NNNN/` 下的 `batch_task.log`、`memory_by_step.csv`、`semantic_exploration_result.json` 和可选的 `videos/overview_6panel.mp4`，以及批次根目录的 `summary.json`、`summary.csv` 和 `aggregate_metrics.json`。

内存建议：保存六联图时优先使用最多 `2 workers`；关闭录制后可从 `2 workers` 开始逐步增加。不要直接以 CPU 核数决定 worker 数量，应以 simulator、recorder 和主机可用内存的实测峰值为准。

## 5.4 语义地图调试启动

适用场景：

- 单独调试 `semantic_mapping_py_pkg`
- 检查 unified graph、navigation hints 与相关 topic

```bash
conda activate mlspaces
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch semantic_mapping_py_pkg semantic_mapping_debug.launch
```

## 5.5 RBY1 初始机械臂姿态验证

适用场景：

- 调试探索过程中机器人手臂碰撞墙体、门框或家具的问题
- 固定机器人位置和外部相机，比较不同初始手臂姿态
- 保存正视图和侧视图，作为后续导航探索配置依据

当前推荐初始姿态为 `shoulder_roll_in_045`，该姿态主要做水平方向贴近身体，不做明显前后或上下抬放：

```text
left_arm_qpos  = 0.28,0.0,-0.45,-0.64,0.39,-0.26,-0.04
right_arm_qpos = 0.28,0.0, 0.45,-0.64,0.39,-0.26,-0.04
```

该姿态已经配置为 `Interactive-Nav-SG-nav/src/nav_pkg/launch/molmospaces_nav_system.launch` 中的默认左右臂初始姿态。启动完整导航系统时默认生效：

```bash
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch start_explore_py:=true
```

如需重新生成固定外部相机快照，可关闭 mapping/nav/explore，只启动仿真并保存 reset 后的 debug snapshot。侧视图示例：

```bash
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch \
  start_mapping:=false \
  start_nav:=false \
  start_explore:=false \
  start_explore_py:=false \
  start_semantic_mapping:=false \
  exploration_only:=true \
  publish_debug_front_camera:=true \
  house_ind:=1 \
  target_types:="" \
  task_horizon:=20 \
  sim_extra_args:="--immediate_noop_after_publish --timing_log_every_n_frames 0 --fixed_robot_xyyaw 6.7,7.8,-0.53 --fixed_debug_camera_pos 5.9,6.9,1.25 --fixed_debug_camera_target 6.7,7.8,0.85 --initial_left_arm_qpos 0.28,0.0,-0.45,-0.64,0.39,-0.26,-0.04 --initial_right_arm_qpos 0.28,0.0,0.45,-0.64,0.39,-0.26,-0.04 --debug_snapshot_path /home/user/ldl/molmospaces/outputs/important_results/rby1_initial_arm_pose_shoulder_roll_in_045/side_view.png"
```

正视图只需要替换固定相机位置：

```bash
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch \
  start_mapping:=false \
  start_nav:=false \
  start_explore:=false \
  start_explore_py:=false \
  start_semantic_mapping:=false \
  exploration_only:=true \
  publish_debug_front_camera:=true \
  house_ind:=1 \
  target_types:="" \
  task_horizon:=20 \
  sim_extra_args:="--immediate_noop_after_publish --timing_log_every_n_frames 0 --fixed_robot_xyyaw 6.7,7.8,-0.53 --fixed_debug_camera_pos 8.15,6.95,1.15 --fixed_debug_camera_target 6.7,7.8,0.85 --initial_left_arm_qpos 0.28,0.0,-0.45,-0.64,0.39,-0.26,-0.04 --initial_right_arm_qpos 0.28,0.0,0.45,-0.64,0.39,-0.26,-0.04 --debug_snapshot_path /home/user/ldl/molmospaces/outputs/important_results/rby1_initial_arm_pose_shoulder_roll_in_045/front_view.png"
```

重要测试结果归档目录：

```text
/home/user/ldl/molmospaces/outputs/important_results/rby1_initial_arm_pose_shoulder_roll_in_045
```

---

## 6. 手动控制与辅助调试

## 6.1 手动控制模式

适用场景：

- 不跑自动策略，手动驱动机器人
- 检查底盘响应、topic 连通和 ROS bridge 是否正常

启动手动模式：

```bash
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch manual_control:=true
```

启动手动控制节点：

```bash
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
rosrun nav_pkg manual_cmd_vel.py
```

控制键：

- `w`：前进
- `s`：后退
- `a`：原地左转
- `d`：原地右转
- `space`：停止
- `+/-`：调速度
- `q`：退出

说明：脚本直接发布 `/cmd_vel_stamped`，由 `RosBridgePolicy` 读取并驱动机器人移动。

## 6.2 机械臂位置调试

适用场景：

- 调试左臂位置或 debug policy 模式

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlspaces && \
python scripts/datagen/run_nav_ros_sim.py \
  --viewer \
  --policy_mode left_arm_debug \
  --left_arm_joint_delta 0.05 \
  --debug_loop_episodes 50 \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 0 \
  --target_types Apple \
  --task_horizon 2000
```

## 6.3 RViz

```bash
rviz -d /home/user/ldl/molmospaces/nav_rviz.rviz
```

---

## 7. 地图、探索与导航子系统

## 7.1 占据地图

```bash
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch struct_mapping_pkg slam_gmapping.launch
```

配置文件：

- `Interactive-Nav-SG-nav/src/struct_mapping_pkg/config/slam_gmapping_params.yaml`

## 7.2 探索策略包

```bash
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch explore_pkg explore_manager.launch
```

配置文件：

- `Interactive-Nav-SG-nav/src/explore_pkg/config/exploration_planner_params.yaml`

## 7.3 导航控制

```bash
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg nav.launch
```

## 7.4 地图/策略统一重置

```bash
source ./Interactive-Nav-SG-nav/devel/setup.zsh
rostopic pub -1 /nav_system/reset std_msgs/Empty "{}"
```

---

## 8. 检测与语义交互图测试

## 8.1 本地检测可视化测试

适用场景：

- 测试 YOLOE prompt-free 检测
- 检查 2D 检测与 3D box 投影结果
- 快速验证 `semantic_mapping_py_pkg` 检测侧是否正常

```bash
conda activate yolo_world
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch semantic_mapping_py_pkg object_detection_visual_test.launch \
  rgb_path:=/home/user/ldl/molmospaces/detection_models/tum_rgbd_scribble_samples/kitchen_22_image.png \
  depth_path:=/home/user/ldl/molmospaces/detection_models/tum_rgbd_scribble_samples/kitchen_22_depth.png \
  backend:=yoloe_pf_box3d \
  provider:=yoloe_local \
  model_path:=/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26x-seg-pf.pt
```

## 8.2 GT 导出测试

适用场景：

- 生成场景 GT 导出文件
- 为后续 `semantic_mapping_gt_replay.py` 提供输入

输出目录：

- `/home/user/ldl/molmospaces/scripts/InteractiveNav/output`

```bash
conda activate mlspaces
python scripts/InteractiveNav/read_scene_room_properties.py \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --variant base
```

## 8.3 GT replay 测试

适用场景：

- 用 GT 结果模拟 detector-like 输入
- 调试 unified graph、room context、navigation hints

```bash
conda activate mlspaces
source ./Interactive-Nav-SG-nav/devel/setup.zsh
python Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/semantic_mapping_gt_replay.py \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/output/procthor-10k_train_1_base_scene_full.json \
  --batch-size 4 \
  --publish-rate 1.0
```

## 8.4 实时 GT 语义交互图

只使用当前 `head_camera` segmentation 中可见对象作为 GT observation：

```bash
conda activate mlspaces
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch \
  start_semantic_mapping:=true \
  semantic_source:=realtime_gt \
  publish_realtime_gt:=true \
  start_explore_py:=true
```

检查实时输入与动态图：

```bash
rostopic echo -n 1 /semantic_mapping/gt_observations
rostopic echo -n 1 /semantic_mapping/unified_graph
rosservice call /semantic_mapping/save_graph
```

实时 GT 默认每 `3` 个仿真 step 采样一次，单次扫描 segmentation 中的 geom ID，最大观测距离默认 `6m`。
三场景完整验收默认每场景录制 `600s`：

```bash
scripts/InteractiveNav/run_semantic_gt_three_scene_test.zsh outputs/semantic_gt_three_scene_10min
```

当前推荐的语义交互录像不在线合成六联图。运行时保存每个 simulator step 的原始相机与地图/语义 JSON，结束并完成 recorder drain 后再离线生成六联图；视频时长由 `frame_count / VIDEO_FPS` 决定，而不是运行时的墙钟录制秒数。历史 `run_semantic_gt_three_scene_test.zsh` 仍有 legacy recorder 参数，若继续使用它，须以该脚本生成的 `offline_video_summary.json` 和帧数校验为准，不能再把“1 FPS 在线关键帧合成”当作当前基线。

前期短测可覆盖运行时间和场景：

```bash
HOUSE_INDS="4" RECORD_SEC=60 TASK_HORIZON=800 \
  scripts/InteractiveNav/run_semantic_gt_three_scene_test.zsh outputs/semantic_gt_debug_short
```

开启 semantic mapping 后，global costmap 自动改读 `/semantic_mapping/planning_occ_map`；semantic 节点仍从原始 `/struct_mapping/occ_map` 构建房间与语义信息，避免处理后的地图反馈回自身。确认的 portal 达到 `open` 后，semantic 层只会在 planning OCC 上清除内缩的窄门洞 slab（默认参考 AABB 内缩 5 cm、厚度上限 25 cm），不会清空整块门叶 AABB、不会改写 raw OCC，也不会为冰箱/抽屉等 container 生成清除掩码。`/semantic_mapping/door_clear_mask` 与小范围 `/semantic_mapping/planning_occ_map_updates` 仅传播这个门洞区域；门关闭后，原区域会短期重复发布以覆盖 global costmap 的低频更新周期。已确认的交互结果优先于后续不稳定的视觉状态，直到下一次交互结果更新。

## 8.5 门状态与语义 OCC 快速测试

定向单测：

```bash
conda activate mlspaces
pytest -q \
  Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/tests/test_portal_state_tracker.py \
  Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/tests/test_semantic_occ_overlay.py \
  Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/tests/test_interaction_graph_store.py
```

7 号场景检查会在 MuJoCo 内部设置真实门关节，但只向 graph 发布最小 GT 五字段；
随后以 executor 语义结果 `object_id + state` 回写闭合/打开状态，并验证门图状态和 OCC 清空结果：

```bash
conda activate mlspaces
python scripts/InteractiveNav/test_semantic_door_occ_house7.py \
  --house-ind 7 \
  --output /tmp/semantic_door_occ_house7.json
```

正式 ROS 联调会把机器人固定在目标门前，初始化时直接关闭全部门，第 `OPEN_STEP` 个仿真 step 直接将目标门铰链设为全开。测试同时保存 raw OCC、semantic planning OCC、door clear mask、move_base global costmap、闭/开门图状态和对比图：

```bash
conda activate mlspaces
OPEN_STEP=100 \
  scripts/InteractiveNav/run_semantic_door_occ_house7_ros_test.zsh \
  outputs/semantic_door_occ_house7_ros
```

测试 House 7 中相邻的双开门（两块门板同时关闭/打开）：

```bash
OPEN_STEP=100 \
TARGET_ROOT=doorway_ada234694d8669f8c477500ae8f01b1a_1_0_4 \
ROBOT_XYYAW=4.20,4.881,0.0 \
  scripts/InteractiveNav/run_semantic_door_occ_house7_ros_test.zsh \
  outputs/semantic_door_occ_house7_double_ros
```

双开门 5 位姿开门、穿门、回头关门测试。这里直接设置机器人 base 位姿，从而同步改变相机、odom 和 TF，而不是只修改渲染相机：

```bash
TARGET_ROOT=doorway_ada234694d8669f8c477500ae8f01b1a_1_0_4 \
ROBOT_XYYAW=4.05,4.881,0.0 \
POSE_SEQUENCE='0:4.05,4.881,0.0,closed,left_far/75:4.35,4.881,0.0,open,left_near/150:4.99,4.881,0.0,open,doorway/225:5.55,4.881,0.0,open,right_forward/300:5.85,4.881,3.1415926536,closed,right_turnback' \
EXPECTED_PHASES=5 \
  scripts/InteractiveNav/run_semantic_door_occ_house7_ros_test.zsh \
  outputs/semantic_door_occ_house7_double_pose5
```

该模式只有在每个位姿的关节状态、语义图状态、机器人位姿、raw/planning OCC 和 global costmap 均满足条件并稳定发布后才截取。输出包括 `occ/phase_*.npz`、每阶段图 JSON、`occ/pose_sequence_summary.json` 和 `occ/pose_sequence_occ_comparison.png`。

验收条件：

- 关门时 raw OCC 与 planning OCC 在门区域一致，clear mask 为空。
- 开门时 clear mask 非空，mask 内 planning OCC 全部为 free。
- raw OCC 仍允许保留静态门板痕迹，证明清空来自 semantic overlay，而不是 GMapping 自行更新。
- move_base global costmap 的门区域收到增量更新，至少产生可通行 free 单元且不再含 lethal 单元。

2026-07-16 的 House 7 正式回归结果：门洞 mask 为 `65` 个栅格；raw OCC 仍有 `16` 个 occupied 和 `6` 个 unknown；planning OCC 的 `65/65` 个栅格均为 free；global costmap 有 `57` 个门洞栅格发生变化，最终为 `53` 个 free、`12` 个 inflation/inscribed、`0` 个 lethal。完整结果见输出目录中的 `occ/summary.json` 与 `occ/door_occ_comparison.png`。

同日双开门 5 位姿严格回归通过：开门后 `115/115` 个 mask 栅格在 planning OCC 中均为 free；左侧近门位姿的 global costmap 为 `107` free、`8` inflation、`0` lethal、`0` unknown；门洞内和右侧位姿均为 `101` free、`14` inflation、`0` lethal、`0` unknown。最终在右侧回头并关门后，clear mask 恢复为 `0`，planning OCC 在参考门区恢复 `31` 个 non-free 栅格，global costmap 恢复 `25` 个 lethal 栅格。

运行中的 ROS 话题检查：

```bash
rostopic echo -n 1 /semantic_mapping/planning_occ_map/info
rostopic echo -n 1 /semantic_mapping/planning_occ_map_updates
rostopic echo -n 1 /semantic_mapping/door_clear_mask/info
rosparam get /move_base/global_costmap/static_layer/map_topic
```

注意：当前 GT 快速版依赖“首次有效门板关节观测发生在交互前、门处于闭合状态”的任务约束；真实场景中的门关节/转轴提取尚未实现。


### 8.5.1 House 7 冰箱内物体 Obj-goal 图像采集

House 7 的冰箱对象为 `refrigerator_4d8cd69ca487b76cae801cfb0248a055_1_0_6`，此前 GT 语义图中确认其内部有 `potato`、`apple` 和 `lettuce`。下面以 `apple` 为目标，初始固定在 `house7_force_route_01` 的起点，门和容器关闭，由规则语义决策自主探索、开门、打开冰箱并导航到苹果。

```bash
ROS_MASTER_URI=http://127.0.0.1:12835 \
METHOD=object_goal_rule \
HOUSE_IND=7 \
USE_FIXED_ROUTE=true \
ROUTE_ID=house7_force_route_01 \
TASK_HORIZON=1000 \
INITIAL_DOOR_STATE=closed \
SEMANTIC_DECISION_OVERRIDE=scripts/InteractiveNav/configs/semantic_decision/object_goal_apple.yaml \
GT_STEP_INTERVAL=1 \
GT_MAX_DISTANCE_M=6.0 \
VIDEO_FPS=15 \
VIDEO_PANEL_WIDTH_PX=640 \
EXTERNAL_VIDEO_WIDTH_PX=1024 \
ENABLE_EXTERNAL_VIDEO=true \
VIDEO_FRAME_JOB_QUEUE_SIZE=1024 \
ARTIFACT_WRITE_QUEUE_SIZE=4096 \
VIDEO_HISTORY_SIZE=1024 \
IMAGE_QUEUE_SIZE=16 \
OBSERVATION_QUEUE_SIZE=16 \
EXTRA_IMAGE_QUEUE_SIZE=16 \
RECORDER_SHUTDOWN_GRACE_S=600 \
CLEAN_INTERMEDIATE=false \
SIM_TIMEOUT_S=1800 \
  scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
  outputs/house7_object_goal_apple_route01
```

主要输出：`videos/overview_6panel.mp4`、`debug/semantic_keyframes/`、`debug/graph/` 和 `semantic_exploration_result.json`。

模块 2 使用 LLM 直接选择具体 `NAVIGATE / INTERACT / EXPLORE` subgoal 时，使用独立配置保留规则基线。模型连接参数继续从仓库 `.env` 或 `SEMANTIC_MODEL_*` 环境变量读取：

```bash
ROS_MASTER_URI=http://127.0.0.1:12835 \
METHOD=semantic_interaction_object_goal \
RUNTIME_TARGET_MODE=none \
HOUSE_IND=7 \
USE_FIXED_ROUTE=true \
ROUTE_ID=house7_force_route_01 \
TASK_HORIZON=1000 \
INITIAL_DOOR_STATE=closed \
SEMANTIC_DECISION_OVERRIDE=scripts/InteractiveNav/configs/semantic_decision/object_goal_apple_module2_mllm.yaml \
GT_STEP_INTERVAL=1 \
GT_MAX_DISTANCE_M=6.0 \
VIDEO_FPS=15 \
VIDEO_PANEL_WIDTH_PX=640 \
CLEAN_INTERMEDIATE=false \
SIM_TIMEOUT_S=1800 \
  scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
  outputs/house7_object_goal_apple_module2_mllm_route01
```

该配置固定 `mission.mode=semantic_interaction_object_goal`，并用 `RUNTIME_TARGET_MODE=none` 禁止运行时随机容器目标覆盖，避免向模型泄露目标容器或强制交互信息。仅消融模块 2：模块 1 保持 `dynamic_rule`，模块 3 保持 `rule_verified`。决策 trace 中应满足正常模型路径的 `model_selected_candidate_id == executed_candidate_id`；若模型响应期间候选失效，则会记录 `candidate_validation_reason` 和 `stale_fallback_used`。

House 7 中仅公开 `apple` 类别、模块 1/2/3 全部使用 MLLM 的隐式容器发现测试使用：

```bash
ROS_MASTER_URI=http://127.0.0.1:12836 \
SEMANTIC_MODEL_ENV_FILE=/home/user/ldl/molmospaces/.env \
METHOD=full_mllm_object_goal_apple \
HOUSE_IND=7 \
SCENE_SEED=7 \
USE_FIXED_ROUTE=true \
ROUTE_ID=house7_force_route_01 \
TASK_HORIZON=1000 \
INITIAL_DOOR_STATE=closed \
ENABLE_RECORDING=true \
ENABLE_EXTERNAL_VIDEO=true \
EXTERNAL_VIDEO_OVERLAY=true \
PAPER_FRAME_EXPORTS=true \
EXTRA_IMAGE_QUEUE_SIZE=0 \
GT_STEP_INTERVAL=1 \
GT_MAX_DISTANCE_M=6.0 \
GT_MIN_VISIBLE_PIXELS=16 \
VIDEO_FPS=15 \
VIDEO_PANEL_WIDTH_PX=640 \
EXTERNAL_VIDEO_WIDTH_PX=1024 \
CLEAN_INTERMEDIATE=false \
SIM_TIMEOUT_S=1800 \
  zsh scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh \
  outputs/house7_full_mllm_apple_hidden_container
```

`METHOD=full_mllm_object_goal_apple` 内部选择的 `full_mllm_object_goal_apple.yaml` 不包含 apple 实例 ID、冰箱 ID、容器类别或强制交互提示。它只使用通用 `full_mllm_mapping.yaml`；已删除 `full_mllm_house7_mapping.yaml` 与 `house7_interaction_geometry.yaml`，因此全 MLLM 路线不会加载 House 7 已校准的容器正面位姿、轴或 AABB。运行时目标采样同样默认只向 `/semantic_decision/target` 发布目标物体类别；完整容器关系保存在 `target_selection.json` 的 `private_target_context` 和候选记录中，仅用于任务构造与离线评测。只有队列脚本显式传入 `--reveal-container-context` 时才公开容器上下文。

论文逐帧导出使用 `PAPER_FRAME_EXPORTS=true`。该模式通过
`/molmo_spaces/step_sync` 为每个 sim step 保存一组严格对齐的图片，并区分“论文静帧”和
“带调试信息的视频帧”：

- 无文字外部相机 PNG（`1024x576`）：`debug/videos/external_camera_frames/`
- 带上方诊断文字、用于外部相机视频的 PNG：`debug/videos/external_camera_overlay_frames/`
- 模拟器原始分辨率、无文字外部相机 PNG：`debug/videos/external_camera_raw_frames/`
- 无上方白条、无标题/step、无右下角 step 的图 3：`debug/videos/room_interaction_frames/`
- 无上方白条、无标题/step、无右下角 step 的图 6：`debug/videos/semantic_topology_frames/`
- 保留标题、任务目标、当前 subgoal 和 step 信息的六面板视频：`videos/overview_6panel.mp4`
- 保留上方诊断文字的外部相机视频：`debug/videos/external_camera.mp4`

2026-07-28 成功长测输出：
`outputs/house7_full_mllm_apple_paper_1000_20260728_v2`。任务只公开 `apple` 类别，
冰箱在 step 835 成功打开，apple 在 step 934 满足可见性完成条件，保留 10 个成功后帧并在
step 944 提前结束；`target_goal_success=true`、
`target_object_visible_navigation_success=true`、`overall_success=true`。六面板、外部相机、
图 3、图 6、sim-step 与离线 composite 均保存 944 张/帧，两个写入队列丢帧数均为 0，
离线六面板对齐为 944/944 exact step match。

2026-07-27 的固定 House 7 冰箱正面位姿回归已废弃：原先的场景专用标定 YAML 已删除，不得再将
`[8.254459, 1.053060, 3.141593]` 作为加载配置。规则 oracle 评测如需正面接近，
`house7_channel_container_interaction_rule.yaml` 可显式启用
`runtime.rule_oracle_gt_interaction_axis=true`；运行时仅从当前仿真 articulation joint 推导容器的
world-frame 正面轴，不存储场景 ID、固定 pose 或 AABB，且该开关会被 MLLM/detector 路线拒绝。后续外部相机默认
采用手动调好的 `CAMERA_REL`：position `[-0.779295308248162, 0.9640243904369644,
1.600000023841858]`、yaw `-34.729731964609215 deg`、pitch
`-16.0519210588521 deg`、FOV `65 deg`。运行脚本中的等价 look-at offset 为
`[0.010510587056107079, 0.4165302897166183, 1.3234916922873792]`。相机水平距离
约 `1.24 m`。原始外部相机 PNG 与视频均为 `1024x576`。节点图右侧决策文字框已移除，
room 横向排列优先依据 portal-room 拓扑连接度，将 House 7 的 livingroom 放在
bedroom 与 kitchen 之间。

2026-07-28 House 7 论文首图相机回归：左后上方位姿会让机器人位于画面右侧，
同时保持冰箱正面和内部物体可见。当前稳定跟随相机配置为
`position_robot=[-1.45, 1.30, 1.90]`、
`lookat_offset=[0.05, 0.40, 1.38]`、FOV `65 deg`，由
`scripts/InteractiveNav/run_house7_semantic_exploration_ros_test.zsh` 默认使用。
House 7 apple 800-step 回归输出为
`outputs/house7_apple_rear_left_compromise_800_20260728_v4`，最终 apple 目标成功，
外部最终 PNG 为 `1024x576`。

#### 手动移动机器人并调外部相机

历史手动可视化工具位于独立实验 worktree：

- `/home/user/ldl/molmospaces-exp-setting/scripts/InteractiveNav/run_manual_interactive_nav_test.py`
- `/home/user/ldl/molmospaces-exp-setting/scripts/InteractiveNav/manual_interactive_nav_camera.py`
- `/home/user/ldl/molmospaces-exp-setting/scripts/InteractiveNav/manual_interactive_nav_policy.py`

启动 House 7 冰箱内 apple 场景：

```bash
cd /home/user/ldl/molmospaces-exp-setting
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate mlspaces
export MPLCONFIGDIR=/tmp/molmospaces-matplotlib
python scripts/InteractiveNav/run_manual_interactive_nav_test.py \
  --catalog scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_train_100/raw/mixed_rough/mixed_rough_catalog.json \
  --case_id 'mixed_h7__refrigerator_4d8cd69ca487b76cae801cfb0248a055_1_0_6__apple_87e4661e0aaedff69f751b5ac78bd93c_1_0_6' \
  --state_preset visualization \
  --camera_mode robot_over_shoulder \
  --camera_position_offset_robot -0.779295308248162 0.9640243904369644 1.600000023841858 \
  --camera_lookat_offset_robot 0.010510587056107079 0.4165302897166183 1.3234916922873792 \
  --camera_fov 65 \
  --camera_translation_step 0.05 \
  --camera_rotation_step_deg 1.0 \
  --capture_dir /tmp/house7_manual_camera_tune
```

控制键：`W/S/A/D` 移动机器人，`I/K/J/L` 平移相机，`;/'` 调 yaw，`./` 调
pitch，`O/P` 打开/关闭交互对象，`V` 截图，`C` 重置相机，`R` 重置机器人，
`Esc` 退出。调好后终端打印的 `CAMERA_REL` 可直接记录；yaw/pitch 转成跟随相机
look-at 时，只需用 `lookat = position + [cos(pitch)cos(yaw), cos(pitch)sin(yaw),
sin(pitch)]`。

注意：跟随相机配置不能同时设置固定 `camera_quaternion`。`RobotMountedCamera` 在
`camera_quaternion` 非空时会忽略 `lookat_offset`；此前正式仿真仍传入
`[0.5, 0.5, -0.5, -0.5]`，实际视线因此被固定为机器人坐标系正前方 `[1, 0, 0]`，
没有使用遥控记录的 yaw/pitch。该覆盖已移除。上述 CAMERA_REL 对应的正确机器人系
forward 为 `[0.7898058953, -0.5474941007, -0.2765083316]`，正式跟随相机现在由
position 与 look-at 两个机器人系 offset 生成，与遥控脚本保持一致。

2026-07-27 耗时回归：三路并行 800-step 测试输出位于
`outputs/house7_perf_parallel3_800_20260727_w{1,2,3}`。三次均完成 800 个 sim step、
800 行 `sim/step_timing.jsonl`、800 张 sim-step PNG 和 800 帧 6-panel 视频；主循环
均值分别为 `1.031`、`1.042`、`1.023 s/step`。合并稳定阶段日志显示，外部相机
同步 ROS 发布平均占 `510 ms/step`，而 sim-step PNG 队列入队仅占 `0.03 ms/step`。

性能修复后，第一视角和外部相机分别使用独立的异步 ROS publisher queue，外部相机
回调不再与 6-panel 渲染共用锁，等待导航命令期间也不再重复发布录像 RGB。最终
100-step 回归位于 `outputs/house7_perf_short_all_async_lossless_100_20260727`：稳定阶段
平均 `0.708 s/step`，外部图像 ROS 发布由 `441 ms` 降至 `1.48 ms`；100 个 sim step
对应 100 张 sim-step PNG、100 张外部原始 PNG 和 100 张 topology panel PNG，且
`artifact_write_dropped_jobs=0`、`video_frame_jobs_dropped=0`。

2026-07-27 apple object-goal 修复：原先实时 GT 消息会携带每个实例的完整像素坐标，
1024x576 下单条 ROS JSON 可达到约 `0.6-1.3 MB`；同时 GT publisher 复用了深度
为 16 的通用 ROS 队列，导致冰箱打开后的新目标观测排在旧消息后。现在 GT 只发布
`bbox_2d + visible_pixels + visible_fraction + box_3d`，发布端和语义映射订阅端均改为
latest-only 队列 1。三路最终测试中，apple 从原始 GT 捕获到节点图 `NEW_NODE` 的
墙钟延迟降为 `7.8-9.1 s`，此前并行长测为 `29-53 s`。

该次回归曾通过精确 apple 与 refrigerator 绑定验证目标状态机。合并后的通用配置仅暴露
apple 类别，不再向决策模块提供目标容器 ID 或强制交互信息；目标一旦稳定进入图中，便由
事件驱动优先级抢占探索。容器内目标复用图关系推导出的父容器及其已验证交互位姿，不再
根据 apple 几何中心生成可能位于冰箱内部或侧面的导航点；机器人已在该位姿时直接进入
可见性验证，不再被无位移看门狗误判。完成后发布 `target_goal_succeeded` 并触发仿真提前退出。

三路最终并行输出：

- `outputs/house7_apple_latest_parallel3_800_20260727_w1`：step 451 成功，保留 10 step 后在 461 结束。
- `outputs/house7_apple_latest_parallel3_800_20260727_w2`：step 339 成功，保留 10 step 后在 349 结束。
- `outputs/house7_apple_latest_parallel3_800_20260727_w3`：step 385 成功，保留 10 step 后在 395 结束。

三路均满足：冰箱交互成功、交互位姿验证通过、apple 当前稳定可见、目标导航成功、
`target_goal_success=true`、`target_container_interaction_success=true`、
`overall_success=true`。`sim_step_frames/manifest.jsonl` 与最终 6-panel 视频帧数一一对应，
且 `video_frame_jobs_dropped=0`、`artifact_write_dropped_jobs=0`、渲染队列覆盖次数为 0。
外部相机原始帧在 `debug/videos/external_camera_raw_frames/`，图 3 在
`debug/videos/room_interaction_frames/`，节点图在
`debug/videos/semantic_topology_frames/`，按 sim step 对齐后的完整 6-panel PNG 在
`videos/offline_composite_frames/`。

## 8.6 Room分割测试

python /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/room_segmentation_debug_tool.py live \
  --output-dir /home/user/ldl/molmospaces/scripts/InteractiveNav/output/room_occ_snaps
这个 live 模式会订阅 /struct_mapping/occ_map，默认只有当和上一次保存相比，变化栅格数超过 500 或变化比例超过 0.02 才保存一次。保存格式是 .npz 快照。

离线回放模式：
python /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/room_segmentation_debug_tool.py replay \
  --input /home/user/ldl/molmospaces/scripts/InteractiveNav/output/room_occ_snaps \
  --output-dir /home/user/ldl/molmospaces/scripts/InteractiveNav/output/room_occ_overlays
这个 replay 模式会读取单个 .npz 或整个目录，调用 import 进来的 RoomSegmenter 做分割，然后输出：
每张占据图的 room 叠加图 *_room_overlay.png
一个汇总 summary.json


---

## 9. 常见测试组合

## 9.1 最小基础检查

适用场景：

- 检查环境是否能跑
- 不进入 ROS 联调

建议顺序：

1. `nav_to_obj` 最小测试
2. 场景路径与资产目录检查

## 9.2 ROS 联调检查

适用场景：

- 检查导航系统能否正常启动
- 检查 RViz、map、explore、nav 侧链路

建议顺序：

1. 启动 `molmospaces_nav_system.launch`
2. 打开 RViz
3. 必要时手动控制
4. 必要时统一 reset

## 9.3 模块化语义图检查

适用场景：

- 检查 detector-only / GT replay 到 unified graph 的路径

建议顺序：

1. 启动 `semantic_mapping_debug.launch`
2. 运行 `object_detection_visual_test.launch` 或 GT replay
3. 在 RViz / rostopic 中检查输出

## 9.4 小规模验证集检查

适用场景：

- 检查第一阶段闭环是否真实成立
- 在进入大规模 benchmark 之前做稳定 sanity check

建议验证内容：

1. 选择少量场景，确保存在明确 `door-blocked path`
2. 分别运行：
   - `pure nav`
   - `nav + oracle open`
   - ROS 模块化链路或 interaction-aware 规划链路
3. 记录：
   - 是否需要交互
   - 交互前后连通性是否变化
   - 是否成功到达目标
   - 路径是否明显缩短或从不可达到可达

建议优先观察：

- 门关闭时纯导航是否失败或绕行
- oracle 开门后是否立即恢复可达性
- 图/提示是否给出正确的交互信号
- 交互后是否触发继续导航而不是停在中间状态

## 9.5 Python 探索与多场景检查

单场景探索：

```bash
conda activate mlspaces
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch \
  start_explore_py:=true \
  exploration_only:=true \
  house_ind:=4
```

同一 simulator 进程顺序加载多个场景：

```bash
roslaunch nav_pkg molmospaces_nav_system.launch \
  start_explore_py:=true \
  exploration_only:=true \
  house_inds:="4,7,10"
```

当前 ROS 探索仿真默认使用 `policy_dt_ms=200`、`ctrl_dt_ms=10`、
`sim_dt_ms=10`，`move_base` controller 为 `5Hz`。场景切换时会取消目标、
清理 costmap，并重置 gmapping 与 `explore_py` 状态。

---

## 10. 参数说明与备注

## 10.1 常用参数

- `--scene_dataset procthor-10k`：指定场景数据集
- `--data_split train`：指定 split
- `--house_ind 1`：指定场景编号
- `--target_types Apple`：指定导航目标类别
- `--task_horizon 3000`：指定任务最长步数
- `manual_control:=true`：ROS 导航系统进入手动控制模式
- `backend:=yoloe_pf_box3d`：检测后端使用 YOLOE prompt-free + box3d
- `provider:=yoloe_local`：本地 YOLOE provider

## 10.2 当前文档边界

本文件当前主要维护：

- 已确认可用的命令
- 常见调试流程
- 常用路径与参数

本文件当前不负责：

- 项目立意与研究主张
- 子任务拆解与阶段判断
- 历史讨论内容归档

这些内容分别由 `readme_pi.md`、`TODO.md` 与 `AGENTS.md` 维护。

---

## 11. MLLM 模块化交互导航

三模块消融配置位于：

```text
scripts/InteractiveNav/configs/semantic_decision/ablations/
```

模式定义：

- 模块一：`static_semantic` / `dynamic_rule` / `dynamic_mllm`
- 模块二：`rule_cost` / `mllm_score`
- 模块三：`direct_atomic` / `rule_verified` / `mllm_skill_verified`

House 7 规则基线示例：

```bash
roslaunch nav_pkg semantic_interactive_ablation.launch \
  house_ind:=7 \
  task_horizon:=1000 \
  ablation_config:=$(rospack find semantic_decision_py_pkg)/../../../scripts/InteractiveNav/configs/semantic_decision/ablations/current_rule_baseline.yaml
```

完整 MLLM 示例需要同时启动属性推理：

```bash
roslaunch nav_pkg semantic_interactive_ablation.launch \
  house_ind:=7 \
  task_horizon:=1000 \
  enable_attribute_inference:=true \
  model_name:=<MODEL_NAME> \
  ablation_config:=<REPO>/scripts/InteractiveNav/configs/semantic_decision/ablations/full_mllm.yaml
```

题库绑定 House 场景图像：

```bash
python scripts/InteractiveNav/collect_mllm_benchmark_samples.py \
  --question-bank scripts/InteractiveNav/mllm_benchmark/question_bank.json \
  --source-dir <IMAGE_DIR> \
  --output-dir <BOUND_QUESTION_DIR> \
  --bind attribute_closed_fridge=<CLOSED_FRIDGE_IMAGE> \
  --bind attribute_open_portal=<OPEN_PORTAL_IMAGE> \
  --bind verify_fridge_opened=<OPEN_FRIDGE_IMAGE> \
  --bbox attribute_closed_fridge=<X0,Y0,X1,Y1> \
  --bbox attribute_open_portal=<X0,Y0,X1,Y1> \
  --bbox verify_fridge_opened=<X0,Y0,X1,Y1>
```

运行指定四模型评测：

```bash
PYTHONPATH="Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts" \
python scripts/InteractiveNav/evaluate_mllm_question_bank.py \
  --env-file .env \
  --question-bank <BOUND_QUESTION_DIR>/question_bank_bound.json \
  --models gpt-5.3-codex-spark qwen3.6-flash qwen3.5-35b-a3b deepseek-v4-flash \
  --timeout-s 15 \
  --reasoning-effort low \
  --image-detail low \
  --crop-margin-ratio 0.10 \
  --crop-max-side-px 512 \
  --output <OUTPUT_DIR>/model_comparison.json
```

默认按角色限制输出预算：属性 `384`、Subgoal `128`、技能规划 `192`、视觉验证 `192`。
输出包含整体及各角色的准确率、有效响应率、逐题耗时、token、reasoning token、可见输出 TPS，并同时生成 CSV。
视觉属性和交互反馈只输入目标 `2D bbox` 裁切；交互反馈只使用交互后的单张目标图。

### 模块 2：历史决策离线 replay

无需启动 ROS 或仿真，可将已保存的 `mllm_metrics.jsonl` 与
`debug/raw/step_boundaries.jsonl.gz` 连接成只含公开感知信息的 M2 测试集：

```bash
DATASET=/home/ldl/outputs/interactive-nav/m2-replay/<RUN_NAME>
TMPDIR=/home/ldl/tmp/m2-replay-extract \
XDG_CACHE_HOME=/home/ldl/.cache/m2-replay-extract \
/home/ldl/conda_envs/mlspaces/bin/python \
  scripts/InteractiveNav/evaluation/m2_replay_eval.py extract \
  --input-root /home/ldl/outputs/interactive-nav/<EVAL_RUN>/evaluation \
  --output-dir "$DATASET"
```

输出包括 `cases.jsonl`、带 SHA256 的 `manifest.json` 和
`annotations.template.jsonl`。旧录像未保存 policy 内部的精确
`recent_decisions`，这类 case 会标记为 `reconstructed_public_context` 及缺失字段；
不得将其描述为逐字节请求复现。模型输入经过递归隐私审计，不包含
`target_instance_id`、oracle plan、required interaction 或 recorder 的
`gt_observations`。

仅测试 M2 并与原模型输出或人工标注对比：

```bash
OUT=/home/ldl/outputs/interactive-nav/m2-replay-results/<MODEL_NAME>
TMPDIR=/home/ldl/tmp/m2-replay-run \
XDG_CACHE_HOME=/home/ldl/.cache/m2-replay-run \
/home/ldl/conda_envs/mlspaces/bin/python \
  scripts/InteractiveNav/evaluation/m2_replay_eval.py replay \
  --dataset "$DATASET" \
  --output-dir "$OUT" \
  --endpoint http://127.0.0.1:8000/v1 \
  --model <MODEL_NAME> \
  --timeout-s 30 \
  --max-retries 1 \
  --annotations "$DATASET/annotations.template.jsonl"
```

`recorded_top1_agreement` 只表示相对历史 policy 的行为一致性，不是准确率。
填写 `acceptable_top1_ids`、`preferred_ranking` 或 `forbidden_ids` 后，才会生成
`acceptable_top1_accuracy`、`preferred_mrr` 和 `forbidden_top1_rate` 等正确性指标。
可通过 `--prompt-file` 比较提示词，通过 `--case-id`、`--limit` 运行小切片。

若要只替换 M2 文本模型，先确认目标服务的 `/v1/models` 中确实暴露该模型，
再在本地未提交的 dotenv 中设置（不会改变 M1/M3）：

```bash
SEMANTIC_M2_ENDPOINT=https://<strong-model-host>/v1
SEMANTIC_M2_MODEL_NAME=<served-model-id>
SEMANTIC_M2_API_KEY_ENV=SEMANTIC_M2_API_KEY
SEMANTIC_M2_TIMEOUT_S=30
SEMANTIC_M2_TIMEOUT_RETRY_COUNT=1
SEMANTIC_M2_TIMEOUT_RETRY_BACKOFF_S=1
```

当前开发机可用的本地服务/权重仍是 `qwen3.6-35b-a3b-fp8`；没有第二个更强的本地
文本模型可供切换，因此已有 high-reasoning replay 只证明同一权重的协议/行为稳定性，
不能写成“更强模型提升”。接入远端或新权重后，用同一 `cases.jsonl` 和人工标注再比较。

#### M2 并发镜像与同提示词模型对比（2026-09-22）

本轮使用 166 条公开历史快照。以 episode 分组冻结 dev/holdout 后，由独立
subagent 盲审规则生成标签；标签不读取历史回答或本轮预测。主指标只计能够
区分优劣的多候选请求：dev 30、holdout 72；46 条单候选另报，18 条歧义/证据
冲突不计主分数。该分数是公开策略规则接受率，不是人工真值准确率或导航 SR。
规则与数据集见 `/home/ldl/outputs/interactive-nav/m2-mirror-20260922/labels/`。

冻结的候选提示词为
`scripts/InteractiveNav/configs/semantic_decision/prompts/m2_rank_public_evidence_v2.txt`。
它仅用于显式指定的 replay。开发集选择后，原/优化提示词分别在完整 166 条上
运行；独立 holdout 为 51/72 → 53/72，尚不足以确认闭环收益。同一优化提示词下，
Qwen / GPT / Gemini 的 holdout 分别为 53/72、58/72、60/72；Gemini 的 4 条
开门后穿越阶段样本仅通过 2 条，需与整体探索排序分数一起看。GPT 有 1 条
服务过载失败，保留在评分分母；Gemini 恢复后正式请求 166/166 有效。
完整结果、逐条输出和复算脚本位于
`/home/ldl/outputs/interactive-nav/m2-mirror-20260922/report.md`。

本地高并发测试（需先有健康的 Qwen 服务）：

```bash
TMPDIR=/home/ldl/tmp/m2-mirror-20260922 \
XDG_CACHE_HOME=/home/ldl/.cache/m2-mirror-20260922 \
/home/ldl/conda_envs/mlspaces/bin/python \
  scripts/InteractiveNav/evaluation/m2_replay_eval.py replay \
  --dataset /home/ldl/outputs/interactive-nav/m2-replay/custom-task-t-20260917012600-dw5wv \
  --annotations /home/ldl/outputs/interactive-nav/m2-mirror-20260922/labels/annotations.jsonl \
  --prompt-file scripts/InteractiveNav/configs/semantic_decision/prompts/m2_rank_public_evidence_v2.txt \
  --endpoint http://127.0.0.1:8000/v1 --model qwen3.6-35b-a3b-fp8 \
  --concurrency 32 --timeout-s 90 --max-retries 1 --max-tokens 1536 --reasoning-effort off \
  --output-dir /home/ldl/outputs/interactive-nav/m2-mirror-local-new-run
```

远端配置文件使用 `url`、`key`、`model1`、`model2` 字段，凭据只在进程内读取。
以下命令对两个模型合计限流：8 并发、任意滚动 60 秒最多 28 次请求，重试也
占用配额。若服务配额按模型独立计算，可分别使用 `--model-key model1` 和
`--model-key model2` 启动独立进程，各自设为 8 并发、26 RPM；本轮用户后续
确认按模型独立配额执行。先用 `--limit 1` 和独立输出目录做协议预检。

```bash
TMPDIR=/home/ldl/tmp/m2-mirror-20260922 \
XDG_CACHE_HOME=/home/ldl/.cache/m2-mirror-20260922 \
/home/ldl/conda_envs/mlspaces/bin/python \
  scripts/InteractiveNav/evaluation/m2_mirror_compare.py \
  --env-file /home/ldl/.env \
  --dataset /home/ldl/outputs/interactive-nav/m2-replay/custom-task-t-20260917012600-dw5wv \
  --annotations /home/ldl/outputs/interactive-nav/m2-mirror-20260922/labels/annotations.jsonl \
  --prompt-file scripts/InteractiveNav/configs/semantic_decision/prompts/m2_rank_public_evidence_v2.txt \
  --concurrency 8 --requests-per-minute 28 --timeout-s 90 --max-retries 1 \
  --output-dir /home/ldl/outputs/interactive-nav/m2-mirror-remote-new-run
```

若本机继承了 SOCKS 代理但环境无 `socksio`，且该服务可直连，可仅在本次命令前
加 `env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy`。
本轮 Gemini 早期两次返回 `User location is not supported for the API use`，
用户恢复服务后预检通过，再执行正式 166-case 测试。预检错误只记录为服务故障，
不混入恢复后的正式模型成绩。

#### M2 公共事实上下文与 Qwen 消融（2026-09-22）

在线 M2 保持提示词不变，新增初始/当前位置、实际房间访问序列、最近30次决策、
房间/门/容器几何和房间前沿统计；房间关键物体只保留名称，不发送可见性标记、
规则房间评分或房间—目标匹配结论。默认候选池为最多12个去重前沿加全部有效交互/
导航候选；`model_policy.candidate_pool_mode=legacy` 保留旧裁剪策略作消融。
`include_pre_scores` 默认false；后置执行守卫并未关闭，离线主指标只评原始模型选择。
在线机器人坐标按位姿时间戳通过TF转换到图坐标系，转换失败标为未知，不假设坐标系
一致；不改变导航执行位姿。房间前沿统计按实际提取时间检查新鲜度，旧状态重发不会
刷新有效期，缺失/过期统计不伪装成零。

历史重建禁止使用请求之后的图、反馈或完整场景GT。旧日志中请求开始时间仍是估计值，
图版本精确对齐不等于原HTTP请求完整恢复；图版本较旧、历史不完整、前沿长度下界均
须标记。最近30条是上限，不能将实际只有数条的记录凑满。新在线请求直接保存
`public_request` 和 `request_started_ts`，后续不必再依赖异步快照拼接。

以下命令从仓库根目录运行；新运行请选择新的输出目录，冻结输入/标签禁止覆盖。

```bash
export TMPDIR=/home/ldl/tmp/m2-context-20260922
export XDG_CACHE_HOME=/home/ldl/.cache/m2-context-20260922
export HF_HOME=/home/ldl/.cache/m2-context-20260922/hf
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

/home/ldl/conda_envs/mlspaces/bin/python scripts/InteractiveNav/evaluation/m2_context_dataset.py \
  --cases /home/ldl/outputs/interactive-nav/m2-replay/custom-task-t-20260917012600-dw5wv/cases.jsonl \
  --log-root /home/ldl/outputs/interactive-nav/custom-task-t-20260917012600-dw5wv/evaluation \
  --output-dir /home/ldl/outputs/interactive-nav/m2-context-new/dataset

/home/ldl/qwen36-fp8/venv/bin/python scripts/InteractiveNav/evaluation/m2_context_ablation.py prepare \
  --records /home/ldl/outputs/interactive-nav/m2-context-new/dataset/enriched_records.jsonl \
  --prompt-file scripts/InteractiveNav/configs/semantic_decision/prompts/m2_rank_public_evidence_v2.txt \
  --tokenizer /home/ldl/qwen36-fp8/model/Qwen3.6-35B-A3B-FP8 \
  --max-model-len 16384 --max-tokens 1536 \
  --output-dir /home/ldl/outputs/interactive-nav/m2-context-new/inputs

/home/ldl/conda_envs/mlspaces/bin/python scripts/InteractiveNav/evaluation/m2_context_labels.py \
  --inputs /home/ldl/outputs/interactive-nav/m2-context-new/inputs/label_inputs.jsonl \
  --split /home/ldl/outputs/interactive-nav/m2-mirror-20260922/labels/split.json \
  --output-dir /home/ldl/outputs/interactive-nav/m2-context-new/labels

/home/ldl/conda_envs/mlspaces/bin/python scripts/InteractiveNav/evaluation/m2_context_ablation.py run \
  --inputs /home/ldl/outputs/interactive-nav/m2-context-new/inputs \
  --annotations /home/ldl/outputs/interactive-nav/m2-context-new/labels/annotations.jsonl \
  --concurrency 32 --timeout-s 120 --max-tokens 1536 --max-retries 1 \
  --output-dir /home/ldl/outputs/interactive-nav/m2-context-new/run

/home/ldl/conda_envs/mlspaces/bin/python scripts/InteractiveNav/evaluation/m2_context_report.py \
  --inputs /home/ldl/outputs/interactive-nav/m2-context-new/inputs \
  --run /home/ldl/outputs/interactive-nav/m2-context-new/run \
  --annotations /home/ldl/outputs/interactive-nav/m2-context-new/labels/annotations.jsonl \
  --guard-shadow \
  --output-dir /home/ldl/outputs/interactive-nav/m2-context-new/report
```

运行前确认Qwen服务的实际上下文窗口与 `--max-model-len` 一致；prepare会对所有消融
共同检查token预算，不静默截断。`compact_control` 是同一新候选池的简化事实对照，
不是旧HTTP请求；`legacy_candidate_pool` 是旧池策略的重算，不等同历史已发送候选。
去近期决策/轨迹消融仍保留候选历史摘要/房间到访状态，分别测事件明细与轨迹顺序的
额外贡献。标签为预测盲的公开策略proxy，禁止将其接受率称为导航成功率。
主报告使用固定的非单候选、可评分样本分母，另列单候选、歧义样本、严格/滞后图版本、
阶段合同和schema错误；不要把runner包含单候选的汇总接受率作为主指标。各条件只采样
一次，微小差异不构成稳定收益证据，阶段合同与历史成功记录冲突需在线闭环验证。

远端模型使用相同冻结输入时，先导出完整上下文这一组；导出只给标签ID加与请求一致的
命名空间，不重建候选、不改评分规则。以下命令只读取既有数据，导出目录必须不存在：

```bash
/home/ldl/conda_envs/mlspaces/bin/python scripts/InteractiveNav/evaluation/m2_context_ablation.py export-arm \
  --inputs /home/ldl/outputs/interactive-nav/m2-context-new/inputs \
  --annotations /home/ldl/outputs/interactive-nav/m2-context-new/labels/annotations.jsonl \
  --arm full_context \
  --output-dir /home/ldl/outputs/interactive-nav/m2-context-new/remote_inputs
```

启动前在选用的Python中检查 `httpx` 和代理所需的 `socksio` 导入。当前基础环境
`/home/ldl/miniconda3/bin/python` 已有这两个依赖；此处只读取冻结请求，不用它重新
计算候选预评分。凭据在runner进程内读取，不要打印 `.env` 或把URL/key放进命令行。
例如GPT保守运行如下，Gemini另用 `--model-key model2 --concurrency 2` 和独立新输出目录：

```bash
/home/ldl/miniconda3/bin/python -m scripts.InteractiveNav.evaluation.m2_mirror_compare \
  --env-file /home/ldl/.env \
  --dataset /home/ldl/outputs/interactive-nav/m2-context-new/remote_inputs/cases.jsonl \
  --annotations /home/ldl/outputs/interactive-nav/m2-context-new/remote_inputs/annotations.jsonl \
  --prompt-file /home/ldl/outputs/interactive-nav/m2-context-new/remote_inputs/prompt.txt \
  --model-key model1 --concurrency 2 --requests-per-minute 20 \
  --timeout-s 120 --max-retries 1 --max-tokens 1536 --reasoning-effort off \
  --output-dir /home/ldl/outputs/interactive-nav/m2-context-new/gpt
```

每个模型只启动一个pool，Gemini不要叠加多个2并发进程；timeout重试也计入滑动60秒
请求额度。保留客户端启动失败诊断，与真实网络请求分开审计；不得用事后重跑替换
正式模型的错误回答。新旧Qwen耗时比较须同时报告服务并发负载、窗口和限流等待。
各模型分别未超限并不保证共享网关配额未超限；也可能受token吞吐或服务端并发限制。
若要同进程共享额度，可以省略 `--model-key`，使用 `--concurrency 4` 给两个模型各2个
worker，并共用 `--requests-per-minute 20`。遇到429应保留失败并降低压力，不能仅凭
客户端未超过28RPM就认定服务错误；本runner只自动重试timeout，不自动重试429。

所有模型完整结束后，用同一份冻结标签比较原始回答；下列入口拒绝不完整结果，自动
从Qwen九组输出中只取full_context，保留模型/服务失败在固定主分母中：

```bash
/home/ldl/conda_envs/mlspaces/bin/python -m scripts.InteractiveNav.evaluation.m2_context_model_report \
  --inputs /home/ldl/outputs/interactive-nav/m2-context-new/remote_inputs \
  --run qwen=/home/ldl/outputs/interactive-nav/m2-context-new/run/predictions.jsonl \
  --run gpt=/home/ldl/outputs/interactive-nav/m2-context-new/gpt/model1/predictions.jsonl \
  --run gemini=/home/ldl/outputs/interactive-nav/m2-context-new/gemini/model2/predictions.jsonl \
  --output-dir /home/ldl/outputs/interactive-nav/m2-context-new/model_report
```

#### M2 语义提示词与建议摘要的成对测试（2026-09-22）

入口 `scripts/InteractiveNav/evaluation/m2_semantic_experiment.py` 只做离线镜像，
不修改在线默认提示词或候选。11 组为 B、新语义链 P1、起点近房 P2、探索历史 P3、
相关容器优先 P4、承载物 P5、融合 P6，以及相对 P6 的阶段摘要、区域摘要、备用动作
多样性与三者组合。每组重复两次、随机交错，共用一个 32 并发池，不给各组分别开池。

先冻结预测盲的新检查和独立人工探针，再冻结全部请求。下列新输出路径不得已存在：

```bash
export TMPDIR=/home/ldl/tmp/m2-semantic-new
export XDG_CACHE_HOME=/home/ldl/.cache/m2-semantic-new
export HF_HOME=/home/ldl/.cache/m2-semantic-new/hf
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

/home/ldl/conda_envs/mlspaces/bin/python -m scripts.InteractiveNav.evaluation.m2_semantic_checks \
  --inputs /home/ldl/outputs/interactive-nav/m2-context-remote-20260922/inputs \
  --output-dir /home/ldl/outputs/interactive-nav/m2-semantic-new/checks

/home/ldl/qwen36-fp8/venv/bin/python -m scripts.InteractiveNav.evaluation.m2_semantic_experiment prepare \
  --inputs /home/ldl/outputs/interactive-nav/m2-context-remote-20260922/inputs \
  --records /home/ldl/outputs/interactive-nav/m2-context-20260922/dataset/enriched_records.jsonl \
  --checks /home/ldl/outputs/interactive-nav/m2-semantic-new/checks \
  --prompt-dir scripts/InteractiveNav/configs/semantic_decision/prompts/m2_semantic_20260922 \
  --tokenizer /home/ldl/qwen36-fp8/model/Qwen3.6-35B-A3B-FP8 \
  --output-dir /home/ldl/outputs/interactive-nav/m2-semantic-new/inputs \
  --repeats 2 --seed 20260922 --max-model-len 16384 --max-tokens 1536

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  NO_PROXY=127.0.0.1,localhost \
  /home/ldl/conda_envs/mlspaces/bin/python -u -m scripts.InteractiveNav.evaluation.m2_semantic_experiment run \
  --inputs /home/ldl/outputs/interactive-nav/m2-semantic-new/inputs \
  --output-dir /home/ldl/outputs/interactive-nav/m2-semantic-new/run \
  --concurrency 32 --timeout-s 120 --max-retries 1 --max-tokens 1536

/home/ldl/conda_envs/mlspaces/bin/python -m scripts.InteractiveNav.evaluation.m2_semantic_report \
  --inputs /home/ldl/outputs/interactive-nav/m2-semantic-new/inputs \
  --run /home/ldl/outputs/interactive-nav/m2-semantic-new/run \
  --output-dir /home/ldl/outputs/interactive-nav/m2-semantic-new/report
```

本次冻结集为历史 166 条和人工探针 18 条，共 4,048 次请求；历史主评分仍是原来的
130 条（dev 38、历史 holdout 92），两次合并分母为 260，但不是 260 个独立场景。
人工探针、新策略相对排序检查、旧 Top1 接受率分别报告，均不等同闭环导航 SR。
旧 holdout 已被查看，不能再称新盲测集。阶段冲突只作诊断，不把历史执行成功当成
当前物理穿越完成。历史区域摘要缺少同坐标系候选关联，不能证明空间去重收益；
承载物和受控起点距离在历史集没有适用检查，只能用人工探针检查偏好遵循。

准备过程会核验提示词、标签及检查文件哈希，并对实际 wire 文本检查完整 token
预算；任何超限均在调用模型前失败，不截断或剔除样本。建议摘要 v2 通过引用原始
历史去重，不删原始上下文。运行保留所有服务/格式错误，只按预定策略重试 timeout。
阶段/区域组同时改变摘要和指令，不能解释成纯提示词单因素实验。

### 模块 3：历史图像视觉操作规划

使用运行时同一提示词与输出 schema，测试门操作方式和多抽屉中心点：

```bash
conda run -n mlspaces python scripts/InteractiveNav/evaluate_module3_visual_planning.py \
  --manifest scripts/InteractiveNav/configs/semantic_decision/module3_historical_eval.json \
  --env-file .env \
  --output outputs/module3_visual_planning_eval.json
```

无可用视觉模型时可用 `--mode mock --model mock-module3` 检查裁图、协议、schema 和评分链路；mock 结果不能作为视觉能力结论。

---

## 12. Habitat ObjectNav-v2 外部 Module-2 评测

适用场景：在不修改 `Interactive-Nav-SG-nav`、不生成交互动作的前提下，
将原始 Module-2 `ModelPolicyClient` 接入官方 Habitat Challenge 2023
ObjectNav-v2 / HM3D-Sem v0.2 验证环境。运行时 policy 只使用公开的
RGB-D、GPS、Compass 和 ObjectGoal；只向 Module-2 提供 `EXPLORE` / `NAVIGATE`
候选，并且只输出连续 `velocity_control` / `velocity_stop`。

先确认共享 HM3D-Sem 资产已获本项目授权，再运行固定的 10 场景、每场 1
episode 验证切片：

```bash
env EGL_PLATFORM=surfaceless \
  TMPDIR=/home/ldl/tmp/habitat-objectnav-eval \
  XDG_CACHE_HOME=/home/ldl/.cache/habitat-objectnav-eval \
  PYTHONPYCACHEPREFIX=/home/ldl/.cache/habitat-objectnav-eval/pycache \
  PYTHONPATH=/home/ldl/molmospaces-exp-setting/scripts/InteractiveNav:/home/ldl/habitat-objectnav/src/habitat-lab/habitat-lab \
  /home/ldl/conda_envs/habitat-challenge-2023/bin/python \
  /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/habitat_v2_adapter/evaluate.py \
  --output-dir /home/ldl/outputs/habitat_objectnav_v2_m2/current_m2_public_ten \
  --scene-count 10 --episodes-per-scene 1 --max-steps 1000 \
  --max-episode-seconds 500 --gpu-id 0
```

结果查看：

```text
/home/ldl/outputs/habitat_objectnav_v2_m2/current_m2_public_ten/<run>/summary.json
/home/ldl/outputs/habitat_objectnav_v2_m2/current_m2_public_ten/<run>/episodes.jsonl
```

2026-08-17 的可复现实测位于
`current_m2_public_ten/run-20260817T085119Z`：10/10 episode 完成、159 次
模型实际选中、1,104/1,104 次 MLLM 请求成功，但 SR=0、SPL=0。该数值证明
外部接口和官方 v2 评测链路可运行，不可表述为导航性能提升；后续模型改动应
先在相同 manifest 上重新报告官方 `success` / `spl`。

### 2026-09-14 位置锁定与交互恢复回归

本轮到位契约：当前接近位首次进入XY容差后锁定位置完成，后续只判断yaw；
同位姿停车重发保留锁定，新目标或新阶段重新建立锁定。碰撞与物理交互面检查仍保留。
Noetic实际读取 `/move_base/latch_xy_goal_tolerance`，不能仅配置DWA插件子命名空间。
逐步执行状态的 `navigation_arrival` 记录锁定步数、位置和 `rotation_only` 模式。

- 动态占据先停车，最多复核3个任务步；短暂恢复后同点重试有次数上限，持续阻塞才换位。
- 同面yaw偏差允许一次公开TF/M1正面信息驱动的原位修正，不能排除整面。
- 物理交互预算为120步，step停止180秒后触发停流兜底；缺少step时才使用墙钟超时。
- 导航不可达与M1失败分开反馈；导航冷却20步，持续60步无可执行候选则明确报告受阻。

完整决策/执行/交互桥回归使用Conda ROS模块：554 passed；感知、房间分割和离线视频补充回归186 passed。
端到端结果独立验收，不能由静态测试推断性能提升。

`nav.launch` 在加载override后显式设置委托 `GlobalPlanner/orientation_mode=0`，
由 `OrientedGlobalPlanner` 负责路径切线与末端yaw。Noetic模式3在两点短路径上存在负索引，
仅改YAML不足以保证实际启动值；启动前用 `roslaunch --dump-params` 验证，启动后查询参数服务器。
同面原位yaw恢复使用独占step控制，不调用 `make_plan`、移动走廊或平移目标。

```bash
env TMPDIR=/home/ldl/.cache/interactive-nav-latch-500 \
  XDG_CACHE_HOME=/home/ldl/.cache HF_HOME=/home/ldl/.cache/huggingface \
  TORCH_HOME=/home/ldl/.cache/torch \
  MPLCONFIGDIR=/home/ldl/.cache/interactive-nav-latch-500/matplotlib \
  CONDA_SH=/home/ldl/miniconda3/etc/profile.d/conda.sh \
  CONDA_ENV=/home/ldl/conda_envs/mlspaces \
  PYTHON_BIN=/home/ldl/conda_envs/mlspaces/bin/python \
  ROS_SETUP=/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/devel/setup.bash \
  MAPPING_SCAN_SOURCE=organized_depth POINTCLOUD_STRIDE=1 CLEAN_INTERMEDIATE=false \
  /home/ldl/conda_envs/mlspaces/bin/python \
  scripts/InteractiveNav/run_semantic_interaction_exploration_batch.py \
  --output-dir /home/ldl/outputs/interactive-nav/batch_latched_v4_h1_h10_500_20260914_014500 \
  --house-inds 1 2 3 4 5 6 7 8 9 10 --workers 10 --base-master-port 15801 \
  --task-horizon 500 --method full_mllm_exploration --scene-timeout-s 3600 \
  --semantic-model-env-file /home/ldl/molmospaces-exp-setting/.env
```

重跑必须使用新输出目录及未占用端口，不覆盖既有数据。完成后核对 `summary.json`、
`force_interaction_events.json`、`debug/raw/step_boundaries.jsonl` 和 `offline_video_summary.json`。

### 2026-09-14 门净空与切向候选回归

- 范围：portal完整到位区域净空、同面小切向候选及正面角预算、按净空选择可达候选。
- candidate.portal_tangent_offsets_m默认[0.10,-0.10,0.15,-0.15]；空列表关闭切向选点。
- 验证：完整决策执行与交互桥574 passed。H2原始costmap离线筛选13个候选、5个通过，
  原点净空不足被拒绝；优选点净空0.591m、中心cost0。未执行新500step仿真。
- 第3项semantic_subgoal_no_progress重试缺口本次保持未改，不能将离线选点通过解释为整场问题已解决。

```bash
env TMPDIR=/home/ldl/.cache XDG_CACHE_HOME=/home/ldl/.cache \
  PYTHONPATH="$PWD/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts:$PWD/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts:$PWD/Interactive-Nav-SG-nav/src/semantic_mllm_py_pkg/scripts:$PWD/Interactive-Nav-SG-nav/src/explore_py_pkg/scripts:$PWD:/home/ldl/conda_envs/ros-noetic/lib/python3.11/site-packages" \
  /home/ldl/conda_envs/mlspaces/bin/python -m pytest \
  Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/tests \
  mlspaces_tests/test_force_interaction_bridge.py -q
```

离线加载真实ROS执行器和OpenCV时使用`/home/ldl/conda_envs/ros-noetic/bin/python`，
不要在MolmoSpaces解释器中优先导入ROS环境的OpenCV二进制，两套OpenSSL版本并不兼容。

### 2026-09-14 最小导航修复回归

- 探索/穿门位置容差保持 0.25 m，不增加 0.10/0.05 m 降级档位；抽屉净空仍为 0.50/0.85/1.00 m。
- 净空不合格的探索点在原点 0.8 m 内寻找替代位置，仍需满足完整到位区域净空、同一 frontier 可见性和全局膨胀地图连通性；穿门替代点必须留在门的目标侧。找不到时保留 deferred 诊断，不标记为已探索。
- 全局 costmap 的完整图和增量更新共同维护四连通可达域；candidate 和 executor 都可拒绝新鲜地图中不连通的目标。过期地图不作为确定的不可达证据，执行器仍保留 make_plan 验证。
- 临时障碍需在不同地图回执和不同 step 上连续确认 3 次；恢复后只重发原锚点，保持原导航尝试计数，不重排全部锚点。20 step 内无法确认时返回可重试的 costmap 未更新，不记为 M1 失败。
- AABB 选面后的独立物理正面复核放宽到 25 度；公开选面 15 度、导航到位距离/朝向、错误面拒绝均不变，不向规划器输出 GT 法向或纠偏角。
- 仍有 frontier 但无可执行候选时，第 20/60 step 最多各尝试一次已有扫描动作。旋转需通过当前局部地图安全检查；无候选总预算为 120 step，最终报告 stalled 而非探索完成。

无仿真回归：623 passed；`git diff --check` 通过。测试入口：

```bash
env TMPDIR=/home/ldl/.cache XDG_CACHE_HOME=/home/ldl/.cache PYTHONDONTWRITEBYTECODE=1 \
  /home/ldl/conda_envs/mlspaces/bin/python -c '
import sys, cv2
from pathlib import Path
root = Path("/home/ldl/molmospaces-exp-setting")
sys.path[:0] = [str(root), *[str(root/"Interactive-Nav-SG-nav/src"/p/"scripts") for p in ("semantic_decision_py_pkg", "semantic_mapping_py_pkg", "semantic_mllm_py_pkg", "explore_py_pkg")]]
sys.path.append("/home/ldl/conda_envs/ros-noetic/lib/python3.11/site-packages")
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
' -q -p no:cacheprovider \
  Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/tests \
  mlspaces_tests/test_force_interaction_bridge.py
```

该入口先加载 MolmoSpaces 的 OpenCV，再将 ROS 库追加到搜索路径，避免 OpenCV 二进制混用。
原始批次 `batch_clearance100_h1_h10_500_20260914_112848` 的离线地图回放结果在
`outputs/analysis_clearance100_500_20260914/minimal_fix_replay.json`：H4 step 100/216 各恢复 2 个安全点；
H10 step 307 恢复 1 个、step 445 恢复 2 个；H3 step 33/57 的 7 个交互锚点均不连通，
0.8 m 搜索范围内仍无安全探索替代点。H9 step 250 的原位姿通过新角度复核单测。
这些结果仅验证控制逻辑和已记录几何，不代表新一轮仿真或覆盖率验收通过。
# Room 分割离线诊断回放

已有录制目录包含 `debug/raw/map_manifest.jsonl` 与
`debug/raw/step_boundaries.jsonl` 时，可不启动 ROS、仿真或模型服务：

```bash
TMPDIR=/home/ldl/tmp XDG_CACHE_HOME=/home/ldl/.cache \
MPLCONFIGDIR=/home/ldl/.cache/matplotlib \
/home/ldl/conda_envs/mlspaces/bin/python \
  scripts/InteractiveNav/evaluation/replay_room_segmentation.py \
  /home/ldl/outputs/interactive-nav/benchmark-v16-1000-20260915_055700 \
  --output-dir /home/ldl/outputs/interactive-nav/room-replay
```

输出三场 OCC / 历史 room 标签 / 离线回放拼图、逐帧 room ID 和最终栅格。
回放使用当前 `semantic_map` 配置、历史公开门观测和已确认的 graph overlay，
按记录的 step 顺序处理；不复现 ROS 的完整异步回调顺序，也不能替代闭环导航验收。

## 目标等价离线重评分（分层协议 v2）

对已完成的批量评估使用独立的分层口径，不覆盖原始严格结果。每个 episode 同时报告
`exact_instance_success`、`category_goal_success`、
`interaction_contract_goal_success` 和 `interactive_episode_success`：

- 同容器内、同类别且位于冻结容器体积内的实例可以计 Category；只有数据集声明的
  `identity_contract`，或经过审核的 candidate-specific interaction plan，才可以再计
  Contract/Interactive。
- 同房间、同类别且支持关系兼容的 0.30 m 内近邻（例如约 0.275 m 的 CD）只计
  Category，不把距离当作交互契约证据。
- 同房间同类目标（包括原目标在容器、候选在台面的情况）可计 Category；共享同一扇门
  仍需门拓扑审计，且共享通道不能证明完整 container/full interaction contract。
- 旧 benchmark 没有 identity contract 时安全降级为 Category-only；不要把事后类别
  成功写回原始 `success`。
不同房间或已知不同容器的目标不自动放宽；若要把门/容器交互也视为等价，仍需候选级
交互计划证据。

```bash
EVAL_RUN=/home/ldl/outputs/interactive-nav/custom-task-t-20260917034403-twst6
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/ldl/tmp XDG_CACHE_HOME=/home/ldl/.cache \
/home/ldl/conda_envs/mlspaces/bin/python scripts/InteractiveNav/rescore_benchmark_goals.py "$EVAL_RUN/evaluation" \
  --benchmark /home/ldl/molmospaces/scripts/InteractiveNav/output/interactive_nav_v3_procthor10k_val_release_v1_2/benchmark/benchmark.json \
  --scene-dir /home/ldl/molmospaces/assets/scenes/procthor-10k-val \
  --near-radius-m 0.30 \
  --output-dir "$EVAL_RUN/goal_equivalence_v2"
```

输出 `report.md`、`report.json`（逐场改分理由与证据 SHA256）、`rescored_results.json`。
输出目录必须不存在；原始 summary/result、benchmark、交互完成事实不变，不启动仿真。
报告同时展示四层 SR；严格的 `nav_success`、`success`、SPL 和 Total Cost 完全保留，
不会因为类别放宽而去除失败惩罚。`category_spl` 保持为空，因为没有 candidate-specific
参考路径，`spl_original_reference` 也不能当作等价目标最短路后的标准 SPL。

对旧 benchmark 进行 Contract/Interactive 晋级必须提供经过人工审核的审计文件（包含
benchmark SHA256、candidate 名、plan ID 和完整 `required_interaction_ids`），例如：

```bash
python scripts/InteractiveNav/rescore_benchmark_goals.py "$EVAL_RUN/evaluation" \
  --benchmark <BENCHMARK_JSON> --scene-dir <SCENE_DIR> \
  --identity-contract-audit <REVIEWED_AUDIT_JSON> \
  --output-dir "$EVAL_RUN/goal_equivalence_v2_reviewed"
```

最小回归：

```bash
TMPDIR=/home/ldl/tmp XDG_CACHE_HOME=/home/ldl/.cache PYTHONDONTWRITEBYTECODE=1 \
/home/ldl/conda_envs/mlspaces/bin/python -m pytest \
  scripts/InteractiveNav/test_rescore_benchmark_goals.py -q -p no:cacheprovider
```

## 2026-09-17 Perception-only 交互后视觉刷新修正

修正仅在 `scripts/InteractiveNav/ablations/` 的第三组适配器生效：纯观察结束不清空
M1 共识；实际动作后禁止旧状态立即重复执行，必须先完成 observation-only 刷新。
Full、原生 mapper/executor 和评分逻辑未改。51 项相关测试通过：

```bash
TMPDIR=/home/ldl/tmp XDG_CACHE_HOME=/home/ldl/.cache PYTHONDONTWRITEBYTECODE=1 \
/home/ldl/conda_envs/mlspaces/bin/python -c '
import sys, cv2
sys.path.append("/home/ldl/conda_envs/ros-noetic/lib/python3.11/site-packages")
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
' -q -p no:cacheprovider \
  scripts/InteractiveNav/ablations/tests \
  Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/tests/test_portal_state_consensus.py
```

第三组重跑目录：
`/home/ldl/outputs/interactive-nav/ablation-v3-perception-selected10-20260917-202540`。
`comparison.md` 为完整状态/结果，`diagnostic_notes.md` 与 `repair_case_evidence.json`
记录 mixed 15/29 的视觉票与交互对照。启动命令冻结在同名 `.launch.json` 中。
入口 `run_ablation_pool.py --variants no_outcome_update` 只运行第三组；场景选择输入使用
上次保存的 `selected_scene_source.json`。本次动态 max2000、无录制，worker 上限25，
实际10场同时运行。原始 Full 与另外两组不重跑，历史结果保留不覆盖。

### Mixed 场景质量审计与条件子集重评分

在上述目标等价报告基础上，审计所有 mixed 场景（包括成功场景），分别输出
原始全量、原始 eligible、场景定义有效、再排除执行器异常四种分母。
已核实的规则包括初始状态不一致、严格原目标无需容器交互即可完成的必要性反例，
以及抽屉扫描/关闭成功后姿态恢复超时。关闭 ROS 时的异常不计为运行期失败；
私有 GT 看见目标、导航警告或算法耗尽预算，均不能单独作为剔除理由。

```bash
EVAL_RUN=/home/ldl/outputs/interactive-nav/custom-task-t-20260917034403-twst6
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/home/ldl/tmp XDG_CACHE_HOME=/home/ldl/.cache \
python scripts/InteractiveNav/audit_mixed_benchmark_run.py "$EVAL_RUN/evaluation" \
  --goal-rescore-dir "$EVAL_RUN/goal_equivalence_v2" \
  --output-dir "$EVAL_RUN/mixed_quality_audit_v2_verified"
```

输出目录必须不存在。`exclusions.json` 列出本次运行的隔离清单，`report.json`
保留逐场证据与日志行号，`filtered_results.json` 是全量场景中保留下来的计分集合。
不改成功标签，不删除 benchmark，不修复算法，也不重跑仿真。此为事后质量条件子集，
不可当作算法提升或替代全量 benchmark；执行超时场景应修复/重跑后再恢复计分。

```bash
TMPDIR=/home/ldl/tmp XDG_CACHE_HOME=/home/ldl/.cache PYTHONDONTWRITEBYTECODE=1 \
/home/ldl/conda_envs/mlspaces/bin/python -m pytest \
  scripts/InteractiveNav/test_audit_mixed_benchmark_run.py \
  scripts/InteractiveNav/test_rescore_benchmark_goals.py -q -p no:cacheprovider
```
# 2026-09-21：评测失败重试与模型服务归属

## 初始化姿态回归（2026-09-21）

V3 benchmark 回放在环境的初始 500 次稳定化之前恢复 episode 的机器人关节、
底盘姿态及静止控制目标；常规任务的初始化钩子为空。碰撞和稳定化步数保持原值，
稳定化之后仍由原有回放逻辑恢复完整场景状态。

仅初始化测试入口：`scripts/InteractiveNav/profile_scene_initialization.py --episode INDEX --output DIR`。
使用 mlspaces Python、`MUJOCO_GL=egl`，并将 TMPDIR、缓存与输出放在 `/home/ldl`。
正式回归不要传诊断用的 `--settle-steps` 或 `--isolate-robot-during-settle`。
脚本只调用 sample_task/task.reset，不运行导航策略；输出 timings.json、physics.jsonl
和 state_audit.json。状态检查范围为机器人底盘／关节和 episode 指定物体的位置／朝向，
不是完整导航性能回归。

本机实测：house 13 的 episode 3、2006 初始化分别为 9.47、9.43 秒，house 2
episode 0 为 8.68 秒；三场均执行 500 个稳定化物理步且状态审计通过。
结果目录：`/home/ldl/outputs/interactive-nav/init-fixed-h13-e3-r2`、
`init-fixed-h13-e2006-r1`、`init-fixed-h2-e0-r1`（后两者同一父目录）。

统一入口为 `python scripts/InteractiveNav/run_benchmark_eval.py --config <config.json>`。
使用已有模型服务时直接运行；自定义任务需要在同一实例启动模型时增加
`--start-qwen`。它按 `CUDA_VISIBLE_DEVICES`（未设置时按 GPU 枚举）启动每卡一个
Qwen 副本，统一入口 `http://127.0.0.1:8010/v1`。副本只在批次开始时启动，
场景不能重启服务；服务进程退出会中断批次并保留结果，结束时只清理本轮服务。
自定义任务 wrapper 保留平台 EGL、缓存及资产环境准备，并调用同一入口。

默认首轮结束后重试运行不完整场景一次；`--retry-rounds N` 可指定重试轮数。
导航失败但正常完成的场景不重试。每轮 `retry_queue_N.json` 保存剩余场景，
通过批处理的 `--resume` 复用已完成结果并为异常场景新建 attempt，最终汇总保留完整采样集。
重试耗尽仍有缺失／异常结果时入口返回非零。每轮复用首次启动生成的
`runner_snapshot.sh`，启动前执行 `bash -n`，避免共享 shell 在执行中被编辑。
运行期间仍应避免修改 Python 算法和依赖配置；此快照不是整个仓库快照。

历史 episode 2150 的 10800 秒超时：只推进到约 655 步，后段部分相邻观测
间隔约 150 秒，另有约 22 秒的区间。297 条模型请求中 288 条无错误、9 条超时。
超时中断栈位于抽屉物理交互、关节驱动、机器人锁定的 `mujoco.mj_forward`。
因此不能用恒定 1.5 秒/步估算该场景；物理计算与同步延迟各自占比仍需定向测量。
当前不修改物理交互语义或扩大 3 小时超时限制。

### 交互过程计时与锁定回调对照

在统一 eval 入口前设置 `INTERACTIVE_NAV_PROFILE_INTERACTION=1`，即可在每个
attempt 的 `eval/interaction_profiles/<decision_index>/` 保存 `steps.jsonl`
（before_step、观测/task.step、after_step 的起止计时）、`calls.json`
（函数调用数、自身耗时、累计耗时）和 `calls.pstats`。函数累计时间存在嵌套，不能
直接相加。cProfile 只覆盖当前同步交互线程，且引入额外计时开销。

`INTERACTIVE_NAV_COALESCE_ROBOT_LOCK=1` 可选择合并物理子步边界的重复机器人
锁定回调；默认关闭，设置 `0` 恢复原逻辑。保留第一次子步前和每次物理子步后的
锁定及 forward，不修改物理步长、控制增益或抽屉扫描策略。环境变量须在启动 Python
前设置；基线与候选均开启相同计时，用独立输出目录和同一配置比较。

单场测量使用 `--workers 1 --episode-indices 1445 --no-recording --retry-rounds 0`；
三场对照使用 `--workers 3 --episode-indices 1445 1095 1280`，其余配置保持不变。
需要同时启动本机模型时增加 `--start-qwen`，已有服务则不增加。比较时同时核对
交互对象、内部步骤、关节轨迹、接触统计和任务结果；导航路径与机器并发负载变化时，
不得将总运行时间差全部归因于交互优化。

### 固定位置的交互专项测试

`scripts/InteractiveNav/run_fixed_interaction_test.py` 是独立诊断入口，不代替正式
导航评测：直接在指定交互位姿初始化机器人，关闭所选柜子的抽屉，固定完整
drawer_scan，不运行导航、ROS 或 Qwen，也不因找到目标提前结束。保留原生平滑
交互控制器、robot control/physics 和前后两次传感器观测；不计算导航奖励/终止。

输入 manifest 是 JSON 列表，每项包含 `episode`、历史 `smooth_interactions/*.json`
的 `macro_path`，可选 `pose_xyyaw`。未指定 pose 时读取历史结果的
`approach_goal_xyyaw`；优先用历史 trace 的 `actual_pose_xyyaw` 显式指定实际到达位姿，
而非把导航期望位置当作实际位置。源文件只提供关节身份、交互位姿和公开扫描参数，不能声称
恢复了未保存的历史完整物理状态。场景仍从同一 benchmark 初始化，两版使用相同
固定初态。初始化后同步控制器保持目标到当前关节/底盘位置，不重置机器人的固定位置。
一个 worker 对应一场；每轮等待全部场景就绪后同步开始计时。

```bash
TMPDIR=/home/ldl/tmp XDG_CACHE_HOME=/home/ldl/.cache \
PYTHONDONTWRITEBYTECODE=1 python scripts/InteractiveNav/run_fixed_interaction_test.py \
  --benchmark /home/ldl/path/to/benchmark.json \
  --manifest /home/ldl/path/to/fixed_cases.json \
  --output /home/ldl/outputs/interactive-nav/fixed-new-run --workers 3
```

`--prepare-only` 仅检查两版初态与模型一致性，不执行交互。输出目录必须不存在。
`--view-only` 先做低视角/回位短测试，不开抽屉，不能作为 drawer scan 加速结果。
`--adaptive` 对照实验性的中间滑动关节位置收敛策略，两轮都开启重复锁定合并。
该模式使用独立的 `effect_comparison.json` 验收，不再声称物理轨迹逐位相等：检查
观察帧/顺序、目标可见性、逐抽屉/阶段兜底、采样接触、观测及最终状态偏差、回位、
物理步与耗时。任一条件不通过即返回非零。它不能代替完整导航 SR/SPL 回归；
正式入口的 `INTERACTIVE_NAV_INTERMEDIATE_POSITION_ONLY` 默认关闭。
`--position-forward` 是另一项独立实验：保留所有物理步，只将锁姿时的完整 forward
替换为位置更新；仍使用严格轨迹比较，不能与 `--adaptive` 混用。
`--baseline-from <既有运行目录>` 可只读复用成功基线，要求 `run_config.json`、manifest
和基线模式匹配，并重新校验候选初态；输出明确记录引用来源，耗时不是同轮重跑的基线。
入口与正式评测一样在 reset 后恢复 benchmark 的物体/关节状态，再关闭所选抽屉。
每版保存 `ready.json`、`initial_state.npy`、`model_hashes.json`、完整轨迹
`trajectory.npz`、`execution.json`、`summary.json` 和 `profile/`。两版模型或完整
MuJoCo integration state 的哈希、控制器类型或保持目标不同时，优化轮不会开始。
结束后复查这些初态条件，并检查阶段/内部步数、
物理子步数、成功状态、所有关节位置/速度、接触数及最终状态；轨迹容差为 1e-8，
不一致时返回非零，不接受等价加速结论。计时不包括场景加载、初态准备和首次渲染。
配置开关由子进程显式设置，不依赖用户 shell 是否设置优化变量。

`--force-frequency-factor 2` / `4` 必须搭配 `--baseline-from`，只降低额外交互力控
循环的物理频率，不能与 adaptive/position-forward 混用。普通机器人控制周期、
policy 周期和物理步长不变；力控 PD 每物理步更新一次，其频率同步降低。
每段重放基线实际推进的仿真时长，末尾不足整步的余量用较短 timestep 补足，
不因提前收敛而缩短该段；稳定窗口按秒换算，原成功/兜底判据保持。
`frequency_drives.jsonl` 检查逐段时钟、物理步数和 MuJoCo 警告，
`frequency_config.json` 记录范围和步长。使用保守效果比较，不能要求不同积分步长
下轨迹逐位相同，也不能把通过专项测试表述为完整导航性能不变。
同机可同时启动两组各 3 worker 的命令，分别指定 factor 2/4 和独立输出目录。
若基线是历史 3 并发而候选合计 6 并发，耗时比仅供诊断，不能视为严格吞吐加速比。

`--lock-experiment camera_batch|geometry_only|combined` 测试不改物理时序的额外计算优化。
该模式的基线已经开启 coalesced lock 和 position-forward；候选分别暂缓力控段内
相机刷新至段末、只刷新锁姿后必需的几何/碰撞数据，或组合两者。具有 flex 的模型
不走精简几何路径。该模式不可与频率或 adaptive 等实验混用。
只读复用上一轮通过的优化波次时，增加 `--baseline-wave optimized`；不支持把尚未
验证的 lock/frequency 实验候选当作普通基线。默认仍可省略 baseline-from 跑新基线。
新生成的 lock 实验保存 `observation_hashes.json`，两版均存在时必须逐数组一致，
涵盖相机图像/深度以及其他数组型观测。严格比较还包括完整 frames.jsonl，
避免仅状态相同而可见性不同的情况漏检。历史基线缺少图像哈希时不声称像素级等价。
`--lock-experiment reference` 不启用任何新增优化，只保留 position-forward 基线，
用于同状态渲染/观测哈希的重复性对照。

### 2026-09-22：M2 在线 Mixed 0–29 配对实验

配置模板：`scripts/InteractiveNav/configs/evaluation/m2_online_mixed0_29_8arm.json`。
使用 v1.2 总表索引 2000–2029，共 8 × 30 = 240 次评测；动态预算 min200/cap2000，
不录制视频，保留决策、交互与请求追踪。G0 是历史上下文兼容基线，仍保留当前
TF、房间包围归属和执行修正，不等同于逐字复原历史 HTTP 请求。

| 组 | 上下文 | 候选池 | 提示词 |
| --- | --- | --- | --- |
| G0 | 历史兼容，recent8 | legacy | B |
| G1 | 新公共事实，recent30 | legacy | B |
| G2 | 新公共事实，recent30 | 全动作＋最多12前沿 | B |
| G3 | 同 G2 | 同 G2 | P1 目标语义链 |
| G4 | 同 G2 | 同 G2 | P2 初始位置邻近 |
| G5 | 同 G2 | 同 G2 | P3 新房间与历史 |
| G6 | 同 G2 | 同 G2 | P4 相关容器优先 |
| G7 | 同 G2 | 同 G2 | P6 综合偏好 |

G0→G1 比较上下文，G1→G2 比较整个候选池策略（不只是数量），G3–G7 与 G2 比较提示词。
Mixed 局部编号 `% 3` 分配到三个 lane，同一场景全部组合留在同一 lane，固定种子交错。
每 lane 15 workers / 80 jobs。本机 lane0/1 共用已有 `8000` TP=2 Qwen，**不得加
`--start-qwen` 或重启共享模型服务**；仿真分别用 EGL 0/1、ROS 19000/19100 起始端口。
远端 lane2 是一张卡，任务内启动 TP=1 Qwen＋仿真，模型上下文统一16384、无 MTP。
M2 timeout120、超时重试1、reasoning off、max_tokens1536；ROS starvation300。
正常算法失败不重试择优；基础设施/不完整结果另入 failure_queue，保留原 attempt。

冻结 manifest 后，通过统一入口分别启动（输出与缓存均须放在 `/home/ldl`）：

```bash
python scripts/InteractiveNav/run_m2_experiment_lane.py \
  --prepare-experiment scripts/InteractiveNav/configs/evaluation/m2_online_mixed0_29_8arm.json \
  --output-dir /home/ldl/outputs/interactive-nav/m2-online-mixed0-29-20260922
python scripts/InteractiveNav/run_benchmark_eval.py \
  --experiment-manifest /home/ldl/outputs/interactive-nav/m2-online-mixed0-29-20260922/manifest.json \
  --experiment-lane 0
python scripts/InteractiveNav/run_benchmark_eval.py \
  --experiment-manifest /home/ldl/outputs/interactive-nav/m2-online-mixed0-29-20260922/manifest.json \
  --experiment-lane 1
volc ml_task submit \
  -c scripts/InteractiveNav/configs/custom_task/m2_online_mixed0_29_8arm_remote.yaml
```

总体监控入口 `scripts/InteractiveNav/evaluation/m2_online_monitor.py --run-dir <RUN>`，
每30秒写 `overall.log` 和原子更新的 `overall_status.json`；可用 `tail -f <RUN>/overall.log`。
展示每组已完成 n 个的平均指标，同时保留计划分母30。ICS 取 episode `success`，
NavSR 单列，不使用 aggregate `success_rate` 冒充正式成功。未完成分母下的成功率只是
临时下界；infra、排队、heartbeat过期与正常算法失败分开显示。新增输出不纳入 Git。
