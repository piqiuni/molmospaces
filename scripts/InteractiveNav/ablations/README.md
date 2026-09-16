# 模块级消融

共同基线为 `154aa0226e7ce42bcbc8b3efe31bfd1990ab5a36`，标签
`codex/experiment-baseline-20260917`。只设置 Full + 三个模块级对照，分别回答：
关系表示是否有用、任务语义是否改善选择、交互结果是否改善后续决策。

| `--variant` | 干预 | 保留内容 | 建议论文名称 |
| --- | --- | --- | --- |
| `full` | 使用原始 runner 和全部原始节点 | 完整方法 | Full |
| `no_interaction_graph` | M2 输入改为扁平对象记忆；移除房间关系、边、显式交互效果、关系提示与图导出的预评分；候选压缩使用距离和原有类型配额 | 同一模型、对象类别/位置/状态、任务、交互执行、结果更新 | w/o Graph-guided Reasoning |
| `no_task_decision` | M2 换成最近可行交互规则：优先交互，无交互时选择最近探索/导航候选；不请求 M2 模型 | 图、M1、执行器、结果更新；共用启动扫描、可靠目标和交互后续动作优先门控 | w/o Task-conditioned Selection |
| `no_outcome_update` | mapping 不消费交互命令/结果以修改图或建立乐观地图覆盖；决策不保留结果驱动的开门后穿越、刷新门控和结果信念 | 正常 RGB-D/GT 可见观测、M1 属性更新、OCC 更新、执行成功/失败回报、冷却、目标完成判定 | w/o Outcome-driven Update |

`no_interaction_graph` 的范围是**高层关系图推理**。底层候选生成、可达性检查和执行仍复用原图，
公共的目标到达与交互后续动作门控也保留。因此不能将本版本写成“系统完全不构建交互图”，
也不能仅凭这一对照证明整套感知/建图组件的独立贡献。它衡量的是：在相同感知和可执行动作接口下，
显式关系信息比扁平对象记忆多带来多少收益。

`no_outcome_update` 不冻结世界。开门后的空间仍可由下一次传感器观测发现；
关闭的是动作结果直接推动语义状态/后续计划的路径。静态门洞经 OCC 确认的通行候选仍保留。
当前基线以 backend interaction result 驱动更新，这个消融不应被描述成独立的视觉 M3 审计消融。

## 运行

新入口复用 `configs/evaluation/benchmark_batch.json`、原 evaluator、动态预算、录制、模型入口、
观测轮次倍率和进程清理。以下命令仅检查配置并输出计划，不创建文件、不启动 ROS/模型/仿真：

```bash
/home/ldl/conda_envs/mlspaces/bin/python \
  /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/run_benchmark_ablation.py \
  --variant no_interaction_graph --workers 1 --episode-indices 10 --max-steps 20 --dry-run
```

另外三个取值为 `full`、`no_task_decision`、`no_outcome_update`。
实际运行时去掉 `--dry-run`；正式实验不要沿用示例中的单场 20 步限制。
通过相同的 `--config` 和 `--episode-indices` 为四组指定相同任务；每组使用独立输出目录。
其他参数与原入口一致：`--workers`、`--max-steps`、`--recording/--no-recording`、`--output-dir`。
当前入口每次创建新目录，不复用已存在的运行目录。

每轮输出包括 `ablation_manifest.json`（变体、基线 SHA、实际 Git HEAD/状态、配置、启动命令、
适配文件哈希）、`launch_config.json` 和原批量评测产物。消融组在 `ablation/` 内保存
适配代码副本、ROS launch 和 runner；原始源码和 launch 文件不做修改。
runner 哈希包含适配代码身份，节点 `~module_ablation` 参数和 ROS 日志记录实际变体。
原 YAML 的 M1/M2/M3 字段用于共用配置加载；消融的实际干预应以 manifest 和节点参数为准。

## 如何呈现贡献

主表保持四行。主指标沿用论文和当前 evaluator 的 SR、SPL、ISR、Interaction Coverage，
同时区分正式 `success`、`task_success` 和 `nav_success`，不能混用分母或目标条件。
在同一批结果内分别统计 channel、container、mixed，以及必要交互数量；这些是结果分组，不新增消融组。

- 图推理：重点看多房间、隐藏目标和多步依赖任务。若 SR/SPL 下降且无关交互或绕行增多，
  才能把收益解释为关系表示支持的规划，而非模型规模或候选可执行性变化。
- 任务选择：重点看相似容器、多可选门和有干扰交互的任务。结合错误对象交互、冗余交互和路径长度，
  判断模型是否将交互用在了目标搜索上。该组 M2 调用减少，墙钟时间需单独报告。
- 结果更新：重点看开门后穿越、打开容器后搜索及连续交互。结合交互成功后仍未完成任务、
  重复选择、停滞和恢复次数，判断结果是否转化成了后续导航进展。

后面提到的诊断计数可从 trace/events 派生；本新增代码没有实现新的汇总指标，不能当作已生成的结果。
操作控制器完全共用，物理交互成功率的变化也可能来自对象选择和状态维护，不等于控制能力变化。

四组固定 manifest、episode/seed、初始状态、传感器限制、目标成功条件、预算和观测上限、
模型版本及推理参数、并发和录制开关。采用同 episode 配对比较，并报告完成比例与异常数量。
正式实验建议对四组都重复相同的三轮，报告均值和配对不确定性；不得挑选某组最好的一轮。
本次只进行了轻量测试，没有产生算法性能结论。

## 轻量验证

```bash
cd /home/ldl/molmospaces-exp-setting
source /home/ldl/conda_envs/ros-noetic/setup.bash
TMPDIR=/home/ldl/tmp/interactive-nav-ablation-check \
XDG_CACHE_HOME=/home/ldl/.cache/interactive-nav-ablation-check \
PYTHONDONTWRITEBYTECODE=1 \
/home/ldl/conda_envs/mlspaces/bin/python -m pytest \
  scripts/InteractiveNav/ablations/tests -q -p no:cacheprovider
```

测试包含扁平记忆在 HTTP 请求中的实际传输、无关系评分泄漏、最近规则、输入不变性、
原决策节点的动作选择/成功反馈生命周期、感知/OCC 回调保留、运行快照与清理，以及四组 dry-run。
ROS transport、模型响应和启动进程使用替身，不需要 roscore、GPU 或在线模型。
仅测试 ROS 节点的模块需要 Conda ROS；未加载 ROS 时该模块会 skip。

2026-09-17 检查结果：28 项新增测试 + 100 项原有相关回归，合计 128 项通过；
三组生成的完整 launch 均通过实际 ROS loader 解析，Python AST 和 Shell 语法检查通过。
日志：`/home/ldl/outputs/interactive-nav/ablation-check-20260917/pytest-reviewed.log`。
