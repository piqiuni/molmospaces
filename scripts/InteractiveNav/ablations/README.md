# 模块级消融（设计版本 3；第三组视觉重观测修正）

## 版本 3：交互后先完成视觉更新

版本 2 的第三组在 mixed 15/29 出现大量重复开门。执行结果事件包含实际操作和
`action_executed=false` 的纯观察结束；后者不能再清空 M1 共识。实际操作完成时仍清除
操作前缓存，重复投递的同一完成事件只处理一次，不读取其开闭状态标签。

第三组新增按对象记录的视觉刷新屏障：操作之后，旧图产生的 INTERACT 必须先进入
原生的 observation-only 流程；新视觉状态写回图后才允许重新操作。门要求新的三视角
共识；状态闭合后仍允许再次操作，不用固定重试上限替代状态感知。观察超时不能走
物理操作 fallback。该屏障同样作用于 M2 返回后的最新候选复核。
Full、Flat、Greedy 和原生映射/执行器源码不修改。

单独重跑第三组可用共享队列入口的 `--variants no_outcome_update`，并用上次保存的
`selected_scene_source.json` 固定十个场景。`--workers 25` 是并发上限；仅十个任务时
实际并发为十。不同并发和随机模型请求会影响耗时与成功率，需在结果中注明。

修正验证：51 项消融适配器和原生门状态共识测试通过，覆盖纯观察不清票、结果去重、
交互后证据新鲜度、跨 episode 重置、M1 超时不直接执行，以及真实决策节点复核。

原始实验基线为 `154aa0226e7ce42bcbc8b3efe31bfd1990ab5a36`，标签
`codex/experiment-baseline-20260917`。旧适配器为 `de80fa41441209910e4851051960dafa80362a88`。
旧结果对应设计版本 1，不能直接放入本版本的消融表。仍只使用 Full + 三个对照：

| `--variant` | 实际干预 | 共同保留 | 建议名称 |
| --- | --- | --- | --- |
| `full` | 完整算法 | — | Full |
| `no_interaction_graph` | 候选节点仅保存类别、位置、几何与可见性等 flat object memory；移除关系、交互状态和操作历史；重新生成候选；M2 不接收状态或关系提示 | 同一 M2 模型、目标语义、局部导航和执行器 | Flat Object Memory |
| `no_task_decision` | 在 Full 相同的候选筛选/压缩之后，以交互优先、距离最近替换 M2 选择，不调用 M2 | 图、M1、候选池及配额、可靠目标优先、执行与结果更新 | Greedy Interaction Selection |
| `no_outcome_update` | mapping 不消费命令/结果以修改图或建立乐观覆盖；M1 不接收结果状态标签；决策清除结果驱动的后续动作和结果信念 | 持续 M1、可见对象观测、OCC、执行生命周期与失败冷却 | Perception-only Update |

## 干预边界

**Flat** 在候选节点收到图时就清除关系、开闭/可达状态、drawer scan 等交互记忆，
不是仅从 M2 prompt 隐藏字段。生成器把门/容器视为状态未知的操作提议，不利用包含关系
定位目标，也不根据成功历史自动穿门。M2 历史上下文排除 INTERACT 记录。
当前动作生命周期、失败/重复调度保护仍保留。底层 mapper 和局部执行器仍维护原生信息，
用于碰撞检查、操作前 M1 观测和物理执行；这些交互状态不进入 flat 规划记忆。
因此应表述为“移除规划层的结构化交互状态表示”，不能写成整个系统没有任何图/状态。
这是结构与交互状态记忆的联合消融，不能将差异全部归因于边结构。

**Greedy** 保留 Full 的 model 分支，只替换 selector，避免旧版 rule 分支在 curation 之前
选择导致候选池不同。公共启动扫描、可靠目标到达、结果驱动穿门、空候选 fallback 和
最新快照复核保持不变。Full 的 curator 仍可使用目标语义，故该组不能声称完全去掉
任务条件；它衡量的是任务条件的 M2 选择在共用候选集合之上的增益。
贪心优先当前可行交互，不代表访问所有对象，也没有成功率必然更高的理论保证。

**Perception-only** 保留正常感知更新，状态不会冻结。执行完成事件只用于清除过期的
在途 M1 请求/缓存并安排新观测，不读取 success/post_state 来确定物理状态。
旧版只关闭 mapping 回调，遗漏了 M1 的 `record_authoritative` 写入，本版本补齐隔离。
门被 M1 与 OCC 联合确认打开后，可以独立生成穿门目标，不要求存在交互成功历史；
复用原生静态门洞的视觉共识、连通性和几何检查，临时转换仅存在于候选计算中。
执行器回报仍用于结束当前动作与失败冷却。本组不是 M3 验证消融，也不是忽略所有反馈。

## 持续 M1：四组必须共用配置

原生 M1 并非只处理 unknown：已知对象默认可在 120 秒后再次分析；uncertain portal
使用较短间隔，门的稳定状态另有 300 个 evaluator step 的冷却。状态变化需要 3 个不同视角
确认。`success_refresh_interval_s <= 0` 会禁止同签名的周期刷新，不能用 0 表示持续分析。

新增 `--m1-refresh-profile continuous`：已知对象刷新间隔 30 秒、门状态冷却 60 step，
保留可见性、视角确认、请求去重、队列和超时限制。这是新实验的初始调度值，尚未证明最优；
它表示可见对象的有界周期复查，不是逐帧查询所有对象。`baseline` 仍复用原始设置，
Full + baseline 直接使用原始 runner。

新实验四组都使用 continuous，不能只给第三组增加观测预算。相同调度不代表实际调用数
相同：Full 有结果信息，可能减少后续视觉确认；必须另报 M1 请求数、tokens、等待耗时。
持续 M1 可估计当前开闭状态，但无法仅从“抽屉现已关上”恢复“已完成打开—扫描—关闭”，
因此核心比较是动作条件的状态/事件更新相对于感知重建的及时性与完整性。

## 运行与记录

60 个 mixed 场景为场景 0–59，对应 benchmark episode 2000–2059。
新版公共配置：`scripts/InteractiveNav/configs/evaluation/ablation_v2_mixed_0_59_20w_dynamic2000_no_recording.json`。
保留动态 200–2000 步、20 workers、观测倍率 1、无录制，新增共同 continuous M1 与独立输出目录。

以下仅验证计划，不创建文件或启动仿真：

```bash
/home/ldl/conda_envs/mlspaces/bin/python /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/run_benchmark_ablation.py --variant full --config /home/ldl/molmospaces-exp-setting/scripts/InteractiveNav/configs/evaluation/ablation_v2_mixed_0_59_20w_dynamic2000_no_recording.json --dry-run
```

四组分别替换 `--variant`，每次创建独立目录。正式运行不应混用旧版 Full 数据；
并发启动不同组时需要独立 master 端口和相同资源安排。
`ablation_manifest.json` 记录 design_revision=2、M1 profile/实际覆盖值、基线 SHA、
当前 Git 状态、配置、命令和适配文件哈希。原生 ROS/算法源码与 launch 不改动；
适配代码及生成的 launch/runner 存于本轮输出目录。原 YAML 模块名不能代替 manifest 判断干预。

## 如何判断 Full 的贡献

主表保持四行：SR、SPL、有效交互精度、重复/无关交互数、路径/动作成本、M1/M2 调用及墙钟耗时。
使用同 episode/seed/初态/成功条件/动态预算，固定模型参数、并发度与录制开关。

- Full vs Flat：关注跨房间、容器搜索和多步状态依赖场景，报告失败原因、绕行与重复操作。
- Full vs Greedy：检验相当 SR 下能否减少冗余交互与行动成本，不预设 Full 的所有指标更优。
  同时报全体场景和双方共同成功场景上的成本；提前失败可使总成本降低，不能据此称更高效。
  Greedy 节省 M2 推理，Full 即使物理行动更高效也未必墙钟更短。
- Full vs Perception-only：统计动作完成到状态确认/下一步推进的延迟、重新观测次数、重复操作，
  以及“物理状态恢复但事件历史不可恢复”的案例。不把重复操作失败全部解释为视觉准确率问题。

结论需由配对结果支持；三组可以验证完整系统各部分的必要性，不能严格识别模块间的统计
交互效应。新诊断指标尚需从事件日志汇总，本次没有生成正式性能结果或宣称 Full 更优。

## 轻量验证

加载 `/home/ldl/conda_envs/ros-noetic/setup.bash`，设置 TMPDIR 为
`/home/ldl/tmp/interactive-nav-ablation-v2`、XDG_CACHE_HOME 为
`/home/ldl/.cache/interactive-nav-ablation-v2`，运行：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/ldl/conda_envs/mlspaces/bin/python -m pytest scripts/InteractiveNav/ablations/tests -q -p no:cacheprovider
```

测试使用原生候选生成器、决策节点和 M1 调度/共识逻辑，替换 ROS transport、在线模型和进程。
覆盖 flat 状态隔离、四组 M1 配置一致性、Full/Greedy 候选池一致性、仅感知穿门、
结果标签隔离与周期刷新去重。实际仿真性能仍需四组配对评测验证。

2026-09-17 版本 2 初始验证：39 项适配器测试及 114 项原生相关回归，共 153 项通过。
四组完整 launch 经 ROS loader 解析，确认对象 M1 实际参数为 30 秒 / 60 step；
四组 60 场景 dry-run、Python AST、生成 runner 的 Shell 语法检查通过。
上述是版本 2 开始评测前的验证记录；版本 3 修正与重跑说明见本文开头。
