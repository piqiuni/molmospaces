# Interactive Navigation Evaluation Metrics

最后更新：2026-09-23

本文档固定当前交互式导航 benchmark 的论文指标定义；实际计算入口为 `scripts/InteractiveNav/evaluation/benchmark_metrics.py`。
可直接纳入论文主稿的 LaTeX 定义见 [`interactive_navigation_metrics_v2.tex`](interactive_navigation_metrics_v2.tex)。

当前主指标固定为 5 个：

| 指标 | 方向 | 作用 |
|------|------|------|
| `SR` | 越高越好 | 最终是否完成导航任务 |
| `SPL` | 越高越好 | 成功前提下的路径效率 |
| `Interaction Success Rate` | 越高越好 | 逐场景计算必要交互效果的完成比例，再对需要交互的场景平均 |
| `Interaction Precision` | 越高越好 | 全部交互尝试中，完成目标交互对象**类别**中新效果的比例 |
| `Total Cost` | 越低越好 | 路径、所有交互尝试、失败/无新效果重复及任务失败的总代价 |

`reachability`、`visibility` 和 `enablement` 不作为论文主表中的独立指标。它们是 benchmark 构建、episode 分层和 `Interaction Success Rate` 判定时使用的交互收益语义。

## 1. 评测对象

一个 episode 至少包含两类信息：

1. Benchmark GT：来自 `EpisodeSpec.interactive_nav`，当前优先采用 `interactive_nav_v3`。
2. Policy rollout：来自被评测方法实际执行后的日志。

Benchmark GT 可以包含：

- `interaction_domains`：`channel`、`container` 或二者混合。
- `interaction_requirement`：`required`、`beneficial`、`unnecessary` 或 `unknown`。
- `success_criteria`：最终 `nav_to_obj` 成功条件。
- `interactions`：可交互对象、关节、初始状态、目标状态、效果类型。
- `oracle_plan.required_interaction_ids`：完成该 episode 所需的关键交互 ID。
- `generation_validation.success_evidence`：数据生成阶段验证出的成功证据。

Policy rollout 应至少记录：

- 机器人 base 轨迹，用于计算实际路径长度。
- 终止状态下的目标距离与 head-camera 可见性。
- 所有交互尝试事件，包括对象、关节、时间戳、执行前后状态和是否达到目标状态。
- episode 是否 timeout、collision abort 或正常结束。

GT 信息只能由 evaluator/scorer 使用，不能暴露给 policy。

## 2. SR

`SR` 表示最终任务成功率。

单个 episode 的成功变量记为：

```text
S_i = 1, if distance_passed_i and visibility_passed_i
S_i = 0, otherwise
```

其中：

- `distance_passed_i`：机器人最终 base 到目标对象的平面距离满足 `success_criteria.distance.threshold_m`。
- `visibility_passed_i`：目标对象在 `head_camera` 中满足 `success_criteria.visibility.threshold`。
- 当前沿用 `NavToObj` 的成功逻辑，即距离条件和可见性条件都满足才算成功。

整体成功率为：

```text
SR = (1 / N) * sum_i S_i
```

`SR` 对所有 episode 都适用，包括 `channel`、`container`、`mixed` 和 `no-interaction` 样本。

## 3. SPL

`SPL` 表示 success weighted by path length，用于衡量成功方法的路径效率。

单个 episode：

```text
SPL_i = S_i * L_ref_i / max(L_ref_i, L_exec_i)
```

整体：

```text
SPL = (1 / N) * sum_i SPL_i
```

含义：

- `S_i` 是该 episode 的最终成功变量，失败 episode 的 `SPL_i` 为 0。
- `L_exec_i` 是 policy 实际走过的 robot base 平面路径长度。
- `L_ref_i` 是该 episode 的参考可行路径长度。

这里的 `L_ref_i` 不能使用纯静态地图上的最短路作为统一基准。对于交互导航任务，参考路径应来自允许必要交互状态变化后的可行计划：

- 无交互 episode：使用普通导航最短路或已有 `NavToObj` 参考路径。
- 通道交互 episode：使用“到门前 -> 打开通道 -> 穿过通道 -> 到目标”的参考路径。
- 容器交互 episode：使用“到容器前 -> 打开容器 -> 调整视角/观测目标”的参考路径。
- 混合 episode：使用包含必要通道交互和容器交互链的参考路径。

这样定义后，`SPL` 评价的是“在正确交互后，导航路径是否高效”，而不是拿不可行的静态最短路惩罚交互任务。

## 4. Interaction Success Rate

`Interaction Success Rate` 表示需要交互的 episode 中，**必要交互效果完成的比例**，允许部分完成得分。

默认只在 `interaction_requirement == "required"` 且 `oracle_plan.required_interaction_ids` 非空的 episode 上计算。对无交互样本，该指标记为 `N/A`，不进入该指标的分母。

单个 required episode 有一个或多个有效的必要交互计划。对计划 `p`，记其必要交互 ID 集合为 `R_{i,p}`，已成功产生预期物理效果的 ID 集合为 `G_i`。对有多个可行计划的 episode，取完成比例最高的一条：

```text
ISR_i = max_p |G_i ∩ R_{i,p}| / |R_{i,p}|
```

整体逐 episode 等权平均，不将所有动作合并为一个动作池：

```text
Interaction Success Rate = (1 / N_required) * sum_i ISR_i
```

例如 mixed 场景需要开门与打开容器两个效果，仅完成门时记 `ISR_i = 1/2`，而不是 0。`required_interaction_success` 仍是完整必要计划是否完成的二值诊断量，用于其他任务成功判定；它不能直接代替论文 ISR。对交互非必需或必要计划为空的场景，ISR 记为 `N/A`。多方案场景中的一个方案即便含更少的必要效果，也只和该方案自身的必要项比较，不把其他备选方案中的交互算作必须全部完成。

当前 evaluator 按 GT `interaction_id` 统计：对应交互操作成功，并在终态达到 0.8 开启比例，或有操作过程中的已完成效果记录，才计入 `G_i`。`effect_types` 描述必要交互服务的场景目标；ISR 不另行对每种 effect type 作独立的通行、可见性或下游动作验收。最终导航目标是否达成由 SR 单独衡量。

| effect type | 场景目标（不作为 ISR 的额外验收条件） |
|-------------|----------|
| `restore_reachability` | 打开通道以恢复到目标区域的可达性 |
| `reveal_target_object` | 打开容器以暴露目标对象 |
| `enable_interaction` | 完成上游操作以使下游交互可执行 |
| `reduce_navigation_cost` | 作为 beneficial/cost 类型标签，首版主要由 `SPL` 和 `Total Cost` 体现，不单独作为主表指标 |

对不同任务类型，`Interaction Success Rate` 的语义如下：

- `channel`：是否完成恢复可达性的关键通道交互。
- `container`：是否完成提升目标可见性的关键容器交互。
- `mixed`：必要交互链完成了多少项；中间通道交互可以体现 `enablement`，最终容器交互通常体现 `visibility`。

`Interaction Success Rate` 不等同于 `Oracle SR`。当前评测协议不设置“给 policy 已知 GT 交互计划”的条件，因此不报告 `Oracle SR` 作为主指标。Oracle plan 只用于 benchmark 生成验证、参考路径和 scorer 的 GT 判定。

## 5. Interaction Precision

`Interaction Precision`（IP）表示全部交互尝试中，有多少成功完成了**目标交互对象类别**中的新交互效果。类别取 benchmark 交互对象的 `object_category`（如 door、fridge、cabinet），同时区分 `channel` 和 `container` 领域；mixed 的目标类别取各领域目标类别的并集。类别匹配忽略大小写及下划线/连字符差异，不要求命中 oracle 的同一个对象实例或 joint ID。目标类别只用于 evaluator 私有评分，不公开给 policy。

单个 episode 中：

```text
Interaction Precision_i = V_class_i / A_i, if A_i > 0
Interaction Precision_i = 1, if A_i = 0 and interaction_requirement is unnecessary
Interaction Precision_i = 0, if A_i = 0 and interaction_requirement is not unnecessary
```

其中：

- `A_i` 是 policy 实际执行的交互尝试次数。
- `V_class_i` 是完成目标类别中新效果的交互尝试次数；一次多关节宏只计一次。

一次交互尝试仅在目标类别匹配、物理操作成功（或已有私有记录证明某个目标关节效果已实现）、且该对象/关节此前未完成相同效果时计入分子。打开另一扇门或另一台冰箱可以得到类别交互信用；打开非目标类别的容器不会得到信用，但也不自动被定性为错误。失败尝试及对已完成对象的无新效果重复仍进入分母，不进入分子。IP 衡量目标类别交互的完成密度，不是部分可观测环境中的最优探索策略证明；应结合 SR 随预算变化及每类操作次数解读。

结果字段 `non_target_class_interaction_attempt_count` 记录已识别类别但不属于目标类别的尝试。保留的旧字段 `task_irrelevant_interaction_attempt_count` 在 v2 中是同一计数的兼容别名，不代表该探索行为一定无用，也不自动计入成本中的错误数。

无交互 episode 也参与 `Interaction Precision`：

- 如果 policy 没有交互，记为 1，表示正确克制。
- 如果 policy 执行了任何交互，这些交互都计入 `A_i`，但不因此被认定为不合理探索。

整体指标默认使用 episode macro average：

```text
Interaction Precision = (1 / N) * sum_i Interaction Precision_i
```

scorer 可以额外输出 attempt-level precision：

```text
Attempt Precision = sum_i V_class_i / sum_i A_i
```

但论文主表优先使用 episode macro average，避免交互次数极多的少数失败 episode 主导整体结论。

## 6. Total Cost

`Total Cost` 与 evaluator v3 口径一致（其 IP 与成本公式承袭 v2，ISR 改为逐场景必要效果完成率），包含所有 episode：

```text
Cost_i = L_exec_i + lambda * A_i + mu * E_i + kappa * (1 - S_i)
```

其中：

- `L_exec_i` 是实际 robot base 平面路径长度。
- `A_i` 是交互尝试次数，成功、失败、重复交互都计入。
- `lambda` 是交互代价权重，用于把一次交互折算成等效路径长度。
- `E_i` 是失败或已无新效果的重复尝试数，同一尝试即使同时满足两项也只计一次；打开其他探索对象不会仅因对象不属于目标类别而计入 `E_i`。
- `S_i` 是严格 NavToObj 成功指示；`mu` 和 `kappa` 分别是错误尝试及任务失败的固定惩罚。当前默认 `lambda=0.3`、`mu=1`、`kappa=5`。

主表报告全部 episode 的平均总代价：

```text
Total Cost = (1 / N) * sum_i Cost_i
```

固定失败罚分并不能完全消除早停的低成本偏差，因此同时报告 SR、共同成功 episode 的成本与成功率随预算变化；不能单独用 Total Cost 判断规划能力。所有权重在评测前冻结：v1 与 v2 的 IP/Total Cost 不可直接比较，v2 与 v3 的 ISR 不可直接比较。

当前交互尝试使用统一 `lambda`。后续如果需要更细，可以扩展为：

```text
Cost_i = L_exec_i + sum_j lambda(type_j, success_j) + mu * E_i + kappa * (1 - S_i)
```

例如为 door、container、failed interaction 设置不同权重。但这属于后续扩展，不进入当前主指标定义。

## 7. Split 与汇总

主结果应至少按以下 split 输出：

| split | 含义 |
|-------|------|
| `all` | 所有可评测 episode |
| `channel` | 包含通道属性交互的 episode |
| `container` | 包含容器属性交互的 episode |
| `mixed` | 同时包含通道与容器交互链的 episode |
| `no-interaction` | GT 中不需要交互的 episode |

建议同时输出：

- micro average：按 episode 数量直接平均。
- macro average：先在各 split 内求平均，再对 split 求平均。

论文主表优先使用 split 结果加 macro average，避免某一类样本数量过大时掩盖方法在小类上的失败。

## 8. 无交互样本

无交互样本必须保留。它们的作用不是验证交互能力，而是验证方法是否会过度交互。

对 `interaction_requirement == "unnecessary"` 或 `interactions` 为空的 episode：

- `SR`：正常计算。
- `SPL`：正常计算，参考路径为普通导航参考路径。
- `Interaction Success Rate`：记为 `N/A`，不进入分母。
- `Interaction Precision`：无交互且无尝试为 1；有尝试但无目标类别时为 0，不以此判定探索决策错误。
- `Total Cost`：正常计算，多余交互会通过 `lambda * A_i` 增加代价。

无交互样本也参与主指标：尝试会支付真实操作成本；IP 的分母记录了这些尝试，但不把未知环境中的首次检查定义为失败。

## 9. 与 Interactive Gibson 的关系

Interactive Gibson 文章中用于训练和比较 baseline 的 reward 可以理解为三部分：

- `R_suc`：导航成功奖励，对应当前的 `SR`。
- `R_pot`：到目标 geodesic distance 的进展奖励，对应当前的路径效率思想，主要由 `SPL` 体现。
- `R_int`：交互惩罚，对应当前的交互 effort，主要由 `Interaction Precision` 和 `Total Cost` 体现。

差异在于，当前 benchmark 不把 reward 当作最终评价指标，而是拆成方法无关、可解释的评测指标。同时，当前不报告 `Oracle SR`，因为当前协议不设置“policy 已知 GT 交互序列”的评测条件。

## 10. Scorer 输出要求

`score_interactive_nav_run.py` 应输出两级结果。

Episode 级字段：

- `episode_id`
- `split`
- `success`
- `path_length`
- `reference_path_length`
- `spl`
- `required_interaction_ids`
- `attempted_interaction_count`
- `valid_interaction_count`
- `interaction_success`
- `interaction_precision`
- `total_cost`
- `failure_reason`

汇总表字段：

- `split`
- `num_episodes`
- `SR`
- `SPL`
- `Interaction Success Rate`
- `Interaction Precision`
- `Total Cost`

保留 per-episode 诊断字段的目的，是支持附录分析和数据质量检查；论文主表仍只展示上述 5 个主指标。
