# Interactive Navigation Evaluation Metrics

最后更新：2026-07-19

本文档固定当前交互式导航 benchmark 的评测指标定义。目标是保持论文主表简洁，同时让后续 `score_interactive_nav_run.py` 能按同一套口径实现可复现评测。

当前主指标固定为 5 个：

| 指标 | 方向 | 作用 |
|------|------|------|
| `SR` | 越高越好 | 最终是否完成导航任务 |
| `SPL` | 越高越好 | 成功前提下的路径效率 |
| `Interaction Success Rate` | 越高越好 | 需要交互时，关键交互效果是否真的完成 |
| `Interaction Precision` | 越高越好 | 执行过的交互中，有多少是有效交互 |
| `Total Cost` | 越低越好 | 同时考虑导航距离与交互次数的总代价 |

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

`Interaction Success Rate` 表示需要交互的 episode 中，关键交互效果是否完成。

默认只在 `interaction_requirement == "required"` 且 `oracle_plan.required_interaction_ids` 非空的 episode 上计算。对无交互样本，该指标记为 `N/A`，不进入该指标的分母。

单个 required episode：

```text
ISR_i = 1, if all required interactions are completed with their expected effects
ISR_i = 0, otherwise
```

整体：

```text
Interaction Success Rate = (1 / N_required) * sum_i ISR_i
```

关键交互是否完成，以 GT `interaction_id` 和 `effect_types` 为基准：

| effect type | 当前判定 |
|-------------|----------|
| `restore_reachability` | 通道交互达到目标状态后，机器人能够通过该通道或到达通道后的目标区域 |
| `reveal_target_object` | 容器交互达到目标状态后，目标对象满足可见性证据，例如 `visibility_fraction > threshold` 或 `visible_pixels > 0` |
| `enable_interaction` | 该交互完成后，下游必要交互变得可到达、可执行或已经被成功完成 |
| `reduce_navigation_cost` | 作为 beneficial/cost 类型标签，首版主要由 `SPL` 和 `Total Cost` 体现，不单独作为主表指标 |

对不同任务类型，`Interaction Success Rate` 的语义如下：

- `channel`：是否完成恢复可达性的关键通道交互。
- `container`：是否完成提升目标可见性的关键容器交互。
- `mixed`：是否完成必要交互链；中间通道交互可以体现 `enablement`，最终容器交互通常体现 `visibility`。

`Interaction Success Rate` 不等同于 `Oracle SR`。当前评测协议不设置“给 policy 已知 GT 交互计划”的条件，因此不报告 `Oracle SR` 作为主指标。Oracle plan 只用于 benchmark 生成验证、参考路径和 scorer 的 GT 判定。

## 5. Interaction Precision

`Interaction Precision` 表示执行过的交互中，有多少是有效交互，用于惩罚乱开门、乱开容器、重复打开已经完成的对象等行为。

单个 episode 中：

```text
Interaction Precision_i = V_i / A_i, if A_i > 0
Interaction Precision_i = 1, if A_i = 0 and no GT interaction is required
Interaction Precision_i = 0, if A_i = 0 and GT interaction is required
```

其中：

- `A_i` 是 policy 实际执行的交互尝试次数。
- `V_i` 是有效交互尝试次数。

一次交互尝试计为有效，需要满足：

1. 匹配 benchmark 中的某个 GT `interaction_id`。
2. 交互方向与目标状态一致，例如需要打开时确实朝打开方向执行。
3. 执行后达到该交互的 `target_state`，或满足对应 `effect_types` 的成功证据。
4. 不是已经完成后的重复无效尝试。

无交互 episode 也参与 `Interaction Precision`：

- 如果 policy 没有交互，记为 1，表示正确克制。
- 如果 policy 执行了任何交互，这些交互都计入 `A_i`，且通常为无效交互。

整体指标默认使用 episode macro average：

```text
Interaction Precision = (1 / N) * sum_i Interaction Precision_i
```

scorer 可以额外输出 attempt-level precision：

```text
Attempt Precision = sum_i V_i / sum_i A_i
```

但论文主表优先使用 episode macro average，避免交互次数极多的少数失败 episode 主导整体结论。

## 6. Total Cost

`Total Cost` 衡量完成任务所付出的总代价，首版定义为：

```text
Cost_i = L_exec_i + lambda * A_i
```

其中：

- `L_exec_i` 是实际 robot base 平面路径长度。
- `A_i` 是交互尝试次数，成功、失败、重复交互都计入。
- `lambda` 是交互代价权重，用于把一次交互折算成等效路径长度。

默认主表报告成功 episode 上的平均总代价：

```text
Total Cost = mean_i Cost_i, for episodes with S_i = 1
```

这样可以避免“很早失败所以代价很低”的方法在 cost 指标上看起来更好。完整 scorer 仍应输出 all-episode cost 作为诊断字段，但论文主表中的 `Total Cost` 默认理解为 success-only cost。

首版使用统一 `lambda`。后续如果需要更细，可以扩展为：

```text
Cost_i = L_exec_i + sum_j lambda(type_j, success_j)
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
- `Interaction Precision`：无交互且无尝试为 1，有多余交互则降低。
- `Total Cost`：正常计算，多余交互会通过 `lambda * A_i` 增加代价。

因此，无交互样本不需要额外设计一个主指标；它们会通过 `Interaction Precision` 和 `Total Cost` 惩罚多余开门、乱开容器等行为。

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
