# 22 日基线选择性迁移报告

日期：2026-09-23。状态：代码迁移与定向回归完成，未运行 ROS 端到端性能评测。

## 1. 分支与工作区

- 当前新分支：`codex/migrate-22-selected`。
- 基线：`d3f27870be13c05fbcff7bfb756da4949197985b`（22 日）。
- 迁移来源：`8fb9460ae8db3e733140b7fc9039c3a128af7cb1`（23 日用户提交）。
- 原分支 `codex/exp-setting` 保留上述来源提交，没有重置、覆盖或改写历史。
- 本轮是选择性迁移，不是整提交 cherry-pick；改动留在新分支工作区，尚未提交。
- 原有未跟踪文件 `mobileclip_blt.ts` 与 `benchmark_local2gpu_30mixed_gpu23_20260923.json` 原样保留；后者没有作为本轮默认配置。
- 未修改 benchmark 数据集、模型权重、认证、个人 dotenv 或云端任务配置；未启动云端任务或长时仿真。

## 2. 八项迁移结果

### 2.1 M1 总额度与定向预留

入口：`Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/interaction_attribute_inference_node.py`，参数在同包 `config/default.yaml`。

- 每 episode、每个检测对象 ID 的 M1 模型提交上限为 10。
- 普通发现最多使用 8 次；最后 2 个额度保留给决策层定向刷新。定向请求与普通请求共用总额度，并不是额外赠送 2 次。
- 模型提交前在锁内检查当前 generation/request sequence 并计数，避免多 worker 超额提交。
- 实际调用模型接口后失败或超时，也消耗总额度；本地排队过期但没有提交，不消耗额度。
- 总额度耗尽时，定向请求及时收到 `m1_call_budget_exhausted`，不再无意义等待下一张 RGB。
- 已有稳定门状态缓存仍可满足请求，不必新增模型调用。
- episode 切换时清空计数；状态监控输出总调用数、达到额度的对象数及预算抑制次数。
- **没有迁入连续失败 3 次后的 M1 节点级永久封锁**：连续失败后仍可用剩余额度和定向预留。22 日决策层已有的失败冷却、候选耗尽与终止规则没有删除。

验证包括：60 个不同视角只能消耗 8 次普通额度；2 次定向仍可用；10 次后拒绝；连续 8 次失败不提前硬封锁；队列过期不扣额度；换 episode 重置。

限制：额度按检测对象 ID 而非真实物理实例计数。实例 ID 碎片化仍可能增加总流量；10 次全部超时也仍会耗尽额度。它不是后端容量不足的根治方案。

### 2.2 Room 失败冷却与 fallback

- 同一 room/evidence signature 失败后冷却 30 秒，再允许模型刷新。
- 模型失败或排队过期时，可由已有 `WeightedRoomAttributeInferencer` 按物体标签生成规则结果；通过 `fallback_enabled` 控制。
- fallback 以 ready patch 发布，保留原始错误，并记录 `room_attribute_fallback`，不会伪装成模型成功。
- fallback 不写入模型成功缓存；真实模型成功仍使用 120 秒刷新间隔。
- 保留基线 room ID + 成员标签的证据签名与 mapper 发布方式；没有引入 XY box、几何签名失效、邻近房间/dirty 增量调度。
- 不新增 15 秒 room 超时配置，继续继承 M1 请求超时。标准 V3 eval 默认路径下为 30 秒；其他入口仍遵循各自的基线设置。

作用：降低相同失败证据的重复模型压力，同时让图可以先使用可识别的规则回退。成员标签变化仍可能触发新请求，规则判断也不保证正确。

### 2.3 ROS 动作等待 0.2 → 0.4 秒

修改标准 eval 配置、batch runner 的环境默认值以及普通 evaluation runner 的默认值。显式环境覆盖仍优先。

这是等待新鲜 ROS 动作的桥接时间，不是 M1/M2/M3 模型超时。可以减少调度抖动导致的 no-op，但在确实没有命令的步骤上可能多等待 0.2 秒；不能仅凭此修改宣称每 step 更快。底层 benchmark API 的其他既有默认值没有整体重写。

### 2.4 同锚点空列表、GPU 检查与监控

- `semantic_behavior_executor.py` 新增 `_mark_same_anchor_resume`：恢复同锚点导航时若历史列表为空，先用已选 preflight/goal 信息创建记录，再设置恢复标记，避免访问 `[-1]` 崩溃。两个恢复分支都使用同一 helper。
- `run_benchmark_eval.py` 新增 GPU preflight：检查 EGL 分配是否在可见设备列表中、是否达到要求的唯一设备数，支持自动分配，保存 `gpu_preflight` 信息。
- 自启动 Qwen 时保留所选设备 ID，不再把非连续设备 ID 重编号为 `0..N-1`。
- 没有迁入 Qwen LB 启动器、自动并发扩容或 max-num-seqs 调整；保留 22 日单实例服务实现。
- 迁入只读周期监控 `monitor_mllm_timeouts.py` 与离线汇总 `collect_timeout_performance.py`，区分 M1 objects / M1 rooms / M2 / M3。
- 迁移适配：监控同时识别基线 `vllm.launcher.log` 和 `gpu*.launcher.log`；截断重建日志后重置增量。离线统计识别显式 timeout 标记，忽略未完整写入的行，并按 attempt 分开统计对象额度，避免把 episode 重跑累加误报为超 10 次。

GPU 检查使用已有 `visible_devices` 接口，不是 GPU 压测或显存容量保证。周期监控仍需云任务 ID；本地离线检查可直接使用 collector。

### 2.5 冰箱接近容差

冰箱 operational-front-face 合同下的位置方位角与朝向容差由 15° 放宽到 25°。仍保留可达性、站距、安全检查以及 M1 正面授权要求；没有对所有容器统一放宽。

导航到达 yaw 容差继续取配置限制与剩余角度预算中的较小值。针对 fan ±5°、配置 0.20 rad 的测试，正确结果为 0.20 rad，不是旧 10°；已同步修正来源提交中遗漏的断言。

### 2.6 目标成功声明

- `TargetMissionTracker.claim_ready` 在已有到达/观察要求之上，要求当前可见，并验证有限、非负且符合阈值的距离证据。
- 导航/交互反馈完成时，决策节点重新查找最新候选，而不是直接用历史选中候选宣布成功。
- V3 full-MLLM 配置开启 `target_require_current_visibility`。
- restricted evaluator 再检查当前可靠感知、公开距离证据与目标实例证据；公开证据账本启用当前帧/有限年龄检查。
- 已有打开容器锚点成功合同保留，但仍必须满足当前可见与有效锚点条件；不是一律要求机器人靠近不可进入的容器内部中心。

这是成功判定收紧，可能降低旧口径中由历史证据或过早声明带来的表面成功率；不能把这类变化简单归因为导航能力退化。

### 2.7 同类物体过滤

- 新增 `scene_distractor_filter.py`，默认启用 controlled single-instance 评测变体。
- 在每 episode 的运行时场景编译前，移除与目标类别相同的非目标实例，兼顾 object poses 与静态 body 名称。
- 保护选定目标、task-relevant 对象、交互计划引用对象、运行时新增对象；支持现有类别别名和带路径的实例名。
- 只修改运行时 EpisodeSpec，不改磁盘 benchmark JSON；运行元数据保留删除列表。
- 通过 `REMOVE_SAME_CATEGORY_DISTRACTORS=false` 或 evaluator 的 `--no-remove-same-category-distractors` 可关闭。
- batch evaluator 指纹包含新增过滤模块，避免在代码合同变化后直接复用不匹配的完成结果。

这改变评测场景难度和歧义，不是纯性能优化。与旧结果比较时必须使用相同开关；过滤开关不同的 SR 不能直接当作算法改善证据。

### 2.8 指标 v3

- Schema：`interactive_nav_v3_paper_metrics_v3`。
- ISR：对 required episode，取必要交互备选计划中完成比例最高者，再逐 episode 平均；例如两个必要效果完成一个记 0.5。完整必要交互是否成功的旧二值字段保留作诊断。
- IP：目标交互类别中成功产生新物理效果的尝试 / 所有尝试，采用 episode macro average；不是必须命中同一个 oracle 实例。
- Total Cost 保留 `L_exec + lambda*A + mu*E + kappa*(1-S)`；E 只计失败或无新效果重复，不把其他类别的成功探索自动当作错误。
- SR/SPL 的论文汇总仍使用 `nav_success`；不应与更严格的组合任务成功字段混用。
- 更新结果字段、CLI manifest、round summary、公开结果裁剪合同以及指标说明/LaTeX 定义。
- 没有迁入手工调整成功样本的论文表格，也没有修改历史评测产物。新旧 schema 的汇总不应直接混合。

## 3. 明确未迁入的超时相关与实验性改动

| 项目 | 本分支处理 | 原因 |
|---|---|---|
| 标准 V3 M1 30 → 15 秒 | 保留 30 秒 | 不把慢响应过早截断成失败 |
| 独立 room 15 秒超时 | 不迁，继承原请求超时 | 避免 room 新增短截止时间 |
| M2 eval 30 → 12 秒 | 保留 30 秒、1 次 timeout retry、1 秒 backoff | 保留慢请求成功机会，90 秒 starvation guard 不变 |
| M3 配置/执行器超时收紧 | 保留 22 日各层原值 | 不混入不同层级等待窗口变化；V3 model verification 为 12 秒，执行状态机默认 30 秒 |
| M1 连续失败/队列过期硬封锁 | 不迁 | 排队或暂时拥塞不应提前剥夺剩余定向额度 |
| 共享模型 client/env timeout 默认变化 | 不迁 | 保留基线公共请求合同 |
| Room XY box 与几何签名、增量 mapper 调度 | 不迁 | 与此次失败冷却解耦，避免同时改变证据失效和提交节奏 |
| Qwen LB / 自动并发 / 新云端 GPU-worker 配置 | 不迁 | 不同时改变服务容量和资源竞争条件 |
| episode stride offset、新实验批次、论文结果表 | 不迁 | 不改变评测采样集合或人为调整统计 |

注意：保留的是“各入口原值”，不是把所有入口都改成 30 秒。非 V3 的原普通探索入口 M1 默认 8 秒、节点代码 fallback 等仍是基线值。显式环境变量和所选 dotenv 仍能覆盖默认配置；实际跑实验前必须核对启动日志中的最终生效值。

## 4. 验证与边界

- 最终定向及相邻回归：**706 passed，0 failed**，35.49 秒，无测试跳过。
- 覆盖：M1 请求状态/额度、room inference、交互图、候选生成、任务完成、执行状态机、同锚点恢复、决策导航恢复、距离验证、GPU launcher、batch runner、监控、同类过滤、V3 指标/评测器/CLI/公开合同/round summary。
- 首轮 449 passed、2 failed：两项都是迁入行为已变化但断言仍使用旧值（冰箱到达 yaw、ROS 0.2 秒）；修正断言后完整扩展回归通过，没有通过放宽代码来掩盖失败。
- 31 个修改/新增 Python 文件通过 AST 解析，3 个 YAML 通过解析，2 个 shell/config 文件通过 `bash -n`；两个监控 CLI 的模块式 `--help` 通过；`git diff --check` 通过。
- 另外逐字节核对 12 个明确排除的核心 mapper/client/env/服务/默认配置文件与 22 日基线相同，确认 M1 新失败硬封锁和 room 几何签名没有残留。
- 测试保留 3 条已有环境警告：两条 Pydantic 旧配置样式弃用提示、一条本机 Objaverse 版本新于 benchmark 的提示。
- 完整日志：`/home/ldl/tmp/migrate22-tests/regression.log`。复测命令与监控命令见根目录 `test.md` 的 2026-09-23 章节。

没有运行新的 ROS 端到端 benchmark、双卡压力测试或 catkin 构建。因此本报告证明的是迁移范围和静态/单元合同，不证明 eval 提速、超时率降低、成功率提高或无性能下降。

## 5. 下一步实验建议

1. 以本分支作为后续优化起点；先固定模型服务、并发度、episode/seed、步数预算、录制开关、同类过滤与成功判定合同，不再同时更换服务拓扑。
2. 先做少量相同 episode 的 ROS smoke，再对相同 mixed 集合做成对比较。长时任务需另行确认。
3. 记录 M1 objects/rooms 各自的提交数、超时率、latency p50/p95、queue lag，以及对象额度耗尽数；不要把所有 M1 合并后猜测物体推理瓶颈。
4. 同时看 applied steps/墙钟秒、每 step 墙钟、bridge timeout/no-fresh-action、真实 SR、ISR/IP 和成功声明拒绝原因，区分“少提交”“更快执行”“更严格评分”。
5. 若 30 秒 M1 仍大量超时，优先降低 worker 并发或隔离 room/object 模型负载，检查后端 waiting/running 和显存；不要再先缩短超时或增加永久封锁。
6. 只有在相同过滤/判定/指标合同下重跑基线，才能把性能差异归因到算法与调度。既有 22/23 日结果可以作为诊断参考，不能代替这次迁移后的公平回归。
