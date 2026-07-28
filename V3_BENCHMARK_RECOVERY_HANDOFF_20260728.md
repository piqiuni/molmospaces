# V3 Benchmark 实验恢复交接

## 恢复边界

从 Codex 会话 `019fa67b-331c-7582-b38e-2824a3187fbf` 的第三轮实际问题分析之后恢复。

后续日志中出现的 `allow` / 审批输出是会话记录异常，不应视为任务结论或用户需求。以本文件、当前工作区和实际输出目录为准。

## 用户的最后有效任务

对 V3 Benchmark 的三个完整交互导航 episode 做 ROS 端分析与修复：

- 2010：Door -> Dresser drawer -> pencil
- 2730：Door -> Fridge -> apple
- 2664：Door -> Fridge door -> internal drawer -> potato

用户观察并要求解决的问题：

1. 2010 约 200 step 后不再移动，6-panel 图像看似崩溃。
2. 2730 穿门时 global planning 失败，虽仍在运行却没有有效导航；开门后可见冰箱，但模型没有关联目标物体与场景。
3. 门后的狭小区域未被房间分割识别；评估是否应将门后空间强制划分为独立 room。
4. 在模块 1 的显著地图更新后补充 room 属性推理；设计二阶段提示词，增强 room--object 语义关联。
5. 候选排序对新房间/大区域的偏好不足，导致在同一房间反复探索。
6. OCC 边缘重叠；核对 V3 eval 传感器输入与普通场景测试是否存在噪声或坐标差异。

## 已验证的基线现象

基线目录：

`scripts/InteractiveNav/output/v3_mllm_mixed_full_2010_2730_2664_20260728_baseline/`

- 三条 episode 都达到动态预算上限，均未成功。
- 这不是 ROS 相机 topic 或整个 ROS 节点崩溃：评测端持续发布 RGB，回调计数接近/达到 step 数。
- 6-panel 黑帧/placeholder 的主因是录制端渲染与同步配对积压；有限 RGB 缓存被淘汰，而不是相机停止发布。
- 2010 同时存在语义执行器在同房间反复选择失败 frontier，以及 `move_base` 无法获取机器人起始 pose 的记录。
- 2010：开门后 traverse 很快变为 `make_plan_unreachable`，未进入目标区域。
- 2730：开门后未能穿入目标房间，候选重复选择；后续缺少 fridge 交互链。
- 2664：存在不可达 portal 重复选择；虽有开门成功，未完成 fridge -> drawer 连续交互。

## 当前工作区状态

工作目录：`/home/user/ldl/molmospaces-exp-setting`

工作树有未提交的 V3 evaluator、语义决策、地图和录制相关修改；不要重置、checkout 或覆盖它们。

主要新增/修改入口：

- `scripts/InteractiveNav/evaluation/benchmark_runner.py`
- `scripts/InteractiveNav/evaluation/v3_round_summary.py`
- `scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_test.zsh`
- `Interactive-Nav-SG-nav/src/explore_py_pkg/scripts/record_explore_debug.py`
- `Interactive-Nav-SG-nav/src/explore_py_pkg/scripts/step_sync_image_cache.py`
- `Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_decision_py_pkg/post_interaction_traversal.py`
- `Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_decision_py_pkg/behavior_execution.py`
- `Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/semantic_mapping_py_pkg/interaction_graph_store.py`
- `Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/semantic_mapping_py_pkg/interaction_result_contract.py`

已报告的定向修复（均未提交）：

- 仅对开门后 `post_interaction_traversal` 的空路径增加有限时间重试，避免立即退回 frontier。
- Portal 候选扩展为门两侧、多 standoff 与切向偏移，并让目标 yaw 朝向门中心。
- V3 运行时默认不再逐 step 写 6 个 panel PNG/composite PNG；保留 MP4、稀疏语义关键帧和 topdown 所需数据。
- recorder 写入采用背压与排水校验，新增 `wait_for_recorder_drain.py`。

此前组合测试报告：`259 passed, 17 warnings`。这是历史结果，恢复后应先复核当前工作树的相关最小测试。

## 尚无可靠结论的事项

- 三 worker 最终组合 smoke（穿门等待、portal 备选、recorder drain）曾启动，但会话在取得最终汇总前截断；不要假设已通过。
- room 强制细分、模块 1 LLM room 属性推理、二阶段语义提示词、新房间偏好、OCC 坐标/传感器差异均未确认完成。

## 建议的恢复顺序

1. 只读检查当前 Git diff、最近 V3 smoke 输出和 ROS 日志，确认最后运行是否结束及其结果。
2. 运行最小相关单元测试，复核 recorder、post-interaction traversal、interaction graph 与 V3 evaluator。
3. 对 2010/2730/2664 分别重放或做短 smoke，按同一时间轴导出：step、robot pose、TF、global plan、costmap、portal 状态、room assignment、候选选择、image callback/placeholder 计数。
4. 将问题拆为独立修复：
   - recorder/sync；
   - TF 与开门后 costmap/规划；
   - room segmentation 与 OCC 坐标；
   - room--object 语义关联与候选排序。
5. 每项修复完成后先跑最小测试；经用户授权后再启动 3-worker 完整录像评测。

## 安全与运行约束

- 未经用户明确授权，不启动长时间仿真、全量 benchmark 或数据采集。
- 保留现有未提交变更；不执行 reset、checkout、清理输出目录或覆盖数据。
- 运行 ROS 多 worker 时使用隔离 master/ROS_HOME，避免与其他会话冲突。
- 用户要求每次完整测试保留 6-panel 视频和俯视结果图。
