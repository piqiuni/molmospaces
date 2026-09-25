# exp-setting → Go2 实物合并记录（2026-09-25）

合并目标：`codex/physical-go2-nav`；来源：`origin/codex/exp-setting` 的
`2d159711`。合并前存档为 `6d5d07f6`，不是直接覆盖实物代码或切换到仿真配置。
采用三方合并，逐项处理算法与实物接口冲突；原有实物回归和上游新增回归同时保留。

## 纳入与保留

- 纳入新版 M2 公共上下文、候选池、房间归属、失败区域记忆和导航恢复判定。
  关闭旧的强制未进入房间选择，让 M2 使用新版排序。
- 纳入新版 M1 精确图像配对、最多两视图证据、调用预算和门状态确认。
  保留实物 `observed_object_name` 名称纠正、持久化、去重、旋转 OBB 和两帧观测门槛。
- 保留实物 M3 `external_mllm_verified`、episode/command/event 身份隔离、
  M2 `target_revision` 旧结果隔离，以及取消/发送零速完成后才能切换下一个目标的保护。
- 保留原始传感器 ROS 数据链、采集时刻 TF、latest-only 点云/OCC、房间异步线程、
  原始 OCC 到全局代价地图的开门后快速通路。网页仍是可选观察者/命令入口。
- 实物门接近点仍按稳定 OBB 的法向/中心线生成；冰箱距离和安全速度参数不变。
  **DWA 仍是默认控制器**；新版 PathFollower 已编译，但没有启用。
- 房间合并确认使用新版规则，同时保留实物 NumPy 向量化重叠统计和拓扑缓存。
  增加 ROS Noetic 对新版 NumPy 的批量栅格序列化兼容，不转逐格 Python 列表。

## 实物时间与证据适配

实际启动入口继续读取 `scripts/InteractiveNav/physical_nav/config/` 下的配置。

- `candidate.progress_clock: monotonic`，`progress_clock_period_s: 0.2`：
  导航/候选恢复预算按独立的 0.2 秒单调时钟刻度计算；例如 60 刻度为 12 秒。
  **不是相机帧数或消息数量**。上下文中的 `source_capture_step` 保留真实采集序号，
  `observation_step_clock` 明确标记时钟来源。仿真默认仍使用 evaluator capture step。
- 执行器继续使用无 step-sync 时的墙钟超时及实物 odom 闭环。
  新版依赖仿真确认的自动倒退恢复在实物配置中关闭，不能无确认地向 Go2 下发。
- 门状态沿用实物两帧检测后一次有效 M1 确认；稳定状态冷却明确为 120 秒，
  不受相机 FPS、序号重置或图像卡顿影响。物体身份持久化策略不变。
- YOLO 的报告携带同一次采集的位姿、图像尺寸和时间戳，mapper 原样转发给 M1。
  没有采集位姿时不虚构零位姿；不把 ROS 各 topic 独立的 Header.seq 当成同一帧。
  旧浮点时间戳仅允许浮点舍入误差，不放宽为相邻帧时间窗口。
- 房间 M1 全局发送上限仍为 1 Hz，YOLO 配置仍为 10 Hz；这些是配置值，
  不是本次实机测得的实际频率。

## 验证与边界

离线回归入口见根目录 `test.md` 的本次合并验收章节。覆盖决策、建图、M1、
探索、实物数据桥与运动开关；不调用真实模型、不连接 Go2、不启动运动。
本次最终结果：1942 项通过，4 条 ROS 依赖弃用警告，无失败。

已构建 `slam_gmapping`、`oriented_global_planner` 和 `path_follower`。
通过 `roslaunch --dump-params` 检查实际脚本配置展开，确认 DWA、实物 M3、
门法向接近、单调时钟、M1 限频和速度限制仍生效。

本次没有重启服务、开启运动或推送远端。运行中的 Python 服务尚未加载本次改动。
离线通过不代表实物成功率或端到端延迟已得到验证；后续应先只读启动检查图像、
点云、OCC、M1/M2 和 TF，再经操作者确认进行导航/交互验收。

子仓库保留合并前存档：navigation `5cff6868`，论文源码 `dcecb88`。
本地 PDF、perf.data、脚本备份及临时测试产物不纳入此次提交。
