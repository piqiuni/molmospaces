# Physical Go2 性能优化分阶段方案

本文档记录性能大优化前的基线、分层边界和验收指标。Go2 不在线时只做
离线代码/回放测试；恢复连接后按同一组指标做实测，避免把网页刷新问题误判
成传感器或建图问题。

## 目标和数据边界

原始传感器链路必须在网页关闭时仍能运行：

```text
D435i/Go2 -> latest-only WebSocket -> physical_sensor_ros_bridge
          -> ROS RGB/depth/CameraInfo/telemetry/PointCloud2
             + capture_context (capture stamp -> selected telemetry/calibration)
          -> YOLOE -> detections -> semantic mapping/OCC/navigation

                         (可选观察支路)
          -> latest-only HTTP mirror -> six-panel web/Foxglove
```

网页只消费镜像和 ROS 状态，不得成为 RGB-D、YOLO、OCC 或控制链路的前置
条件。每个高频阶段都只保留最新帧；旧帧等待队列属于延迟而不是有效数据。

## 当前基线（最近一次真实日志）

| 阶段 | 当前测量 | 主要原因 |
| --- | ---: | --- |
| RGB/depth 输入 | 目标 10 Hz | Go2 发送端和 ROS bridge 已是容量 1 的 latest-only |
| YOLO GPU 推理 | 15--19 ms | 不是主要瓶颈 |
| YOLO 后处理 | 128--207 ms/帧 | 每个实例都做全深度网格投影、连通域、点云和 OBB |
| YOLO 总周期 | 3.8--5.5 Hz | 后处理完成后才进入下一帧 |
| gateway CPU | 约 40%（无新帧时仍高） | 相机 overlay/面板固定定时渲染、旧 report 仍触发状态构建 |
| semantic publish timer | 约 10 Hz 空转 | 没有输入 revision 时仍重建 graph/markers |

上述日志来自 `/tmp/molmospaces-physical-nav-1000/logs/yoloe.log`；Go2 断开时
不把冻结的 `frame_seq` 当作算法性能数据。

下面的数字是本机合成/回放基准，不是 Go2 在线频率；硬件恢复后仍需用同一
组 `rostopic hz/delay` 和组件 timing 日志复核。

## 分阶段实施

### P1：感知实时性和确定性（当前阶段）

1. ROS YOLO worker 直接使用 callback 已解码的 NumPy RGB-D，保留 HTTP/录制
   回放的编码格式作为兼容回退。
2. RGB-D 按时间戳配对，输入和网页镜像均为 latest-only；HTTP 镜像在独立
   线程发送，不阻塞 GPU/后处理。
3. 通过 `/physical_nav/telemetry` 发布带时间戳的最新遥测，YOLO 用有界历史
   选择采集时刻最近位姿，3-D 点和 3-D box 不再依赖网页状态。
4. mask->depth 只扫描检测框的保守 ROI，连通域只在 mask 的非零包围盒内执行；
   `max_geometry_instances` 按交互类别优先限制昂贵的 3-D lifting。
   对非零基线的 D435i 对齐投影，先按本帧所有检测框求保守深度 ROI，再在
   ROI 内计算深度相关投影；输出仍保持完整栅格形状，像素结果与全图投影一致。
   纯 RViz/Foxglove 的分割点云只有在配置打开且存在订阅者时才解码/发布。
5. 目标：RGB/depth/points 9--10 Hz，YOLO report >=9 Hz，检测端到端延迟
   p95 <150 ms；报告中记录 decode、GPU、mask projection、geometry、overlay
   各阶段耗时。

当前离线收尾结果：3-D lifting 已拆成无状态候选任务，持久 4-worker 池在
合成 12 个大候选上约从 40.3 ms 降至 20.7 ms，且串行/并行的 mask、点云、
世界坐标、3-D box 与 OBB 朝向逐字段一致；候选 FIFO、上限和 geometry budget
仍在主线程控制。动态 M1 图 payload 不再重复递归复制，静态语义消融仍保留
深拷贝隔离。

本轮离线收尾还做了三项低风险优化/纠错：网关把同一 receipt 的 packed
segment 点云解码结果共享给 3-D 映射和可视化支路，并且稳定 box track 不再
复制 mask/点云大字段；同分辨率的 RGB/depth 在未明确 `align` 时仍执行标定
投影，避免 `align_to=none` 被尺寸判断误当成已对齐；房间分割的可选封闭障碍
填充兼容 list/ndarray 返回值，且 topology 输入缓存包含
`room_free_threshold`，运行时阈值变化不会复用旧分类。
感知帧内的几何配置也改为一次构造、候选共享，避免每个 3-D 候选复制整份
YAML 配置。

### P2：建图和房间分割隔离

1. GMapping 输入只接收最新 PointCloud2，使用点云采集时刻对应的 odom；
   不允许 HTTP 或可视化回调同步等待。
2. OCC 更新、OCC 发布、room segment 计算、room-grid 镜像拆成独立的
   latest-only worker。room segment 的计算频率与发布频率分离，不能阻塞
   GMapping。
3. 只有 occupancy revision 改变时才运行增量房间分割；噪点先做局部时空
   滤波，再做连通域/门洞切分，未知区不直接伪造为房间。
   语义图对同一 room-grid epoch 内未改变几何的门复用两侧房间采样结果，
   room-grid 内容或合并发生变化时自动失效。
4. 目标：点云输入与 GMapping 延迟 p95 <120 ms，OCC 发布 >=5 Hz（有新图时），
   room segment 独立运行且不降低 OCC 频率。

### P3：稳定物体、M1/M2 与指令链路

1. 物体跟踪的重复 box 合并先按 label/alias 分组；语义图的 track 关联使用
   label+空间网格索引，避免无稳定 ID 时对全部历史节点线性扫描。
2. 物体状态机区分 `tentative/confirmed/persistent/hidden`。门、容器和房间
   在“两帧观测 + M1 确认”后持久化，不因短暂漏检过期；普通噪声仍可回收。
3. M1/M2 使用去重后的 latest-only 队列和目标级冷却；状态确定后不重复调用，
   目标属性变化才重新排队。M1 普通目标在整帧视觉证据合成前先通过对象级
   reservation/cooldown；已 pending/完成的目标不再重复分配整帧证据图，已接受
   请求共享同一份历史 RGB 缓冲。
4. Goal 注入、subgoal 生成、路径/碰撞验证、到达停止接口使用明确的状态机，
   watchdog 只监测健康和超时，不重置业务状态。
5. 目标：同一目标不产生重复 track/subgoal；无旧检测排队；导航到达后一次性
   发布 stop/terminal 状态。

当前离线收尾结果：语义对象 worker 与 YOLO 输入均为容量 1 的 latest-only；
重复轨迹合并先做标签族早筛，再计算 2-D/3-D 几何；跟踪镜像的 JSON 压缩与
ROS 序列化移出 mapper 锁，避免阻塞 OCC 回调。

本轮离线性能收尾又加入了三项语义图优化：

- `ObjectMapStore` 对稳定 `instance_id` 使用身份索引；无 ID 的检测使用
  XY 邻域桶，并为 portal 保留跨视角候选。候选仍按原插入顺序送入历史匹配
  条件，避免改变跨标签 AABB 合并或 portal 规则。离线 1,200 条轨迹/400 次
  查询基准由约 212.25 ms 降到 2.80 ms（约 75.8 倍）；该数字是本机合成
  `_find_match` 基准，不代表在线传感器频率。
- `InteractionGraphStore` 对稳定 `instance_id` 使用批次身份索引；没有稳定
  ID 的检测使用 `type + label + XY` 邻域桶，避免首批/换视角时对所有历史
  节点做二次线性扫描。合成 2,500 个远离目标的首批关联由约 1.9 s 降至
  约 0.24 s，近邻合并行为保持不变。
- 房间父子关系按 `room_id` 分桶，并对候选容器/支撑面和目标几何做上下文
  缓存；候选几何、房间、可见性或交互状态变化时自动失效。合成 1,000
  目标/200 候选的重复帧由约 0.35 s 降至约 0.10 s；开放冰箱的深度侧推断
  保持走未缓存路径。
- 房间网格首次统计直接把已计算的中心/尺寸传给房间节点，跳过默认几何的
  两次全图扫描；1984×1984 单房间离线首帧约 0.64 s，后续同内容心跳约
  0.07 s。检测帧复用房间统计但不再增加 room-grid 观测计数，避免 10 Hz
  YOLO 提前满足“两帧房间网格”门槛。
- 关系重建中重复出现的 refrigerator/label marker 判别增加 4,096 项有界
  规范化缓存；1,500 个合成目标的 `update_observations` profile 从约 0.58 s
  下降到约 0.43 s，判别结果逐项保持一致。该缓存只影响字符串谓词，不改变
  目标关联、房间或交互状态。

网页镜像支路的离线运行态审计还发现：ROS 侧 `unified_graph` 约 10 Hz、
`explore/status` 约 5 Hz，即使没有新 RGB-D 帧也会持续被可选网关解析并通过
HTTP 转发，旧进程因此出现约 40% CPU 占用。该支路现在保持 ROS 原始发布频率
不变，只对展示镜像做 latest-only 限频：graph/consistency 为 1 Hz，探索与
候选/决策 trace 的最小间隔为 0.5 s，选择/执行/反馈为 0.2 s；事件类结果和
telemetry/YOLO 仍按原路径转发。小状态
发送线程的唤醒上限由 0.5 s 降为 0.05 s，大 OccupancyGrid 仍使用独立的 0.5 s
队列。这样网页状态不会拖慢原始 ROS 链路，也不会把高频语义心跳当成网页刷新率。

GMapping 的物理 launch 仍开启了局部观测覆盖（半径 7.9 m、0.5° 伪激光）。
该层原先对每个波束的每个采样点重复调用 `worldToMap`，一帧约产生数万次
浮点除法和边界检查。现在先把传感器位置换算到栅格坐标，再在栅格坐标中完成
射线采样；占据结果、无回波清空规则和 TTL 语义不变。同时把局部覆盖单独加入
GMapping pipeline telemetry，下一次在线运行会在 `local_overwrite=avg/max` 中
给出该阶段的实际耗时，能够与 `projection/addScan/updateMap/map_publish` 直接
对比，而无需猜测 OCC 延迟来源。

网页朝向链路也已改成姿态独立的 latest-only 更新：capture-time telemetry 的
`yaw`/IMU 姿态优先于旧 `map_yaw`，带 transport generation 和 capture sequence
的姿态不会被无序 odom 回显覆盖；`telemetry_revision` 单独使地图箭头失效重绘，
而不触发相机/整图重复编码。当前在线网页仍由旧进程提供，且最后一帧已停在约
21 小时前，故必须在下次授权重启后再以真实转向验证箭头。

### P4：M3 交互评判和可观测性

1. M3 只在交互状态转移或验证窗口触发；采样和结果写入独立队列。
2. M3 结果用版本号更新 semantic graph，成功/失败/超时均是终态，不能被下一
   帧 YOLO 覆盖。
3. 网页/Foxglove 仅订阅低频状态、revision 和可选 debug cloud；原始 ROS 数据
   在没有网页时保持完全相同。

## 验收命令

在线恢复后：

```bash
rostopic hz /physical_nav/rgb/image_raw /physical_nav/depth/image_raw \
  /physical_nav/points /physical_nav/yolo_report /physical_nav/detections \
  /physical_nav/occupancy
rostopic delay /physical_nav/rgb/image_raw /physical_nav/depth/image_raw \
  /physical_nav/points /physical_nav/occupancy
pidstat -p <yolo_pid>,<gateway_pid>,<sensor_bridge_pid>,<gmapping_pid> -u -w 1
```

离线可先运行：

```bash
python3 -m py_compile scripts/InteractiveNav/physical_nav/physical_*.py
python3 -m unittest \
  scripts/InteractiveNav/physical_nav/tests/test_physical_raw_recorder.py
```

每个阶段完成后单独提交，提交信息必须包含实测的输入频率、p50/p95 延迟、
CPU/GPU 和丢帧数；不要以网页 FPS 代替原始 ROS 频率。

## 本机离线验收记录（2026-09-09）

- 未连接、未 SSH、未重启 Go2；以下均为本机合成/回放或静态构建。
- 物理端、房间缓存、语义规划和 OCC 相关针对性回归：`174 passed, 4
  warnings`。
- ObjectMapStore/语义父关系针对性回归：`15 passed`。
- `slam_gmapping` 与 `voronoi_mapping_node` CMake 构建通过。
- 最新物理回归（含姿态竞态、网关小状态限频、原生面板 204 增量请求）：`140
  passed, 2 warnings`；GMapping odometry-lock 快路径静态回归 `1 passed`。
- 当前本机运行态只读核对显示最后一帧约 21 小时前，`/physical_nav/odom` 与
  `/physical_nav/points` 暂无新消息；因此网页箭头静止不能作为新代码的在线
  性能结论。现有运行进程仍是修改前启动的旧网关，下一次授权重启后才会加载
  raw-sensor 单写者和面板 revision 逻辑。启动脚本的持久网关指纹已修正为
  `${ROOT_DIR}/../offline_semantic_renderer.py` 的真实路径，避免缺失文件导致
  指纹计算报错或旧网关被错误复用。
- 完整 semantic-mapping 测试当前为 `225 passed, 21 failed`；失败集中在
  工作树中已有的 M1 请求 schema/多视角断言和旧 portal 状态断言，本轮索引
  与缓存新增测试均通过，未把这些既有失败当作在线性能结论。
- 归档基线仍是 Git commit
  `7db7561528e2ca7309447a8134f246aad63aa4e`（性能大优化前存档）；本轮优化
  代码保持未提交，便于恢复 Go2 后按在线频率/延迟指标单独验收。

### 后续网页与状态转发审计（2026-09-09）

- 发现此前将 `video` 赋为空函数并未取消 `setInterval` 保存的旧函数引用。
  现在在生成原生面板页面时移除旧整图定时器及首次请求，同时移除重复的
  raw OCC/graph 轮询。原生房间图已绘制 global path；撤掉按全图坐标重复
  叠加的浏览器路径，避免与裁剪后的房间图错位。
- panel 3/6 分别维护已成功显示的 revision；HTTP 请求跨越渲染时刻或单张
  图片解码失败时，另一张仍能显示，失败者下次重试，不提前确认未显示的数据。
- ROS graph/decision 等 JSON 回调只替换待发送字符串，由单一后台线程限频、
  解析并入发送队列。消除回调即时发送与旧缓存延后发送之间的状态回退竞态。
- 针对性验证 `104 passed, 2 warnings`。其中 Node.js 执行真实轮询脚本，
  覆盖旧定时器去除、独立 revision、204、解码失败重试及图片资源释放；
  Python 回归覆盖限频边界新状态覆盖旧缓存。未连接或重启 Go2。
- 网页相机检查仍为 10 Hz，原生地图面板检查为 5 Hz，状态卡为 1 Hz；
  这些是配置上限，不是实机吞吐实测。下一次在线测试仍需核对 ROS 帧龄、
  OCC 各阶段耗时和真实转向时的箭头更新。

### 候选碰撞与批量可达性（2026-09-09）

- 查实 `candidate_occupancy_use_astar` 此前未写入实物覆盖配置，运行时默认
  为 false，实际只有直线采样检查。现在实物配置显式启用本地栅格寻路。
- 不再把空闲中心线当作机器人可通过的证明；开启栅格寻路时，每个有效端点
  都检查整段路径的圆形机器人半径（沿用 0.25 m），能剔除中心线空闲但机身
  无法通过的窄门，同时保留需要绕墙的可达候选。
- 一批候选复用同一 OCC receipt 的膨胀碰撞缓存和起点 Dijkstra 波前。
  单独调用仍使用 A*；共享上下文时不依赖目标启发式，已确定的最短路可用于
  所有 Anchor。12,000 个扩展上限约束整批同起点搜索，耗尽预算返回
  `search_limit`，不伪称无路径；执行层最终碰撞检查仍保留。
- 缓存只活在一个候选批次内，下一地图或参数不能误用上一批碰撞结论。
  严格禁止未知时，修复对角线从未知格角落穿过的漏洞；成功路径的 unknown
  标志根据实际路径及对角侧格判定，不再根据所有曾搜索的邻居判定。
- 本机合成 120×100、0.1 m 栅格、20 个绕墙 Anchor：独立 A* 约 852.70 ms，
  批量共享搜索约 115.52 ms（约 7.4 倍）；累计扩展从 58,415 降到 9,230。
  两者可达性和路径长度一致。这是离线基准，不是实机 OCC 或导航频率。
- 22 项针对性回归通过：窄门拒绝/扩宽后通过、绕墙、旋转地图、未知格角落、
  新地图缓存隔离、跨目标续搜、整批预算和随机地图上的 A* 路径代价对照。
  尚待实机核对当前半径/余量是否符合 Go2 实际占用及地图噪声。

### Goal/M2 跨任务结果隔离（2026-09-09）

- 候选生成快照包含独立 `target_revision` 与相同版本的 target context；
  生成期间若用户指令变化，丢弃这一批旧候选，下一 tick 重建。相同内容重发
  不增加版本，A→B→A 则保留不同版本，避免旧请求误匹配重新下达的任务。
- M2 在分析前、模型返回后和下发提交时校验任务身份（episode、目标版本、
  目标内容），不把旧目标的回答套用到新目标的候选上。丢弃原因通过
  `stale_mission_discarded` trace 暴露；模型输入与 selection 保留目标归属。
- 旧目标的终态反馈释放执行槽，但不能把新目标标为成功、计入新任务完成
  判定或生成旧门后的继续导航。目标切换也清空上一任务的待续行状态及
  完成/无路确认计数，地图与持久物体不受影响。
- 49 项针对性回归通过（目标版本、真实 M2 决策方法中的模型调用期间切换、
  旧目标成功回执、完成状态机和候选检查）。本轮仍未连接/重启 Go2。
- 这不等于整个 Goal 控制链已验收：指令到候选节点的时延、正在执行动作的
  安全抢占/取消确认、到达后物理 stop 持续性和 watchdog 所有权还需继续
  独立检查。当前修复首先保证候选/模型结果/终态反馈不会跨任务串用。

### 终态停止与速度复用器（2026-09-09）

- 找到真实状态机流程中的漏停：`BehaviorExecutionStateMachine._finish()`
  已切至 SUCCEEDED/FAILED，但 `_finish_terminal()` 只在 NAVIGATING 时
  发送取消/零速度。现在按持有的动作类型处理 NAVIGATE/EXPLORE/INTERACT/
  SCAN 的终态停止，测试通过真实 start→navigation result→terminal 流程。
- 停止发送完成之前保留执行器 busy 状态，避免新动作先开始又被旧动作停止。
  terminal 命令核对 decision/candidate 身份，重复和迟到命令不影响继任者。
  cancel 发送失败仍尝试零速度；停止发送失败不能向决策层报任务成功，原动作
  结果与停止通道错误分别保存在反馈 detail。
- 普通导航通常没有 semantic 速度租约，原 mux 会忽略其终态零速度。新增
  非 latch 的 String `base_stop` 通道（内容 `stop`），实物配置和 mux 均指向
  `/physical_nav/base_stop`。显式停止保持 0.35 s，清除旧速度缓存并丢弃保持
  期间的两路速度；结束后仅接受新指令。普通空闲 semantic 零值仍不抢占导航。
  mux 的实际发布与 stop 接收共用锁，避免先取到非零旧值、收到 stop 后才发出。
- 98 项执行器/速度 mux 离线回归通过；实物 YAML/XML 的停止通道匹配校验通过。
  未启动 ROS 或连接 Go2。Web health watchdog 的当前默认启动带
  `--keep-running`，且 headless 禁用这条检查；该 watchdog 只记录健康问题，
  不直接发布速度或替换 subgoal。其他业务 watchdog 尚需逐条审计。
- 当前证明发送与软件仲裁顺序，未证明物理车身已静止。跨任务主动抢占、
  actionlib 取消确认、控制器静止确认与真实停车距离仍是后续验收项。

### 噪声 OCC 的房间分割热点（2026-09-09）

- 未知孔洞、封闭家具和小障碍处理原先每个连通块都执行一次全图
  `labels == component_id`，噪点越多重复扫描越多。改为按连通块统计包围框
  局部索引；核心索引也按局部范围生成。两种障碍过滤共享同一次 occupied
  连通域标记，输入原始 OCC 不被修改。
- 修复无核心时的拓扑错误：多个窄小自由连通域不再合并成一个人工房间。
  兜底按四连通分块，并保留原面积门槛与稳定 ID 映射。
- 本机 640×640 合成图含 2,704 个未知小孔及大量独立占据点，完整 segment
  耗时约 1,291.75→343.43 ms（约 3.8 倍）。前后 room IDs 和置信度的
  SHA-256 均为 `0f8adfc42228ab9e7a260bcbb1635d6257c4ad380b662b1657d5ff079b92b3b0`。
  这是固定输入的离线对照，不能替代真实断墙/地图抖动的质量验收。
- 房间分割 18 项测试通过，包括开放未知前沿保留、众多孔洞、局部索引
  正确性、狭窄非连通房间以及重复观测 ID 稳定。本轮未改变未知填补、
  断墙闭合或核心半径参数；真实大地图+GT 标注仍用于后续质量调参。
- 配置审计另发现 `physical_nav.yaml` 的 `room_inference.backend` 仍为
  `object_rules`。用户要求的纯文本房间属性 M1 不能仅以这项规则配置宣称
  完成，下一步需核对实际属性推断节点及其异步/缓存入口。

### 后续热点复测与 M1 房间入口核对（2026-09-09）

- 房间分割前述修改补跑房间刷新、图缓存联合回归：42 项通过。继续剖析发现，
  单房间连通域在种子置信度一致时仍执行最近种子的距离变换和查找表分配。
  现在先验证整个连通域只有一个 room ID，再对一致置信度直接赋值；原有种子
  分数不变，混合置信度仍走原距离变换，多房间连通域仍走拓扑约束扩展。
- 同一 640×640 噪声地图，预热后交替执行旧/新传播实现各五次，完整分割中位数
  185.21→169.20 ms（再减少约 8.6%）；所有输出的 SHA-256 与上一轮相同。
  这里是本轮同进程对照，不能将此前冷启动 343.43 ms 与本轮预热结果直接
  相减作为新增优化收益。其余热点主要是每个噪声连通块的边界环统计。
- 新增均匀种子置信度 40/60/70/100、混合置信度回退、多房间不提前修改输入
  的回归；房间分割/后台刷新/房间图缓存联合测试为 48 passed，4 个 ROS
  依赖弃用警告。未启动 ROS 或连接、重启 Go2，未提交优化代码。
- 进一步确认上一段的配置疑点：`object_rules` 并不代表没有 M1 房间通道。
  `semantic_mapping_node._build_room_attribute_request_locked` 从房间成员构造
  纯文本证据，`interaction_attribute_inference_node._room_worker_loop` 使用
  独立队列，`_infer_room` 调用模型且不携带图像。实物配置开启 room_mllm，
  当前模式为 dynamic_mllm；本轮修正误导性的配置注释，未改变推断开关。
- `room_mllm.min_interval_s: 5.0` 当前是每房间完成后的间隔，并非整个模型
  通道的全局 1 Hz 限制；证据变化还会使在途请求失效。全局调用限频、持续
  证据变化时的公平调度和在途结果被丢弃比例仍需继续审计，不以注释代替保证。

### M1 房间共享调度预算与冷却（2026-09-09）

- 实物增加 `room_mllm.dispatch_interval_s: 1.0`，所有房间 worker 共享同一个
  monotonic 调度窗口；通用配置默认 0，其他实验不被隐式降频。等待发生在
  房间 worker，期间不持有节点锁，不阻塞 RGB、物体 M1 或 OCC 回调。
- 调度等待最多每 0.1 s 重新检查请求身份，可被 shutdown 唤醒。等待期间
  证据替换或 episode 改变则丢弃；等待计入原请求 deadline，超时不调用
  模型、不占新窗口。模型指标及成功 patch 新增 `dispatch_wait_sec`。
- 修复证据变化清掉 `room_last_request` 的问题：请求取得调度窗口时记录
  冷却起点；在途证据变化仍使旧结果失效，但不能抹掉已花费的模型调用成本。
  保留原有每房间 5 s 冷却和相同成功证据缓存。
- 21 项房间调度/房间属性/队列测试通过，其中真实 Python 三线程共享调度
  窗口；假模型直接执行 `_infer_room`，覆盖纯文本、请求替换、截止期限、
  任务切换、关闭、中途证据变化和成功缓存。YAML 解析与 diff check 通过。
- 扩展到 `test_attribute_inference_request_state.py` 的结果是 36 passed、
  3 failed：多视角期望 [10,34,50] 与实际 [20,34,50] 不同，以及两项物体
  请求上下文多出 detector_class。对应 `_infer` 和
  `_select_diverse_target_visual_history` AST 与存档 HEAD 完全相同；本轮
  未修改这些物体路径，后续需根据所需物体确认契约检查，不能宣称整个 M1
  套件全绿或直接删除断言。
- 未连接 Go2、未调用在线模型、未提交/重启。此处限的是软件调度窗口，不是
  相机或 OCC 频率。已发送 HTTP 请求不能靠这一层取消，持续变化证据下的
  响应作废率、队列公平性与模型服务竞争仍待进一步验证。

### 物体 M1 证据与类别纠正契约（2026-09-09）

- 查实提示词冲突：同一请求既要求 M1 纠正 detector_class，又要求 locker
  “若不是冰箱，保留 locker 语义”。删除后者，保留饮水机/冰箱的视觉区分
  要求；模型可返回真实名称和 none/portal/container，而不是由规划器覆盖。
- 新增 `attribute_inference.include_detector_class_hypothesis`：通用配置
  false，防止默认将仿真/GT category 注入图像判断；实物配置 true，把 YOLO
  类别明确作为可被纠正的假设。物体 ID、位姿、私有类别仍不进入默认模型
  context，实物 semantic_class 优先于其他元数据。配置 YAML 解析通过。
- `_is_portal_detection` 补充 semantic_class/raw_class：原始实物门检测即使
  没有 semantic_name，也使用完整门框上下文而非容器式 inset 证据。
- 多视角选择维持“最新有效视角优先、输出按采集顺序”原则。增加采集序号
  排序和有限数姿态检查，迟到帧不再被误当成最新，NaN/Inf/无效姿态不能
  通过间距判断。未改变图像数量和视角间隔阈值。
- 上轮三项失败按契约逐项处理：默认不暴露 category 是代码修复；多视角
  [20,34,50] 是正确的新帧优先；旧测试强制把 portal 改回 container 与用户
  “M1 判定后全部按新属性处理”冲突，改为验证 planner 假设仅作诊断、M1
  门类别及 aperture 证据完整保留。没有简单删除失败断言。
- 物体 M1 请求状态/过滤、房间调度/属性及队列共 45 项通过。新增测试将真实
  `_infer` 输出的 water_dispenser/none patch 应用到 InteractionGraphStore，
  再输入 locker 观测，确认名称保持 water_dispenser、类型保持 object。
  另两项既有确认物体持久化回归通过。未测试在线模型准确率或实机导航。
- 此轮未连接、重启 Go2 或提交代码。存档仍为 7db75615。

### 新 Goal 抢占与终态停止复用（2026-09-09）

- 查实旧 `_preempt_callback` 在 cancel_goal 之前清空 selection，且没有发送
  zero/stop hold。这与之前修复的自然终态路径不同，允许继任动作先启动再被
  前任取消。现在抢占复用 `_finish_terminal` 的 busy fence 和停止发送流程，
  正常取消回报 CANCELED，停止发送失败回报 FAILED；重复请求和错误候选被忽略。
- 同 episode 内新 Goal 版本到达候选回调后，决策层向当前执行器发一次
  mission_changed 请求，保留 active decision 直到反馈，不再仅靠等待原动作
  自然结束。现有任务版本隔离仍防止旧动作结果完成新 Goal。
- 新 Goal 可停止 NAVIGATE/EXPLORE/SCAN，以及尚在接近阶段的 INTERACT。
  已发出的外部交互没有确认式取消接口，因此不强行释放其所有权；后续自然
  终态由任务版本过滤处理。cleanup 提交处再次检查 INTERACT 状态，覆盖
  验证与停止之间从接近转为交互的竞态。
- 停止完成前撤销私有导航 run token；探索 finalize 也发生在 busy fence 内，
  避免其命令越过继任者。finalize 发布异常记录在反馈，不悄悄报成功取消。
- 112 项 Goal/执行器/速度 mux 回归通过，4 个 ROS 依赖弃用警告。覆盖取消
  中插入新 selection、取消发送失败仍发零、重复抢占、已发交互不取消、
  接近阶段在提交前切换交互、新 Goal 只发一次请求且不提前释放执行槽。
- 未连接/重启 Go2、未提交。这仍是软件停止发送顺序验证；actionlib 的服务器
  取消确认、物理停车、新 episode 重置分支仍需进一步检查，不能据此宣称
  所有 watchdog/抢占竞态或跨 episode 行为都已解决。

### 跨 episode 执行所有权（2026-09-09）

- 查实 episode 切换分支直接清空 active decision/candidate，造成旧执行器
  尚在导航时决策层已认为空闲。现在只清理新任务的历史/缓存/完成统计，
  保留旧执行器身份；通过上一轮 mission_changed 请求停止并等待终态反馈。
- 反馈的 episode 校验改为优先使用该执行动作绑定的 mission token，而非
  最新候选的 episode。这样旧交互的正确终态能释放旧执行槽，但不会给新
  任务记成功或生成旧门后的续行。错误 episode 的带戳反馈对导航也拒绝；
  跨任务释放还必须显式匹配 decision_id，缺 ID 的旧消息不能解除忙状态。
- 扩展回归覆盖仅 Goal 版本变化、仅 episode 变化、两者同时变化，分别测试
  空闲/运行中；还覆盖 NAVIGATE/INTERACT 的旧任务反馈、错误 episode、缺失
  或错误 decision ID，以及重复终态只释放一次。Goal/step-ready/执行器/
  速度 mux 共 129 passed、4 个 ROS 依赖弃用警告，diff check 通过。
- 后续只读检查确认决策节点剩余清空 active ID 的位置均在初始化或终态
  处理。执行器已有同 decision 的 successor quiescence 等待，但跨 decision
  停止发送到服务器确认之间的安全边界仍需审计，不能把软件取消发送等同于
  控制器已取消或实体 Go2 已静止。
- 未连接 Go2、未启动/重启服务、未提交。整项目实机频率、控制器取消确认、
  抖动地图质量和在线模型效果仍未验收。

### M3 迟到结果与后置状态（2026-09-09）

- 查实 VisualMLLMVerifier 只在模型调用前检查 cancel/deadline；同步 HTTP
  返回后，即使任务已取消或超时，仍可能沿 stable-open 路径返回成功。现在
  模型返回后再次检查，取消回报 CANCELLED，超时回报 TIMEOUT 并记录
  `m3_response_after_deadline`，不发 SUCCEEDED 事件。
- 验证器原来虽然接受 expected_state，实际只判断 open/ajar。现在依据
  open/closed/ajar 后置条件判断；open 仍沿用允许 ajar 的既有策略，但
  post_state 保留实际观测，不把 ajar 伪称完全打开。关闭任务不能被 open
  或 ajar 判为成功。不支持的后置条件明确失败。
- 非有限、超出 [0,1] 或无法解析的 confidence 作为不可信证据处理，即使
  配置 min_confidence=0 也不能成功，避免 Infinity 或异常 JSON 值误判。
- 新增假时钟/假模型测试复现第二次本可成功的回答在 cancel/deadline 后
  返回；覆盖关闭、打开、半开以及异常置信度。M3 新帧与 physical platform
  联合回归 98 passed、2 个 ROS 依赖弃用警告，diff check 通过。
- 仍未中断已发出的 HTTP 请求，当前只是结果有效性保护；模型请求的预算
  向底层超时参数传递仍需继续检查。Graph 命令匹配代码另有显式 target
  metadata 与原命令冲突时的处理问题待审计，未据此宣称整个 M3 回写已验收。
- 未连接 Go2、未调用在线模型、未重启或提交代码。

### M3 结果与原命令的身份绑定（2026-09-09）

- 查实 take_pending_interaction_command 只按 command ID/episode 匹配，随后
  merge 允许结果中的 node/object/action 覆盖原命令。现在取出 pending 前
  检查双方都携带的 episode、decision、candidate、node、object、action；
  冲突抛出明确拒绝，由 mapper 回调捕获，原 pending 保留等待正确结果。
  result event_id 可与 command event_id 不同，仍兼容生产者的独立事件编号。
- merge 补齐省略的 episode_id，并保留相同校验，避免以后直接调用 merge
  绕过身份边界。结果中的 success/post_state 等观测结论仍来自结果本身。
- 实物启用 semantic_map.require_interaction_command_id，缺 ID 的结果
  不再按物体名称猜测归属；通用默认保留无 ID 仿真适配器。实物发布端正常
  和异常路径均通过 _result_identity 写入 command_id，代码入口已核对。
- 命令与结果回调在当前 graph episode 不匹配时直接返回；旧开门命令不能
  给新地图设置 pending-clear。匹配命令补齐 episode 后也再次校验。
- 命令契约、mapper 房间刷新和 OCC overlay 联合回归 51 passed、4 个 ROS
  依赖弃用警告；覆盖目标/动作/episode 冲突保留 pending、缺失/未知 ID 无
  图和 overlay 副作用、正确结果后续可处理、重复结果只生效一次、旧命令
  不能设置 pending。实物 YAML 开关解析与 diff check 通过。
- 未连接或重启 Go2、未提交。真实发布者身份字段一致性、跨 ROS topic 的
  命令/结果到达顺序仍待实测；M3 图像是否严格来自操作后的采集时刻也仍需
  进一步审计，不能仅靠“帧 ID 未重复”认定时间因果已满足。

### M3 采集时序与按需 JPEG（2026-09-09）

- 核对传感器 bridge：RGB header.stamp 使用 `_normalise_ros_capture_stamp`
  映射后的本机 ROS 时钟。实物 policy 将 ROS now 作为 capture_clock 注入
  verifier，并开启 require_post_start_frames；验证起点在 actuator 返回后
  的 verify 入口，缺失/早于起点/不递增/超前当前时钟 0.1 s 以上的采集时间
  均不提交模型。通用回放默认不启用，避免假定所有源都使用 Unix/ROS 时钟。
- 实物稳定时长改按已接受帧的采集时间差计算，而非模型调用开始时间差；
  少量毫秒间隔的缓存帧不能因慢模型调用积累出三秒稳定证据。事件记录
  verification_start_capture_stamp 与 stable_duration_basis 便于实机核对。
- 查实 physical policy 在空闲时仍对每帧 RGB 做 PIL→JPEG→base64。现在
  ROS 回调仅保留最新消息引用；M3 取图时编码，同一消息复用缓存。编码不
  持有回调锁，新帧在编码期间到达时不会被旧编码结果覆盖；缓存为有界单帧。
- 本机固定黑色 848×480 RGB、100 条不同消息，预热后五次中位数：逐帧编码
  106.30 ms，latest-only 回调后仅取一次图 1.098 ms。该数值仅说明消除了
  99 次无用转换，不是整体链路 97 倍加速。正常 M3 活跃期转换频率随取图
  频率而定，空闲期为零；首个取图仍需要一次实际编码。
- 新帧/时钟、图像缓存、physical platform 联合测试 102 passed、2 个 ROS
  弃用警告；新增采集顺序、旧帧、未来帧、缺时间戳、慢模型不能替代采集
  稳定区间、100 次空闲回调零编码以及编码中有新帧到达的竞态测试。
  Python 编译及 diff check 通过。未连接、重启 Go2 或提交。
- 此校验依赖 ROS header.stamp 的采集含义；重新基准化期间的真实源帧龄、
  实机时钟漂移和操作开始/结束语义仍需真实数据验证，不能把消息时间检查
  当作已经证明机器人动作或物体物理状态变化。

### 3D 跟踪的独立采集确认（2026-09-09）

- 查实 ObjectMapStore.update 对同批多个命中同一 track 的框逐次递增
  observation_count/hit_streak，单张图可凑满两次确认。现在一个 track 在
  一个 batch 内最多记一次；多个独立目标仍分别参与匹配与确认。
- Store 保留最新处理采集时间，重复/倒序/非有限时间直接返回 False。Mapper
  尊重该返回值，在导出 tracked detections 和更新 graph 之前退出，因此
  重复帧既不增加命中/漏检计数，也不增加 graph 的确认次数或重复计算。
  沿用 objects 清空的 episode-reset 入口后，允许新 episode 时钟重新起步。
- 配套修复 mapper 时间读取只支持 secs/nsecs 的问题：现在接受 stamp_sec、
  stamp、capture_stamp_sec；显式无效时间不再伪造成当前回调时间。完全
  没有时间字段的旧适配器仍保留 receipt-time 兼容，无法据此保证去重。
- 几何进入匹配和稳定历史前检查 NaN/Inf 坐标、中心、尺寸、yaw，以及负
  尺寸和非法置信度；坏观测跳过，不阻止同批其他目标处理。未凭合成数据
  改动实物匹配半径、箱体尺寸门槛或有限大幅跳变的空间剔除策略。
- 跟踪/mapper 刷新/graph ablation 联合回归 42 passed、4 个 ROS 弃用
  警告；另一次跟踪与现有 object_store/物体持久化子集 26 passed。测试
  覆盖一帧十框只能计一次、整帧重放不能确认/移动/累计漏检、下一帧确认、
  时钟重置、无效几何不污染历史、不同时间字段在 mapper 入口阻断重复。
- 未连接或重启 Go2、未提交，diff check 通过。真实断流重连、有限位置
  离群值、相邻同类目标区分和稳定 3D box 质量仍需回放或实机数据验收。

### 几何解析收紧与部署目录回归（2026-09-09）

- 查实 `_point_from_detection` 对损坏的数组字段吞掉转换异常后返回零向量，
  完全缺少位置的检测也能落到原点；展示用 aabb 字段还在稳定几何检查之后
  独立读取，可把坏值送入 viz box。现在显式损坏/不完整字段抛错，由单物体
  更新入口隔离；稳定与展示的中心/尺寸均在更新前检查。
- 位置缺省时允许回退到有效 world_box3d_center/box3d_center/aabb_center；
  相应 box 中心和尺寸也支持 AABB 字段。所有位置来源都缺失则拒绝创建
  track，不再虚构 [0,0,0]。旧的仅位置观测仍可保留无尺寸的兼容表示。
- 新增短数组、坏字符串、不完整字典、坏展示几何、缺失位置、仅 OBB/AABB
  观测等测试。跟踪/mapper/ablation 联合 52 passed，已有 object_store/
  M1 持久化子集 16 passed；两次实物部署 tests 目录回归均为 159 passed、
  2 个 ROS 依赖弃用警告，后一次使用本轮最终代码。diff check 通过。
- 159 项是当前实物部署目录的离线测试，不代表整个仓库测试通过，更不
  代表实时帧率、实机几何质量或导航成功率。未连接或重启 Go2、未提交。

### M1 名称提交与实验重置隔离（2026-09-09）

- 修复 mapper 在 graph 校验前就修改 tracker 名称的问题：现在先由 graph
  接受 patch，再同步已经确认的规范名称。首次仍待确认的 refrigerator
  提案、过期 request_sequence 和上一检测流的结果均不能提前改名。
- 已接受的名称立即写入 tracker 标签和投票历史，并使匹配索引失效；后续
  原始 YOLO 检测即使没有携带 tracker 自建 ID，也沿已匹配物体的规范名称
  更新，避免饮水机被下一帧 refrigerator 原始类别改回去。
- ObjectMapStore.reset 清理对象、ID 计数、改名缓存、匹配索引与采集水位，
  保留配置；mapper reset 同时旋转 tracked_stream_epoch。M1 回调在锁内
  校验当前 graph episode 或当前检测流身份，拒绝旧流迟到结果。
- 继续发现锁外发布的身份竞态：旧检测快照会读取 reset 后的新流标识。
  现在在同一 mapper 临界区取检测与流标识快照，JSON 编码仍在锁外进行。
  回归注入快照后的流切换，验证旧数据不会被标记为新实验。
- 跟踪、mapper、graph ablation 初次回归 55 passed；增加发布竞态测试后，
  与实物部署 tests 目录联合回归 215 passed、4 个 ROS 依赖弃用警告，
  耗时 11.45 s。git diff --check 通过。未连接或重启 Go2、未提交。
- 此轮不代表整个 reset 链路已验收：无 episode 的旧适配器仍保留兼容；
  锁外 portal hints、跨 topic 到达顺序和实际断流重连仍需继续审计。
  上述测试不代替真实帧龄、实机建图与导航停止效果的验收。

### 门提示交接解除检测线程对分割锁的等待（2026-09-09）

- 查实 `_process_object_message` 最后仍同步调用 `_update_room_portal_hints`，
  后者等待整个房间分割持有的 `_room_lock`。虽然 OCC 回调和分割线程已
  分离，检测 worker 仍会在这里停顿，并使 latest-only 检测被动丢帧。
- 实际后台模式改为提交门观测到 64 批有界队列，由房间 worker 在每次房间
  作业开始时按顺序应用。非门观测不触发额外房间作业；未改变既有门确认
  次数和几何稳定逻辑。队列满时淘汰最旧批次并累计 dropped counter，
  避免分割异常缓慢时无界积压；这不是对实际吞吐率的保证。
- reset 不再从回调等待分割锁；worker 在新 epoch 作业入口清空自己的
  分割状态和缓存，即使还没有 OCC 也先完成状态清理。每批提示携带观测
  快照的 epoch，旧 worker 请求不能清空新队列，旧门提示不能留到新实验。
- 修复房间请求合并无条件继承 pending post_open_results/force_stable
  的问题：仅同 epoch 合并。OCC 回调、GT 和门提示入队携带其快照 epoch，
  防止旧 OCC 回调在 reset 后才提交时把旧开门结果标记为新实验。
- 新增实际 mapper 方法的锁竞争和队列测试：其他线程持有分割锁期间，
  检测提交与 reset 请求均完成；两批观测仍累计两次确认；覆盖 100 批输入
  保留最新 64 批、跨 epoch 隔离、同 epoch 保留开门证据、无 OCC 清理。
- 初次房间专项 61 passed；最终部署目录、房间、mapper、规划缓存和图缓存
  联合回归 235 passed、4 个 ROS 依赖弃用警告，11.03 s。diff check 通过。
  测试入口已写入 test.md。未连接/重启 Go2，未提交。
- 本轮证明的是解除特定锁等待，不代表实机达到 10 Hz。真实分割排队时延、
  队列淘汰量和 OCC 帧龄仍需实测；reset 与锁外 planning 发布、旧 raw OCC
  快照之间的隔离仍需继续审计，不据此宣称整个重置链路完全验收。

### 四层图父子关系持久化与缓存边界（2026-09-09）

- 本轮先跑完整 mapping tests：298 passed、18 failed。失败集中在既有
  portal 测试；已确认部分测试用默认 0.1 m 立方体模拟门，不满足当前门
  几何限制，其他门状态/确认前提仍待逐项核对。本轮未放宽过滤或删除断言。
- 复现真实关系缺陷：重建第一阶段已将物体 parent_id 覆盖成房间，后面
  隐藏内容的 previous_parent_id 实际读到的是房间。现在在重建前快照父
  节点，物体不可见时优先保留仍存在且房间不冲突的父容器；可见新观测
  仍按几何重新判断。邻近容器的新重叠不是隐藏物体转移的证据。
- 修复无合适父容器时重新使用 stale room_id 的问题，保留第一阶段选定
  的有效房间/scene 回退。显式观测首次引入、尚不存在的房间仍正常创建，
  与已有但被停用的房间区分；不能用停用来误删已知房间，也不重新激活它。
- 复现父关系缓存每次位置变化保存一个版本、一直增长的问题：固定单物体
  300 次更新原为 301 条，现为 1 条；每次重建仅保留当前活跃物体的键，
  删除物体时相应缓存消失。开放冰箱的无缓存推断路径不会保留旧版本。
- 缓存几何从 1 cm 取整改为精确浮点元组，并包含 centroid。原规则在
  1.459→1.461 m 的包含边界变化中误用缓存，修复后正确撤销 contains。
  几何不变时仍命中原缓存测试；几何变化时可能增加重算，这是正确性取舍。
- 新增六项回归，先复现四项缺陷再修复。最终完整 mapping tests 为
  304 passed、同样 18 failed；部署目录与层级/缓存/房间属性联合测试
  177 passed、2 个 ROS 弃用警告，11.24 s。diff check 通过。
- 尚不能宣称四层图或完整 mapping 测试验收通过。18 个门相关失败、真实
  OBB/yaw 下的包含关系、M1 改类型后的派生属性清理和真实数据质量仍待
  后续检查。未连接/重启 Go2，未提交。

### 门测试前提与 M1 确认顺序收敛（2026-09-09）

- 逐步核对上轮 18 个 portal 失败：14 个由状态/可见性测试的默认 0.1 m
  立方体触发现有门几何拒绝。测试 helper 仅将默认 door/is_door 尺寸改为
  0.1×0.9×2.0 m，显式尺寸仍优先；另一个 doorframe 测试补齐有效几何。
  未修改真实门尺寸阈值，新增三种显式坏几何 + 高置信 M1 仍不能建房测试。
- 两个潜在房间测试缺少现行两帧与 M1 具体名称确认，补齐真实输入前提，
  保留房间连接、潜在房间不写 OCC 等原断言。补齐后还复现了代码顺序缺陷：
  首次 M1-open 先尝试建房，最后才更新 persistence gate，故同次确认不能
  生效。现改为先更新 gate 再按已接受 open/ajar 状态创建图上潜在房间。
- 复现 M1 confidence=0/0.49/负数/超范围/NaN/Inf 时仍可能写 open 状态，
  非数字可抛异常；portal persistence 分支还漏掉普通物体已有的 0.5 下限。
  现在 ready 语义结果进入任何修改前要求有限且在 [0.5,1]，否则拒绝，
  不覆盖已接受的名称、状态或确认元数据；pending/failed/stale 通知仍保留。
  `_has_m1_portal_confirmation` 同样检查置信度，避免载入的坏属性绕过门控。
- 失败 M3 的 post_state 已被 resolver 丢弃，但丢弃原因未进入 portal gate
  诊断。现在仅为诊断保留 requested_state，失败明确拒绝，仍不应用该状态；
  即使来源字符串为 oracle/verification 也不能用 success=False 声称开门。
- 新增/扩展低置信及非法结果、已确认状态不能被坏结果覆盖、坏几何、失败
  来源等回归。mapping 全目录先为 332 passed；最终增加两项来源参数后，
  mapping 全目录 + physical_nav/tests 联合为 493 passed、4 个 ROS 依赖
  弃用警告，12.33 s，diff check 通过。上述 18 项均已处理，未删除断言。
- 这只是两个目录的离线回归，不是全仓库或实机验收。M1 低置信被拒后上游
  的重试/状态显示、真实传感器帧龄、OCC 稳定性与导航成功率仍需继续验证。
  未连接/重启 Go2，未提交。

### M1 原始置信度与成功缓存一致性（2026-09-09）

- 查实 graph 拒绝低置信度后，上游 `_infer` 仍会把合法 JSON 发布为 ready，
  计入 completed 缓存；普通容器可等待成功刷新间隔，locker 冰箱复核还会
  消耗有限的两次尝试。这会让未确认对象迟迟不再被评估。
- 进一步查实 schema 的 `_confidence` 使用 min/max 归一化，NaN/Inf 或
  大于 1 的值可能变成 1；因此校验放在 validate_attribute_patch 之前，
  检查原始顶层 confidence 有限且处于 [0,1]，并满足 graph 的 0.5 下限。
- 失败走原有 finally 清理：发布 failed 与 low_model_confidence 或
  invalid_model_confidence，释放 pending，记录实际调用的冷却时间；
  不写成功缓存、不累计 locker 复核次数。不通过取消冷却来强行提高调用率。
- 八种坏/弱置信度均先复现原始代码错误 ready 后再修复。实际 `_infer`
  回归覆盖拒绝后立即重试被冷却阻挡、2.1 s 后可重新预约、后续 0.95 回复
  成功且仅计第一次复核。未调用在线模型，使用假时钟与固定测试图。
- mapping 与实物部署目录联合 501 passed、4 个 ROS 弃用警告，11.85 s；
  后续补全成功重试断言的请求状态专项 30 passed，diff check 通过。
- 当前只校验物体回复的顶层原始置信度；嵌套 aperture/frontality 等置信度
  的归一化、room 回复，以及已确认物体遇到新 pending/failed 时图端是否
  保留语义确认还需继续检查。不能把本轮视为所有 M1 状态流验收。
  未连接/重启 Go2，未提交。

### 请求状态与已确认拓扑分离（2026-09-09）

- 复现已确认门收到 pending/failed/stale 后，provisional-room 清理将其
  潜在房间删除的问题。`attribute_status` 是最新请求状态，却被直接当作
  语义确认有效性的唯一状态；模型刷新或服务失败不应撤销既有空间关系。
- 新增小型 `attribute_last_ready` 快照，仅记录有效语义回复的确认状态、
  置信度、类别、名称、冰箱待复核标记和更新时间。当前请求状态保持原值，
  不伪装成 ready；持久化/门拓扑门控在请求未完成或失败时读取历史确认。
  旧图已有 ready 属性时，在首次请求状态覆盖前也会建立快照。
- 新的 ready 语义结果会更新快照；有效 M1 把 door 改为 wall_panel 后，
  门确认与潜在房间仍会撤销。尚无有效回复的 pending/failed/stale 不产生
  确认，快照不包含 view_state/approach_ready，不替代接近阶段的新视角审核。
- 三种状态均先复现删除再修复，回归验证同一潜在房间 ID 保留及后续纠错
  可撤销。最终 mapping + physical_nav/tests 为 504 passed、4 个 ROS
  弃用警告，12.29 s；决策 pending portal 专项 1 passed、79 deselected。
  diff check 通过，未连接/重启 Go2，未提交。
- 仍待审计：跨 track 合并时的确认快照迁移、failed 状态在各候选/网页
  消费者的展示与许可边界、嵌套置信度。同时本轮测试观察到强制潜在房间
  缺几何时可沿默认 room-ID→坐标回退产生极大坐标，下一步需核对该路径。
  尚不据此宣称完整 M1 状态链或实机拓扑稳定性已验收。

### 强制潜在房间的几何与身份统一（2026-09-09）

- 复现 forced-room 分支只 `_ensure_room_node(id)` 不提供几何，因此走
  `_default_room_center` 的 `[float(room_id),0,0]` 回退；从 1000000 起的
  潜在房间编号变成百万米坐标。开门分支另行分配 1000001，留下重复房间。
- 强制分支改用已有门侧矩形推断与统一 allocator，先要求已确认门和源
  房间关联，再创建局部几何。强制生成与开门结果复用同一 child ID；
  旧的仅 potential_room_ids 表示也能认领，异常几何在原 ID 上修正。
- 实际开门接近方向可更新 prior 的门侧估计；之后强制维护不会把更强的
  开门几何降回 prior。无需在缺少源房间时虚构一个带无依据坐标的房间。
- 给 `_ensure_room_node` 直接传入 center/size/cell_count，消除创建这种
  本来就不在 OCC 中的 ID 时对整张 room grid 的两次查找。测试用抛错
  替身确认不再进入中心/尺寸默认扫描；未据此报告实机毫秒级收益。
- 修复 prune 使用整数 potential room ID 匹配字符串节点 ID 的问题，
  分开维护数值 ID 集合与节点 ID 集合，清理失效引用，也覆盖被改类的原门。
- 五项新回归覆盖局部几何且不扫描、两侧接近方向下同 ID 复用、旧百万米
  坐标原位修复及拒绝门后的引用清理。最终 mapping + physical_nav/tests
  509 passed、4 个 ROS 弃用警告，11.84 s，diff check 通过。
- 未改变 OCC 自由/未知单元，潜在房间仍标记 observed_free_space=False、
  cell_count=0，不是实测房间边界。通用未观测房间的默认 ID 坐标回退、
  潜在房间真正可达性、倾斜 OBB 的方向估计及显示效果仍待继续验证。
  未连接/重启 Go2，未提交。

### 候选足迹几何一致性与端点缓存（2026-09-09）

- 本轮决策目录完整回归基线 417 passed、5 failed，失败均在容器观测点
  排序预期，尚未逐项确认是否历史测试前提。本轮未修改排序或放宽断言。
- 另行复现真实碰撞缺陷：端点只按障碍格中心距离检查，且忽略目标在
  格内的位置；0.1 m 栅格中 (0.29,0.29)、半径 0.25 m 可碰到 (4,5)
  障碍格的角，原实现仍报告 free。改为圆与方形栅格区域的相交检查，
  保留亚栅格目标坐标与旋转地图原点，接触边界按保守碰撞处理。
- 共享寻路核此前把半径先向上取整成整格，再判格中心距离，与端点不一致；
  现在按同样圆/格相交语义预计算格中心核，避免无依据的整格半径扩大。
  例如 r=0.12 m、障碍格边界距离 0.15 m 的位置不再被误拒绝。
- grid_path_status 单独核对实际目标坐标的足迹，不用目标所在格中心代替。
  OccupancyTraversal 新增每批次端点缓存，候选预检与寻路入口复用一次
  精确检查，新增 endpoint_footprints_checked/cache_hits 计时统计字段。
  新地图批次建立新缓存，不复用前一地图的 free 结论。
- 新回归覆盖格边碰撞、旋转/平移原点、四种半径下 11×11 全格中心语义
  一致性、半径不整格扩大及候选/寻路只检查端点一次。mapping、实物部署、
  寻路与候选预检联合 539 passed、4 个 ROS 弃用警告，12.28 s。
  中间决策全目录为 424 passed、同样 5 failed（最后另加一个旋转参数）。
- diff check 通过，未连接/重启 Go2，未提交。本轮未改变机身半径配置或
  inflation，也不取代局部规划最终轨迹碰撞检查；通用输入几何非法值、
  候选排序失败与实机窄通道通过率仍需继续处理/验证。

### 容器距离排序索引与寻路热循环（2026-09-09）

- 核对此前决策目录的 5 项失败：旧测试要求固定面/半径顺序，而运行逻辑
  已按机器人到候选的距离排序。测试现在先断言最近优先，再按标签检查
  各面的几何、观测预算、动作点及索引关系；没有删除原有几何要求。
- 同时发现真实缺陷：切向观测点只能引用前面的基准点。距离重排把它放到
  基准点前面时，会用切向方向生成接触点。现在允许前向索引，动作坐标与
  标签均从基准面获取，M1 拍摄仍保留切向偏移；新增专项回归。
  决策全目录 431 passed；三个相关目录联合基线 940 passed。
- 继续 profile 批量栅格寻路：120×120、0.1 m、半径 0.25 m、有墙/门洞/
  未知区域、20 个候选的合成场景中，原实现约 17.4 万次 cell_state 调用。
  已有更优代价的边仍重复检查足迹与对角角点。将代价筛选提前，只有可能
  改善路线的边才检查碰撞；不修改足迹、障碍阈值、预算或局部规划器。
- 同进程交替跑优化前后各 9 次：中位耗时 131.82→89.75 ms（约减少 32%）；
  缓存命中查询 162483→24132，20 个完整结果的 SHA256 相同。另做 80 张
  随机小地图、960 个查询的逐字段差分，包含未知区策略、不同半径、搜索
  预算和对角开关，全部一致。新增无效回边不查询足迹的单元测试。
- 最终 mapping、decision、physical_nav 三个测试目录联合 941 passed、
  4 个 ROS 依赖弃用警告，13.29 s；git diff --check 通过。
- 上述为离线合成候选搜索收益，不是 OCC 发布频率或在线导航成功率。
  未连接或重启 Go2，未提交优化改动；实际传感器帧龄、GMapping 各阶段
  延迟与真实房间分割仍待实测。

### 点云过载时的持续输出与连接隔离（2026-09-09）

- 检查传感器→GMapping 时发现此前 latest-only 实现过度取消：
  `_cloud_sequence_is_current` 同时比较待处理/已解码 RGB 帧，在点云
  投影后再检查一次。若投影持续超过输入周期，每次计算期间都来新帧，
  所有完成结果都被丢弃，OCC 将缺少输入。不能把这种断流当成降低延迟。
- 新增离线复现：模拟连续四次投影期间都有后继帧，修改前发布列表为空。
  改为只替换等待槽，已开始的同连接投影允许完成；下轮直接取最新等待帧。
  不改 PointCloud2 的 capture stamp、frame_id 或 bridge seq，不伪装新鲜度。
- 点云任务携带原传输 session/connection，在计算前后核对连接代际，
  拒绝旧连接排队任务及重连期间算完的结果。兼容尚未启用标记连接的
  legacy/replay 流，一旦出现标记连接，旧 legacy 结果不再接受。
- 六项专项覆盖持续输入下有输出、同 session 重连/新 session 期间丢弃、
  旧连接在投影前拒绝、legacy 边界及实际 worker 的 1→3 最新等待槽交接。
  之前要求“有新帧便丢弃完成投影”的测试同步改成代际隔离要求。
- 最终 mapping、decision、physical_nav 三目录联合 947 passed、4 个 ROS
  依赖弃用警告，12.87 s；git diff --check 通过。
- 时间链路检查确认：当前 GMapping OCC header 保留触发 scan 的 seq/stamp；
  publish 调用计时只覆盖本节点 publish 开销，不代表下游接收延迟。入口
  source_age 采样仍在 time_sync_guard 内，TF 等待没有独立阶段统计；这些
  可观测性缺口需后续补齐，不能仅凭 callback 耗时判断整条 OCC 链路健康。
- 这次是可复现的过载逻辑修复，不证明它就是此前实机卡顿的唯一原因；
  未连接/重启 Go2，未修改 C++ 或发布频率配置，未提交。

### OCC 延迟统计边界补齐（2026-09-09）

- GMapping 的入口 source_age 原来只在 enable_time_sync_guard 内采集，
  关闭丢弃/对时门控后会失去诊断。现在在启用 timing 的回调入口独立采样，
  不修改任何接受/拒绝策略，继续保留输入和 OCC 的采集时间戳。
- 为 PointCloud2 与 organized-depth 两条路径，在消息加入 TF filter 前
  记录 SteadyTime；成功回调统计 tf_filter_wait，失败回调记录 wait_ms。
  原回调总耗时不包含这段等待，新增指标独立展示，避免等待被遗漏。
- 新增 IngressTimingBuffer：每路最多 256 条指针/时间元数据，不持有
  传感器消息，不增加传感器队列；成功与失败均消费记录，未观测、淘汰或
  无效时钟返回缺失值。显式保存/断开 subscriber→filter 回调连接，避免
  析构后继续调用过滤器。TF 的目标 frame、容差与队列配置保持不变。
- map_source_age 在 OCC publish 完成后采样；semantic mapper 新增
  occupancy_source_age_at_callback，在接收回调入口记录同一采集时刻的
  累计帧龄。原 room worker 的收到消息时间记录在获取 mapper 锁之后，
  会漏掉锁等待；现改在入口捕获并随原 OCC 指针保存。
- C++ 小型运行测试覆盖乱序出队、一次性消费、有界淘汰、地址复用、坏
  时钟与零容量，另有接线静态检查。Python 专项覆盖锁等待不能消失，以及
  零/未来/非有限时间戳不伪装为 0 ms。slam_gmapping 目标编译链接通过。
- mapping、decision、physical_nav 三目录联合 956 passed、4 个 ROS 依赖
  弃用警告，12.84 s；git diff --check 通过。
- 这些是定位瓶颈的指标补齐，不是线上提速结论。TF wait 只从本进程
  subscriber 回调开始，ROS 调度/网络/解码前的等待未独立分离；累计帧龄
  依赖 ROS 时钟域一致，不能与各阶段耗时相加，也不能跨统计窗口相减来
  宣称网络延迟。未连接/重启实物或启动 ROS 服务，未提交。

### YOLO RGB-D 回调错位下的配对进度（2026-09-09）

- 未发现 YOLO 像旧点云 worker 一样在推理完成后因新帧到达而取消结果。
  但原 `_latest_raw_frame` 仅查看各路最新图；RGB N、RGB N+1、Depth N 的
  持续到达顺序会令严格帧号检查一直失败，两路在发布而 YOLO 没有输入。
  新增实际 image callback 测试，修改前首个完整捕获就返回 None。
- 引入 `_ExactRgbdPairBuffer`，每路最多保留 4 个未配对帧，按
  `(bridge seq, ROS capture stamp)` 精确匹配。只暴露最新完整对；更新的
  不完整半帧不覆盖完整对，不把全部匹配结果排队给推理。回调自有 ndarray
  只保留引用，缓存不额外复制图像；完整对更新后清除更旧未匹配半帧。
- 使用 bridge 已归一化的单调 ROS capture stamp 排序，不按源相机硬件
  时间排序。去重键加入 capture stamp，避免 bridge 重启后 seq 恰好复用
  时被当成重复。seq=0 的 legacy bag 仍保留旧时间容差路径，未放宽实物
  的配对条件；完全退回旧 ROS 时钟域而不做 bridge 归一化不在此契约内。
- 八项回归覆盖 RGB/深度分别领先一帧、取最新完整对、相同 seq 不同 stamp
  禁止混配、seq 重用、标定晚到、缓存容量/旧包回流与 legacy 路径。
  本机 7×10000 对引用配对微基准中位 1.633 µs/对，无额外图像复制；不含
  ROS 解码、GPU 或 3D 后处理，不是在线 10 Hz 证明。
- mapping、decision、physical_nav 三目录联合 964 passed、4 个 ROS 依赖
  弃用警告，14.18 s；git diff --check 通过。未连接/重启实物，未提交。
  大于缓存窗口的跨路积压、ROS 图像传输缓冲、采集位姿缺失时的 3D 输出
  策略和实机帧龄仍需继续验证。

### 采集位姿缺失时禁止虚构世界几何（2026-09-09）

- `_telemetry_at` 未找到匹配遥测时返回空字典，但 infer 原来无条件调用
  `_world_transform`，后者默认 position=[0,0,0]、yaw=0。由此生成的
  world box/segment 带 telemetry_fallback，看似有效，可以进入语义图。
- 在实际 infer 的世界变换/投影前加入采集位姿门控：必须有完整、有限
  三维位置及有效四元数或显式 yaw/rpy。缺失、非法数据不再表示世界原点；
  显式合法零位置/零 yaw 仍正常接受。
- 无有效位姿时继续 GPU 检测、类别/置信度过滤及 2D box/overlay 发布；
  所有候选标记 geometry_skipped 与具体原因，report 增加
  capture_pose_status。不调用世界变换/深度投影和 geometry worker，避免
  计算后再丢弃。复用 gateway 已有 geometry_skipped 的 3D admission
  过滤，不将这些 2D-only 记录伪装成原点物体。
- 修复 world transform 的 yaw 回退表达式提前求值：即使显式 yaw 合法，
  可选 imu.rpy=[] 也曾触发越界并丢失整帧；现在仅在需要回退时读取它。
- 14 项专项使用固定模型输出来执行真实 infer：覆盖缺失/非有限/坏形状
  位姿、保留 2D 且无世界字段/投影、合法零位姿、恢复后重新进入 lifting
  和空 rpy。恢复测试的 geometry worker 使用深度拒绝替身，不冒充真实
  3D 质量验收。最终三个目录联合 978 passed、4 个 ROS 依赖弃用警告，
  13.93 s；git diff --check 通过。未连接或重启 Go2，未提交。
- 本轮未改变最近遥测的 0.5 s 容差，也未完成位姿插值、camera_pose 与
  body pose 回退来源、外参非法值等审计。时间相关几何质量仍需真实同步
  数据验证；不能把本轮理解为所有“有效字段”的位姿都已足够新鲜准确。

### 统一传感器 TF 与 YOLO 的机身姿态（2026-09-09）

- 新建真实 publish_pose/世界变换对比测试，原代码 5 项均失败：显式机身
  yaw 被 YOLO 的 camera_pose 回退抢先使用；sensor 原样发布非单位 IMU
  四元数，而 YOLO 归一化；字典格式四元数、base_pose 及 IMU rpy 在两端
  的支持不一致。仅“都有姿态字段”不足以证明几何一致。
- SensorRosBridge 提供共享的 body position/quaternion 解析器，YOLO
  姿态门控与世界变换共用。Unitree IMU 列表用 wxyz，字典与其他明确
  body-pose 列表用 xyzw；有效四元数统一归一化，缺失/非法返回 None。
  camera_pose/d435i_pose/camera_imu 不再作为机身方向的替代来源。
- 原 sensor publish_pose 也会把缺失位姿变成零位置/零方向。现在在更新
  时间水位与发布 odom/TF 前验证机身位置和方向，坏样本不发布、不推进
  水位，后续有效样本恢复。相机杆 IMU 校正仍在 base→camera 变换中处理，
  不改变相机外参角度、杆长或坐标轴约定。
- 14 项专项涵盖真实发布与 YOLO 一致、三轴非单位四元数、机身 yaw 与
  相机提示共存、无效/相机-only 位姿、恢复及替代 orientation 字段。恢复
  时间断言直接比较输入 ROS Time（保留纳秒），避免用十进制 float
  10.1 对 ROS 纳秒截断结果做错误的精确比较。
- 最终 mapping、decision、physical_nav 三目录联合 992 passed、4 个 ROS
  依赖弃用警告，13.84 s；git diff --check 通过。
- 另查实时间匹配仍不一致：sensor 私有参数 telemetry_max_delta_sec
  为 0.15 s，YOLO 最近遥测阈值仍硬编码 0.5 s。这需要后续统一和记录
  匹配偏差；本轮先验证空间姿态契约，未宣称两个消费者拿到的是同一
  时刻的遥测样本。未连接/重启 Go2，未提交。

### 统一遥测匹配容差与拒绝策略（2026-09-09）

- 两项新测试先复现：10.3 s 捕获匹配 10.0 s 遥测，sensor 拒绝而 YOLO
  接受；sensor 空历史直接返回最新字典，即使无法证明其采集时间。
- 两端共用 `_nearest_capture_telemetry`，默认最大偏差统一 0.15 s。
  仅查询已到达历史，不等待、不外推、不重写时间戳；空历史和非法捕获/
  容差拒绝，坏历史时间戳不遮蔽好样本。等距时优先过去样本，同时间戳
  优先后收到的快照，返回副本不修改原历史。
- sensor 在锁存 depth_calibration 中携带实际 telemetry_max_delta_sec；
  YOLO 支持启动/replay 默认参数，收到 sensor 契约后采用实际值。避免
  YOLO 启动早于 roslaunch 私有参数加载时永久用错默认值，不在每帧进行
  ROS 参数 RPC。非法更新不放宽已有 YOLO 容差，其他标定照常处理。
- 新 raw/report 字段 capture_pose_match 记录匹配绝对偏差、容差及是否
  接受；位姿本身的有效性继续由 capture_pose_status 检查，时间匹配不
  等于具有合法位置/方向。过旧遥测返回空字典，复用前轮的 2D-only 路径。
- 19 项时间匹配回归覆盖超龄/空历史、双端正负偏差一致、锁存更新、非法
  更新、乱序等距、同时间戳替换、坏时钟及输入不被修改。最终三个目录
  联合 1011 passed、4 个 ROS 依赖弃用警告，13.97 s；diff check 通过。
- 未连接/重启 Go2，未提交。这一轮统一的是选择策略，不保证两端在不同
  到达时刻拥有相同历史；仍需核对 telemetry envelope 的 stamp 是否与
  内容严格对应、是否需要发布权威的逐帧匹配位姿，以及实测匹配偏差。

### 遥测发布的内容/时间身份一致性（2026-09-09）

- 通过真实 update_telemetry→发布器替身→YOLO callback 的链路复现 5 项
  失败：旧包合并了当前姿态却用旧包 stamp 发布；未标记连接的包可被附上
  活跃连接标签；重连只清 source stamp 不清 capture seq；允许新 capture
  证明时钟回退后仍保留旧时间最大值，随后正常遥测会被当作旧包。
- 迟到包仅合并非位姿诊断到本地状态，不再向 ROS 遥测支路发布该合成
  姿态，避免 YOLO 将当前姿态错误匹配到历史图像。pose 字段保护集合
  补上 base_pose/odom/pose/camera_pose/d435i_pose，防止通过别名绕过。
- 已激活标记连接后拒绝无标记包，重连同时重置 source/capture 水位。
  已接受的新 capture 证明源时钟回退时，源时间水位跟随新基准；反向
  保护旧 capture seq，不能用回退前较大的 source stamp 抢回当前姿态。
- 六项专项覆盖原五个复现分支及回退前旧捕获重放；传感器与 YOLO 对旧
  时刻均不再查到合成新姿态，已有正确样本的 timestamp/content 保持不变。
  最终三个目录联合 1017 passed、4 个 ROS 依赖弃用警告，13.40 s；
  git diff --check 通过。没有新增等待/队列或在线调用。
- 未连接/重启 Go2，未提交。仍需审计 capture-time TF 与周期 TF 水位在
  时钟回退时的关系，以及两个独立遥测历史是否应替换为逐帧权威位姿；
  本轮没有宣称完整时间同步或实际 OCC 墙面质量已经验收。

### 周期 TF 的时钟域与并发发布水位（2026-09-09）

- 原 refresh_tf 读取 received_at 源时间而不是已归一化的 ROS 时间；
  `_last_tf_source_stamp=max(...)` 在源时钟回退后会阻止后续正常周期
  更新。另有历史捕获经 publish_pose 重新发布较旧 odom 的路径。
  四项新增测试先复现错误时间、停更、odom 倒退与缺 ROS 收据仍发布。
- update_telemetry 在接受快照时，原子保存对应 `_telemetry_ros_stamp`；
  迟到诊断合并不改变它。周期刷新仅用该 ROS 收据，无有效时间时不采用
  source clock 或 now 兜底，按 ROS 纳秒水位去重。
- capture worker 与周期 timer 共用 pose publish 锁，完整发布后推进
  ROS 时间水位。源时间仅为诊断，新的 ROS 捕获可带较小的源时间而不冻结。
  历史样本仍发送 TF，供历史点云查询，但不再倒退实时 odom 发布。
- 两项事件驱动并发测试暂停真实发布路径，验证 timer 等待同一发布锁、
  水位顺序和捕获已更新时跳过旧 timer 样本。测试不靠长时间 sleep，不
  连接 ROS master。最终三个目录联合 1023 passed、4 个 ROS 依赖弃用
  警告，13.81 s；git diff --check 通过。未连接/重启 Go2，未提交。
- 未据此宣称新增锁在线无开销，仍需测量发布序列化时间与实际 TF 查询。
  两个消费者独立挑选遥测样本、逐帧权威位姿方案、重连期间标定上下文及
  GMapping 对历史 odom/TF 的时间门控仍需后续集成验证。

### 重复门合并时保留有效 M1 身份（2026-09-09）

- 原合并排序仅按检测次数/置信度挑选幸存节点。新 track 即使没有 M1
  证据，也可删除已经确认的公共门 ID；M1 刷新为 pending/failed/stale
  时尤其容易丢失历史确认，门后潜在房间因 source portal 消失而被清除。
  四种状态的新增测试先复现这一问题。
- 合并优先级改为有效 M1 门确认优先，再比较原检测次数/置信度；复用
  已有证据有效性与 attribute_last_ready 规则，不降低两帧/M1/几何阈值。
  保留已确认公共 ID、属性、交互历史，合并观测次数最大值与可见性并
  重新计算持久化门控。不累加重复 track 的次数，避免人为创造观测证据。
- 又复现旧的 bare-ready 复制分支会用低置信度 ready 覆盖有效 pending
  节点状态；删除这一绕过证据校验的复制分支。有效确认已由排序保证
  优先保留，重复节点不得仅凭 ready 字符串覆盖幸存节点的交互状态。
- 7 项专项覆盖上述状态、无确认时保持检测排序、不捏造 M1、可见性
  合并、原公共 ID 接收后续 M1 结果以及低置信度覆盖拒绝。开放状态测试
  提供原有门洞可见证据，不放宽 open 判定。联合三个目录 1030 passed、
  4 个 ROS 依赖弃用警告，13.56 s；git diff --check 通过。
- 本轮仅离线验证，不连接或重启 Go2，未提交，不宣称在线延迟已验收。
  两个都已经确认且各有潜在子房间的重复门，仍需单独审计双向房间引用
  与合并后的异步结果路由；本轮只保证确认门对未确认新 track 的身份稳定。

### 修正真实 ROS 序列化下的 RGB-D 配对身份（2026-09-09）

- 继续审计逐帧权威位姿协议前，发现先前“Image.header.seq 保留本地
  receipt 序号”的假设不成立。本机 ROS1 `rospy/msg.py:serialize_message`
  会重写顶层 Header.seq；`rospy/topics.py` 为有连接的每个 publisher
  分别计数。RGB-only 订阅早于 depth 或订阅重连可造成两边永久计数差。
  此时要求 seq 相等的配对器会拒绝本来同步的 RGB-D。此前仅直接调用
  图像 callback 的测试绕过了这一步，没有覆盖真实传输契约。
- 新建 3 项失败复现，使用真实 Image 构造、ROS 序列化/反序列化和真实
  YOLO callback，不构造模型或连接 ROS master。序号相差 50、双向回调
  错位均复现无输出。改为完全相同的归一化采集时间配对，原容量 4 的
  pending buffer/latest-complete 策略保留；各 topic seq 原样留作诊断。
- 完整帧去重同样按 stamp，不把相同采集因计数变化的重发当成新观测。
  不增加时间容差、阻塞等待、图像复制、发布线程或 topic；legacy seq=0
  的历史容差路径本轮保持兼容，正常 ROS publisher 从 1 开始计数。
- 另加同 topic 计数但不同时间的拒绝、同采集重发及计数重启测试；修正
  原静态 fixture 将不同 seq 误当成不同采集的断言，以真实不同 stamp
  验证不能混帧。最终三个目录联合 1035 passed、4 个 ROS 依赖弃用警告，
  13.42 s；git diff --check 通过。没有连接/重启 Go2，未提交。
- 本轮修复的是正常 ROS 传输下会直接阻断感知输出的身份错误，不宣称
  已完成逐帧权威位姿方案。sensor 仍先发图像再匹配遥测，而 YOLO 独立
  查历史；两端仍可能选到不同样本。后续帧上下文协议必须用采集 stamp
  绑定，不能再使用独立 topic 的 Header.seq 作为跨 topic 关联键。

### 逐帧遥测和标定的权威上下文（2026-09-09）

- Sensor 在本帧 telemetry_at 选择完成后，把同一份遥测快照交给
  capture-time TF 与新 `/physical_nav/capture_context` 发布；String JSON
  version=1 携带序列化 ROS stamp、原 receipt seq、匹配诊断及本帧内参、
  深度单位/外参和 frame 名。空位姿也是明确结果，不等后继遥测再改写。
- YOLO 正常模式把 RGB/depth/context 视为三路完整采集，按相同归一化
  stamp 关联。每路最多保留 4 个未完成项，另保留最新完整帧；后继半帧
  不抹掉已完成帧，旧上下文不能重新释放已消费采集。图像不再次复制，
  不增加 worker 或阻塞等待。逐帧数据齐备时不依赖最新 CameraInfo 或
  另一次独立 telemetry_at 查询，避免重连/回调顺序使标定和位姿串帧。
- 校验元数据版本、时间、内参、深度单位、frame 和结构，使用前核对
  RGB/depth 图像尺寸；非法上下文不放行本帧，后继完整帧正常恢复。空
  telemetry 仍进入已有 infer 的 2D-only 门控，不补最新姿态生成假 3D。
- 新版运行默认必须有 context；旧 ROS bag 可显式指定
  `--capture-context-topic ''` 使用兼容历史匹配。启动日志输出模式。
  这意味着部署时必须同时更新本地 sensor bridge 和 YOLO；新录制应
  包含 context topic，不能把元数据缺失时无输出误诊为 GPU 性能问题。
- 新增 16 项离线回归：三路全部 6 种顺序、context 落后一帧仍推进、
  没有匹配位姿时保持拒绝、无效/错时数据、容量、图像引用复用、分辨率
  错配及恢复；真实 sensor.publish 的测试核对 TF 与 context 取同一
  快照，测试中的 TF 发送使用替身，不冒充 TF 网络端到端验收。
- 7×2000 次小型元数据基准，序列化+callback 校验+三路关联+取 raw 的
  每帧均值中位数 56.24 µs（约 0.056 ms）。使用小型合成元数据和已有
  2×2 数组引用，不含完整图像处理、ROS 传输/调度、真实遥测体积或 GPU，
  不能据此承诺实机总延迟。联合三目录 1051 passed、4 个 ROS 依赖弃用
  警告，13.79 s；diff check 通过。未连接/重启 Go2，未提交。
- 后续仍需检查运行中相机安装外参/IMU 参数的一致性、周期 TF 是否会
  在相同时间覆盖捕获 TF、真实 context/RGB/depth 到达偏差及掉帧率。
  本轮保证的是传递同一选样与标定，不是 IMU/odom 硬件时钟精度或
  GMapping 墙面质量已经满足实机要求。

### 相机变换在 TF 与 YOLO 之间只计算一次（2026-09-09）

- 两项真实 infer 回归先复现：即使 telemetry/深度标定相同，YOLO 仍用
  自己的 camera_xyz/rpy 与启动时读取的 camera_imu 参数推导相机变换。
  故意给长生命周期 worker 保留旧高度/倾角/IMU 开关后，世界变换与
  sensor 发布的 TF 不同；同帧遥测本身不能排除空间变换不一致。
- 从 sensor TF 发布中提取相机分支构造器，每个采集只算一次。context
  发送它的 `depth_to_base`（parent/child/translation/quaternion_xyzw），
  随后的 publish_pose 复用同一个 TransformStamped 列表；周期 TF 没有
  预计算列表时仍按原路径构建。动态/静态发布选择与静态增量去重保留。
- 正常 YOLO 使用该变换与机身位姿合成世界变换，不再重复进行相机安装
  RPY、深度基线及杆 IMU 修正。检查 parent=base_link、child=本帧深度
  frame、四元数和位移形状/有限性；缺失、错误或不支持的变换触发已有
  2D-only 路径，不回退本地陈旧参数。legacy 输入仍支持旧计算路径。
- 合成结果维持 float32，避免元数据校验用的 float64 位移使整个点云
  lifting 意外升精度、增加内存与计算。实际 TF 参数值未调整。采集 TF
  同时使用本帧明确的 depth_to_color 标定，缺省空字典不再沿用旧基线。
- 10 项新增回归包含上述两个先失败的场景、TF/context 同一对象复用
  且仅算一次、错误父/子 frame、零/NaN 四元数、错误形状/溢出位移，以及
  live context 缺失变换不能静默走旧参数。geometry worker 使用替身以
  检查真实 infer 传入的变换，不将其冒充真实点云质量测试。
- 联合三个目录 1061 passed、4 个 ROS 依赖弃用警告，13.77 s；diff
  check 通过。未连接/重启 Go2，未提交。没有增加图像复制或等待线程。
- 周期 TF 与采集 TF 在消费者缓存中同时间戳冲突/历史更新的行为仍需
  单独验证；同一发送侧变换并不保证 TF 消费者最终采用它。硬件姿态
  精度、时钟同步、实际点云墙面质量和端到端延迟仍待实测。

### TF2 首次写入语义与冲突帧隔离（2026-09-09）

- 用本机真实 tf2_ros.Buffer(debug=False) 顺序写入同一 child/stamp 的
  x=1 与 x=2，TF2 输出 TF_REPEATED_DATA 并保留 x=1。故周期 TF 先占用
  时间戳后，采集 TF 再发“更正”不会改变消费者；YOLO 采用后者将与地图
  使用的变换不同。测试直接用本地 C++ TF2 缓存，不启动 ROS master。
- 发布端在已有 pose 锁内检查每个 stamp/child 的不可变几何签名；相同
  内容不重复发送，允许为同一采集补发尚不存在的 depth child。位置或
  旋转矛盾返回 conflicting_capture_tf_stamp，拒绝前不改动 odom/TF/
  标定缓存。四元数 q/-q 等价、近零数值抖动不产生伪冲突。
- 绑定历史最多 512 个时间戳，淘汰边界以前返回 expired_capture_tf_stamp，
  避免忘记本地记录后重写消费者仍保留的旧数据。签名只保留小数值元组，
  不保留图像/点云；历史捕获仍可在保留范围内按原时间发布，不倒退 odom。
- 调整采集发布顺序：相机分支仍只计算一次；先核定/发布 TF，再发布
  context，携带 capture_tf_status 与累计 capture_tf_conflict_count。
  冲突、过旧、空/非法采集位姿保留 RGB-D 和 2D，但不入点云 worker，
  YOLO 也不生成世界几何；不再让无匹配位姿的点云尝试使用其他历史 TF。
  正常新帧无额外线程/等待窗口，不改采集时间戳。
- 9 项新增测试覆盖真实 TF2 周期先到、机身/杆姿态冲突及下一帧恢复、
  相同数据去重、补发新 child、四元数等价、缓存边界；真实 sensor.publish
  →context→YOLO 回调在冲突/无位姿/坏位姿时不排点云，真实 infer 保留
  2D overlay 并跳过 geometry。此前联合 1070 passed、4 个 ROS 依赖弃用
  警告，13.88 s；后续小型诊断字段变更另跑专项回归，diff check 通过。
- 该绑定记录是本进程的发送记录，不是 TF 消费者收包确认。ROS 丢包、
  其他节点竞争发布同名 child、静态/动态 frame 跨进程切换仍需集成验收；
  不能据此宣称整个 TF 传输无冲突。若实测拒绝频繁，需要继续修源遥测
  时间戳/采集关联，不应放开门控去增加错误地图的“发布频率”。
  未连接/重启 Go2，未提交。

### 保留 sport-state 自身的样本时间（2026-09-09）

- 核对 Go2 源码：顶层 received_at 来自 on_sport 回调，on_low 电池更新
  存在独立 battery 字典，不重写这个姿态时间。相机帧附带的是最新姿态
  快照；若 sport-state 停顿，快照可能旧于仍在增长的相机 stamp。
- 本地 update_telemetry 原来用相机/发送 envelope 的 ROS stamp 登记
  姿态历史、ROS telemetry 与周期 TF。8 项测试复现旧样本 10.0 被登记
  为 10.5、重复相机帧不断刷新同一姿态年龄，以及坏/缺失时间仍建历史。
  因而此前统一的 0.15 s 容差仍可能被绕过，显示的匹配误差为虚假的零。
- 两个正常入口（相机包、独立 telemetry 包）现在显式传递源 envelope
  时间和归一化是否回退；用 received_at 加该 envelope 的 source→ROS
  偏移换算 pose_ros_stamp，历史、遥测 ROS stamp、周期 TF 一致保留
  样本年龄。不改变图像/采集 TF stamp，不延长匹配容差或等待新数据。
- 缺失/非法样本时间和不合理未来时间不建立姿态历史；已有有效姿态
  不被覆盖。新相机 seq 不再自动证明旧 received_at 是新的时钟纪元，
  只有归一化模块确认回退才允许回退水位。无源参考的旧调用接口仍保留
  历史兼容行为，不作为正常实物链路的时间保证。
- 11 项新测试还覆盖新 frame 携带旧快照时保护较新姿态、确认的时钟
  回退保留 0.1 s 年龄，以及真实 sensor.publish→context→YOLO 回调：
  10.1 秒图像接纳 10.0 秒姿态、10.5 秒图像拒绝同一姿态且不排点云。
  联合三目录 1081 passed、4 个 ROS 依赖弃用警告，14.06 s；diff check
  通过。未连接/重启 Go2，未提交；本轮不需要修改或部署 Go2 端源码。
- 这仅纠正应用层重标时间问题。sport-state received_at 仍不是 DDS
  原始测量硬件时刻，相机 stamp 仍主要来自读取后的主机时间。相机/IMU
  时间、DDS 排队、两个 envelope 流回退时偏移一致性及真实采集延迟还
  需继续核查和实测，不能据此宣称已实现完整硬件时间同步。

### IMU 异常样本不能污染持久估计状态（2026-09-09）

- 审计相机校正的新鲜度时，先复现源端状态污染：NaN/Inf 加速度使
  roll/pitch 永久 NaN；坏 gyro/timestamp 使 yaw NaN/Inf 或抛异常；
  正常样本随后仍受污染。零加速度被当作有效重力并推进参考校准。
  8 项新回归先失败；这是注入异常的离线复现，不表示实机已出现这些值。
- 向量/时间戳先检查形状可读性、数值有限性和非零重力范数，再更新
  滤波状态；gyro 与 accel 分量独立接纳，坏 gyro 不推进其时间基准，
  坏 accel 不推进参考校准。只用标准库 math，不加队列或线程。
- `_update_imu_orientation` 返回是否用了有效样本；真实 read 路径仅在
  有效时替换 latest_motion，全部无效时保留之前的观测和 received_at，
  不把旧姿态重标为新的 IMU 观测。网络库移到 publish 入口加载，使本机
  无 websocket-client 时可直接测估计器，不安装依赖或构造传感器。
- 14 项专项覆盖正常四元数契约、异常向量/时钟与恢复、独立有效分量、
  参考校准和真实 read（SDK 替身）。联合三目录 1095 passed、4 个 ROS
  依赖弃用警告，14.42 s；diff check 通过。7×10000 次合成有效更新的
  每次均值中位数 5.18 µs，不含 SDK/I/O/网络，也不是实机频率。
- 未修改 roll/pitch 的 alpha=.98、参考校准样本数或 yaw 积分策略。
  合成 0.2 rad roll 阶跃仍需 114 次有效 accel 更新达到 90%。这是
  后续响应优化的重要基线：实际耗时取决于有效而非配置的 IMU 频率。
  当前 roll/pitch 没有 gyro 快速传播；重复 motion 时间戳、gyro 有效但
  accel 停更时整体 received_at 仍刷新，以及独立分量年龄门控仍待审计。
- 本轮涉及 Go2 端脚本的本地副本，尚未部署；未 SSH、未连接/重启 Go2，
  未提交。不能把异常恢复回归等同于杆晃动补偿或实机墙面质量已验收。

### IMU 新样本去重与有效更新计数（2026-09-09）

- 10 项新测试先复现：同一 accel_timestamp_ms 可反复推进滤波/250 次
  参考校准；相同 gyro 时间戳仍被标成更新，迟到的 gyro 时间倒退基准，
  使后续 dt 被多算；新 gyro 携带旧 accel 时会重复算一次重力滤波。
- 加速度和陀螺仪各自使用有限非负 SDK 时间戳水位，严格新样本才可更新。
  无效向量不推进水位，首次 timestamp=0 允许。两个水位独立，不能因为
  一路更新就把另一路旧值重新计数。旧/重复/缺时间戳的双无效 poll 在
  滤波、三角函数和输出元数据分配前返回，不刷新 latest_motion。
- 新增 camera_imu.imu_valid_updates={accel,gyro} 累计计数，capture 日志
  同时输出，用于实测有效处理频率；不把 SDK 配置频率等同于实际被估计器
  使用的频率。计数随采集器实例重置，日志跨进程不能直接相减。
- 旧数学/异常恢复测试补上独立新测量的硬件时间戳，不通过允许未标时
  加速度绕过正常协议。新增 12 项专项含最初 10 个复现分支、初始零时间
  与重复 poll 无输出加工；原真实 read 替身测试补上重复 SDK 帧不换
  latest_motion。联合三目录 1107 passed、4 个 ROS 依赖弃用警告，
  13.64 s；最后补充断言另跑专项，diff check 通过。
- 7×10000 次合成 accel 更新，每次均值中位数新样本 3.93 µs、重复检查
  1.86 µs；无 SDK/网络/硬件，不是实机频率或端到端优化比例。
- 尚未调整 alpha=.98、校准阈值或 gyro yaw 积分上限；有效 accel 停更
  而 gyro 更新时整体 motion 收据仍会刷新，独立分量新鲜度仍需处理。
  真正的硬件时钟回退目前不会自动重置姿态/参考，需重建采集器，防止
  把单个迟到包当成重启。本轮未加入自动设备恢复策略。
  Go2 脚本只修改本地副本，未连接/部署/重启机器人，未提交。

### 重力滤波按真实样本时间保持响应（2026-09-09）

- 7 项新测试先失败：固定 alpha=.98 使 10/20/50/200 Hz 与 100 Hz
  下的每秒响应不同；不均匀采样也取决于收到多少次 callback，而不是
  经过时间。100 Hz 原基线通过，作为保留的默认响应。
- 新权重 alpha=exp(-dt/tau)，默认 tau=-0.01/log(.98)≈0.495 s，
  等价于配置 100 Hz 下的旧值。dt 来自通过去重门控的 accel SDK 时间戳，
  不用网页/图像频率。增加 --imu-gravity-tau-s，CLI 和构造器在打开
  传感器前拒绝非有限/非正配置；默认不是任意选取的更激进增益。
- 首次样本用名义 dt=.01 s；超过 .25 s 的缺口按一次名义样本恢复，
  不把长时间无观测当成充分稳定的重力证据而突然跳变。记录
  imu_gravity_dt_s/imu_gravity_gap，后续正常采样继续按真实 dt。
- 合成已校准 0.2 rad roll 阶跃的 90% 响应：10 Hz 11.4→1.2 s；
  20 Hz 5.7→1.15 s；50 Hz 2.28→1.14 s；100 Hz 1.14→1.14 s；
  200 Hz .57→1.14 s。最后一项说明时间常数也不会因回调变多而过快。
  这些是输入有效频率的控制实验，不是实机测得的频率或墙面质量。
- 13 项专项覆盖五个采样率、非均匀时刻、断流恢复、自定义 tau、构造器
  与 CLI 提前拒绝坏参数。联合三目录 1120 passed、4 个 ROS 依赖弃用
  警告，13.76 s；最终新增的恢复断言另跑专项，diff check 通过。
  7×10000 次含合成向量构造的有效更新，每次均值中位数 4.76 µs；
  与前轮基准输入不同，不作严格性能比值，也不包含 SDK/I/O。
- 没有加入等待窗口/线程，没有改变 250 个有效 accel 的参考校准条件。
  因此启动校准在低有效频率下仍可能很久，本轮响应数据不能当作就绪时间。
  roll/pitch 仍未加入 gyro 快速传播，运动加速度/噪声、独立 accel 新鲜度
  和实机参数选择仍需继续处理。只改本地 Go2 脚本，未部署、重启或提交。

### 独立 IMU 分量年龄及周期 TF 补漏（2026-09-09）

- 源端分别保留最近有效 accel/gyro 的主机收据时间，发布快照复制字典。
  新 gyro、重复 accel 不再刷新旧重力校正的年龄；真正新分量缺少合法
  收据时记录未验证，不能继承上次收据伪装正常。
- 采集端按原 Go2 帧时间检查必要分量，默认 yaw 关闭只要求 accel，
  yaw 打开还需 gyro，沿用 telemetry_max_delta_sec（默认 .15 s）。
  过期/非法时保留图像和 2D，拒绝该帧世界点云、3D 检测与采集 TF；
  context/YOLO 报告 camera_imu_match 状态和有符号年龄。新 accel 恢复
  后可正常发布，不新增等待、队列、线程。
- 继续审计复现周期 refresh_tf 漏洞：2 项测试先失败，旧 accel 或非法
  收据仍发布新相机 TF。补齐源时间检查，只跳过相机分支，不回退名义
  外参；底盘 odom/TF 保持有效发布。1000 s ROS stamp 与 10 s Go2
  时间的合成偏移不会误算成 IMU 年龄，周期重复回调不会重发旧姿态。
- 新增 15 项专项：独立分量、新旧快照隔离、缺失/NaN/Inf/未来时间、
  可选 yaw、源→capture context→真实 YOLO infer 的 2D 保留/3D 隔离、
  后续恢复与周期底盘解耦。联合三目录 1135 passed，13.61 s，4 个 ROS
  依赖弃用警告；diff check 通过。此前采集门控基线 1132 passed。
- 旧协议无分量时间显式标 legacy_unverified 并兼容放行；disabled、
  unavailable、calibrating 保留旧行为，不代表重力新鲜。严格保证需要
  Go2 与本机 bridge 同步部署。收据仍非曝光/测量硬件时刻；跳过相机 TF
  不清除消费者已有缓存，不能据此保证任意 latest TF 查询都是当前姿态。
- 当前剩余重点：250 有效样本参考校准在低频下耗时，roll/pitch 尚无
  gyro 快速传播，运动加速度抑制、硬件时钟回退和真实 OCC 数据验收。
  本轮未 SSH、部署、连接/重启 Go2 或提交，不宣称实机频率已达标。

### IMU 参考校准按连续静止时间收敛（2026-09-09）

- 离线复现固定 250 样本门槛的频率依赖：有效 accel 为 10/20/100/200 Hz
  时分别约需 24.9/12.45/2.49/1.245 s。200 Hz 时低通尚未收敛，合成
  1 rad 静态倾角却固化约 0.864 rad 参考；持续摆动或旋转也没有静止门。
- 改为 D435i accel 硬件时间上的连续 2.5 s 候选窗口，与网页、RGB 帧率
  和 callback 次数无关。相邻 accel 间隔超过 0.25 s、模长偏离
  9.80665 m/s² 超过 0.75 m/s²、当前 gyro 模长超过 0.08 rad/s，或
  原始重力方向相对窗口锚点跨越 0.035 rad，都会安全重开窗口。
- roll/pitch 参考由窗口内原始单位重力向量平均后计算，不再平均从零启动
  的低通状态；锁定时令滤波值等于参考，首次 correction 为零，避免动态
  TF 突跳。参考锁定后保持不变。修正角使用包角差，跨 ±π 不会产生整圈
  跳变。输出 imu_calibration_status/elapsed_s，Go2 五秒诊断同步打印。
- 20 项新增/调整专项覆盖 10–200 Hz 同时长、1 kHz 短 burst、非均匀
  时间、长 gap、摆动/旋转/异常模长、阈值内外、小噪声和抖动、恢复、
  锁定不可变及角度包络。四个 IMU 专项共 59 passed；联合三目录
  1155 passed、4 个 ROS 依赖弃用警告，14.07 s；py_compile 与
  diff check 通过。
- 7×10000 合成更新（含 Python 字典构造）中位数：先校准后锁定7.15 µs，
  已锁定6.78 µs；不含 SDK/I/O/网络，输入和既往基准不完全相同，不能
  当作实机性能增益。校准不阻塞 RGB-D 发布。
- 限制仍很明确：启动参考会把启动时已有的静态杆侧倾当作零点，只能校正
  后续相对晃动。要同时消除长期侧倾，需要一次已知直立参考，或将 D435i
  与 Go2 机身 IMU 做外参标定后估计相对姿态。匀加速且模长接近 g、无
  gyro 时绕重力轴旋转也可能伪装静止；实物阈值和墙面质量仍须真实数据
  验收。本轮未连接、部署或重启 Go2，未提交。

### D435 RGB-D 采集、编码、发送三段解耦（2026-09-10）

- 原 Go2 循环在每次 `wait_for_frames` 后同步执行 1280×720 JPEG、
  848×480 16-bit PNG 与网络封包/发送；任一阶段变慢都会停止继续排空
  RealSense。现在采集取得 RGB-D 后立即复制 SDK 缓冲，并提交容量为 1
  的 latest-only raw slot；独立 codec worker 使用持久双线程 executor
  并行 JPEG/PNG，编码结果再进入 latest-only encoded slot。慢编码或慢网
  只替换过时图像，不形成无界队列，也不直接阻塞下一次相机读取。
- 本机高熵随机数组的诊断基准中，一组 1280×720 RGB + 848×480 depth
  编码中位数 57.68 ms、最大 58.88 ms；两张数组复制中位数 0.261 ms、
  最大 5.375 ms，共 3,578,880 bytes。它只测本机 OpenCV/内存，不能
  外推 Jetson、真实图像压缩率、USB 或 Wi-Fi 带宽。
- sender 在成功发送后用全连接代际共享的 seq 去重；断线重连不会重发
  已成功发送的最后一帧。默认 `--max-frame-age-s 0.5` 拒绝已经过时的
  编码结果。连续 stale 原来不推进 deadline，会形成紧循环并饿死 asyncio
  取消/其他任务；现按下一发布周期继续并显式 yield，stale 日志每秒限流。
- `sensor capture` 分开输出 attempt/read Hz、失败数、连续失败、最后成功
  年龄、read/copy 耗时以及 codec 累计 processed/error/replaced/alive；
  `sensor codec` 输出实际编码率/均值/峰值；`sensor transport` 的
  packet 时间包含 base64、JSON/zlib，并新增采集到发送完成的 delivery
  age、stale 数和 wire Mbps。持续采集错误不再以 10 Hz 刷屏。
- 生命周期不再仅依赖 asyncio done callback。`publish` 的直接 finally
  按 source.stop、capture join、codec close、executor shutdown 回收；
  stage/thread 构造失败也走幂等清理。D435 pipeline.start 后若 align、
  profile、intrinsics/extrinsics 初始化异常，会在构造器内部 stop pipeline。
- 8 项专项连续跑 3 轮均通过，覆盖 codec 阻塞只保留最新 pending、错误
  后恢复、取消无残留线程、stale 不发送、断线不重发、复用 SDK backing
  buffer 时异步编码仍看到不可变快照，以及 D435 构造失败回收。联合三
  目录 1163 passed、4 个 ROS 依赖弃用警告，14.50 s；py_compile 与
  diff check 通过。
- 仍未连接、部署或重启 Go2，未提交。同步 `wait_for_frames` 每个 candidate
  至多取得一个 accel/gyro，尚不能证明 100/200 Hz motion 完整消费；下一
  阶段需用有界 SDK frame_queue + 专用 drain/FIFO，记录 frame number gap、
  实际 Hz、overflow 与图像曝光时刻的因果姿态。Go2 端当前仍是一次性
  nohup，无自动重启；codec native hang 的强制回收和 USB 拔插恢复需结合
  外层 supervisor 后再做实机故障注入，不能用离线测试宣称已闭环。

### D435i 有界 SDK ingress 与连续性门控（2026-09-10）

- 上一节遗留的同步采集已替换为
  `pipeline.start(config, rs.frame_queue(capacity, keep_frames=False))`。
  librealsense callback 只写 C++ 有界队列，已有 `d435i-capture` 线程作为
  唯一 drain owner：motion 逐样处理，RGB-D 复合帧 latest-only；Python
  不进入 SDK callback，也未额外增加一条常驻线程。
- 默认 queue capacity=128，并从本地总启动脚本一路传到 Go2 参数。采集端
  按 accel/gyro/color/depth 独立记录 frame number 与硬件 timestamp：
  dequeued、unique、missing、gaps、duplicate、late、timestamp regression、
  effective Hz 和 continuous duration。queue high-water 只是 dequeue 后的
  近似值，是否溢出以每流 frame-number gap 为权威。
- 每个 RGB-D 包只取一次不可变的相机 IMU 快照，顶层 `camera_imu` 与
  telemetry 内嵌字段共用同一对象。姿态历史按 SDK timestamp/domain 有界
  保存，depth 帧只选择相同时间域且不晚于曝光的已有状态；不会用回调时刻
  的“最新姿态”改写旧图像。host 在原分量年龄门控之外，要求必要 IMU 流
  连续健康至少 0.5 s；发现 gap 后保留 RGB/2D，但世界点云、3D box 与采集
  TF 等到新连续窗口恢复。
- 标准 pipeline 的同步视频保持原生 frameset。额外兼容 individual
  color/depth 的后端时，只在相同且非空 timestamp domain 内配对，阈值为
  `delta < min(50 ms, 500/min_fps ms)`；单流只留最新候选，旧 orphan 明确
  计数。native frameset 到达会清空旧 single pending，防止跨交付模式错配。
  `align color` 因 `rs.align` 必须接收原生 composite，应用层 pair 会
  fail closed；当前实物 `align depth` 不经过此限制。
- Source 新增 closed event：pipeline.stop 唤醒阻塞读取后立即退出，重复
  close 不会二次 stop。构造后初始化失败仍回收已启动 pipeline。
- 41 项直接相关回归通过，覆盖混合 200 gyro + 100 accel + 10 RGB-D、
  exact-once、latest-only、缺口/重复/迟到、曝光因果选样、10/30 Hz 单帧
  配对、未知时间域、single/composite 切换、align-color fail-closed、
  capture/codec/send 解耦及 IMU host 门控；2 个警告来自 ROS Noetic
  actionlib 的既有转义弃用。
- 联合 semantic mapping、semantic decision 与 physical nav 三目录回归为
  1181 passed、4 个 ROS Noetic 既有弃用警告，15.06 s；py_compile、
  shell syntax 与 diff check 通过。
- 本轮仍只完成本地离线验证，未连接、部署或重启 Go2，未提交。真实设备
  仍需确认 pipeline queue 的视频事件确为 frameset、accel/gyro 分别达到
  100/200 Hz 且无 frame gap，并测默认容量的高水位。当前姿态估计仍是
  gravity roll/pitch + optical gyro-z 相对 yaw，不等于完整 D435i/机身外参
  融合；跨 stream 极端乱序的严格全局重排和曝光 watermark 也尚未实现。

### D435i 曝光身份与跨主机时间基准（2026-09-10）

- 图像时刻以 `rs.align` 之前的原始 depth frame 为权威。采集线程选定最新
  RGB-D candidate 后立即固定 dequeue wall/monotonic，再读取 depth 的
  `SENSOR_TIMESTAMP`（曝光中点，µs）、`FRAME_TIMESTAMP`（读出起点，µs）
  与 SDK `get_timestamp()`/domain；align 生成的帧不能替换原始采集身份。
  当 SDK domain 为 GLOBAL_TIME 时，用同一原始帧的
  `global frame stamp + (sensor timestamp - frame timestamp)` 恢复全局曝光
  中点；hardware clock 则保留其设备时钟曝光中点。缺少元数据时明确降级为
  frame readout、host time-of-arrival 或 unknown，不把它们标成曝光时刻。
- D435i hardware clock 没有 Unix epoch。Go2 端 `_DeviceClockMapper` 在同一
  canonical domain 内维护 `host monotonic - device time` 的低延迟包络；至少
  累计 8 个样本且观测跨度达到 0.5 s 才令 `source_epoch_valid=true`。成熟后
  只以最高 250 ppm 跟踪包络的缓慢漂移；突然高于既有模型 5 ms 的样本视为
  USB/SDK backlog，不进入 clock anchor，而是按已学习 rate 进入 holdover。
  lifetime minimum、recent minimum、rate、anchor age、holdover age、残差与
  queue excess 均保留诊断。预热前发送 stamp 使用 host read fallback；设备
  时间回退会清除旧 epoch，未知 domain 和明显不合理的 host-epoch 值均
  fail closed。
- 本机 bridge 不能假定 Go2 与本机 wall clock 数值相同。每次 WebSocket
  transport generation 在第一张大 RGB-D 之前按顺序接收带发送时刻的
  `hello + 2 clock_probe`，本机 `_SourceEpochMapper` 用三次小包收据的最小
  `local receive wall - Go2 send wall` 建立源到 ROS epoch 的偏移。传感器
  大包的到达时刻不参与拟合，否则 codec、Wi-Fi 或 socket backlog 会被
  吸收到时钟偏移而让旧帧伪装成新帧；重连代际彼此隔离。成熟映射最高以
  500 ppm 跟踪缓慢漂移，正残差超过 20 ms 时保持既有低延迟模型并进入
  holdover。hello/probe/telemetry 同时携带 Go2 wall 与 monotonic；接收端用
  Go2 自身的 wall-minus-monotonic phase 识别墙钟跳变，TCP head-of-line 延迟
  不参与该判断，且连续两包确认后才重置 epoch。
- 两级预热均采用 fail-closed。Go2 设备时钟映射未成熟时不声称绝对曝光
  epoch；本机对带 transport tag、但三次握手尚未成熟的生产帧仍可暴露
  RGB/depth 与诊断，却不使用该帧更新采集遥测、TF 或世界点云。正常新客户
  在首个 RGB-D 前已经发送三次小包，因此该门控不增加逐帧等待或线程。
- 该跨主机算法只有单向发送/接收时刻。观测量等于真实 wall-clock offset
  加单程网络延迟，因此最小观测偏移仍只是时钟偏移的上界；无法观测的最小
  网络延迟仍包含在结果中。低延迟 anchor 与 ppm 有界 holdover 可阻止突发
  backlog 被吸收，并处理常见缓慢漂移，但无法区分“与时钟漂移同斜率且长期
  持续”的单向延迟变化，不能给出 PTP 级绝对同步保证。若必须严格分离时钟
  偏移和链路延迟，应使用 chrony/PTP 或带四时间戳的往返探测。
- Go2 原始 `capture_timing` 随 RGB-D 协议进入网页 runtime state 与 raw
  recorder manifest，保留曝光身份和设备映射。本机新增的跨主机映射、ROS
  最终 stamp、收包时刻、门控原因及 transport generation 只写入 ROS
  `/physical_nav/capture_context`；网页/raw manifest 不能被误解为已含本机
  epoch 修正。对应离线回归为 `test_go2_exposure_timing.py`、
  `test_source_epoch_mapping.py` 和 `test_physical_raw_recorder.py`。
- 本节尚未在真实 Go2/D435i 上部署或验收，不能据此宣称曝光 metadata 在
  当前固件完整可用、USB/Wi-Fi 延迟已消除、OCC 时间同步已改善或发布频率
  已达标。实机需要同时核对 metadata/domain、两级 warmup 状态、source 到
  ROS 偏移、transport age、TF/点云采集 stamp 和 OCC source age。
- 本轮最终本地联合回归为 1244 passed、4 个 ROS Noetic 既有弃用警告，
  15.73 s；另已通过相关 Python `py_compile` 与 `git diff --check`。其中包含
  正负 100 ppm 一小时级时钟漂移、持续 400 ms backlog holdover、source
  wall-step/HOL 区分、连续 stamp 回拨门控和跨 SDK domain 的 IMU 因果匹配；
  这些均为合成离线验证，不替代下一轮实机时间域与 OCC source-age 验收。
