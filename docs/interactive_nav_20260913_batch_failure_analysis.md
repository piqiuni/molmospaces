# 2026-09-13 十场交互导航失败分析

## 范围与结论

分析批次：`/home/ldl/outputs/interactive-nav/m1_fallback_h1_h10_1500_20260913_202839`。
本轮只分析保存的日志、地图、图像与当前代码，并做无仿真的内存复现；没有修改运行逻辑或重跑仿真。

- 十场 `mllm_metrics.jsonl` 均没有对象交互属性 M1 请求，只有房间属性与 subgoal 选择请求。
- 十场 `force_interaction_events.json` 的 `events` 均为空。因此应区分交互前导航/观察失败与物理动作失败；本轮没有实际物理交互可供判定成功或失败。
- 实际完成步数：H1 1500、H2 488、H3 310、H4 1414、H5 1500、H6 373、H7 463、H8 1500、H9 1500、H10 1500。
- H2/H3/H4/H6/H7 以 `semantic_mission_no_progress` 提前结束。脚本退出码 0 不代表任务成功或完成了 1500 step。

## 1. M1 配帧回归阻断所有对象请求

[image_frame_pairing.py:117](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/semantic_mapping_py_pkg/image_frame_pairing.py:117) 用 `capture_step == RGB header.seq` 筛选图像，然后要求时间戳精确相等。

但是本机 [rospy/msg.py:145](/home/ldl/conda_envs/ros-noetic/lib/python3.11/site-packages/rospy/msg.py:145) 会在序列化时覆盖 `header.seq`；[topics.py:1065](/home/ldl/conda_envs/ros-noetic/lib/python3.11/site-packages/rospy/topics.py:1065) 使用的是每次发布递增的消息计数，不是仿真 step。手动赋值 `header.seq = step` 不能维持该约定。

H1 记录中，step29 的 `latest_rgb_step_seq=31`，step923 为925。当前逻辑可能先选中较早帧，再因 stamp 不符拒绝。十场均持续记录 `image_stamp_mismatch`，而不是 M1 HTTP 请求全部超时。

无 ROS master 的最小序列化复现：给 Image 设置 seq923，再由 `serialize_message(..., 925, message)` 序列化，实际 header.seq 变成925，正确图像被当前配对器拒绝。

另一个独立缺陷是时间戳精度：[realtime_gt_observation.py:602](/home/ldl/molmospaces-exp-setting/molmo_spaces/policy/learned_policy/realtime_gt_observation.py:602) 只携带浮点秒，但配对器将其还原为整数纳秒并精确比较。构造 `(1789304037,594208001)` 后，经浮点路径得到 `(1789304037,594208000)`，同帧也可能不相等。

配帧失败在 [interaction_attribute_inference_node.py:524](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/interaction_attribute_inference_node.py:524) 提前返回，尚未进入对象请求队列。执行器最终生成 `interaction_observation_timeout`，掩盖了请求根本未发出的事实。

特别注意：[behavior_execution.py:3692](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_decision_py_pkg/behavior_execution.py:3692) 在超时反馈中直接填 `is_currently_visible=false`，不能据此推断相机没有看到对象。H6 step222 的原始门 bbox 为 `[368,198,609,514]`，可见像素62973，正是反例。

## 2. 失败兜底没有覆盖主要执行分支

[behavior_execution.py:2177](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_decision_py_pkg/behavior_execution.py:2177) 要求行为为 `INTERACT`、标记物理到位，并明确排除：

- `observation_only_reobserve`，即门的 NAVIGATE 重新观察。
- 两阶段容器 staging 与 `container_pre_action_observation`。
- `drawer_pre_action_observation`。

所以 `interaction_fallback_on_m1_failure=true` 并没有实现所有“已到交互位置但 M1 无结果”的兜底。容器外侧观察位姿与真实物理交互位姿必须先区分，不能直接将所有 staging 到位都视为可执行物理动作。

## 3. 房间分割稳定逻辑冻结

H1 的 room 标签在真实采集 step275 后不再变化，H9 在step102后不再变化，持续至1499。对保存的全部 room PNG 检查，冻结后图像哈希和栅格几何均保持不变，但发布与 OCC 更新仍在继续。

| 证据 | H1 | H9 |
| --- | ---: | ---: |
| 冻结标签有效格数 | 7255，即 ID1:7169、ID124:86 | 3169，全为 ID1 |
| step501 raw free 格数 | 10279 | 6899 |
| 同冻结状态及 step501 地图，内存 force_stable 输出有效标签格数 | 10186 | 6875 |

故障链：

1. [room_segmentation.py:812](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/semantic_mapping_py_pkg/room_segmentation.py:812) 优先匹配已经发布的 stable ID，没有优先继承待确认候选 ID。
2. 一个旧 ID 一轮只能分配给一个组件；另一个分裂组件每轮领取递增的新 ID。
3. [room_segmentation.py:430](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/semantic_mapping_py_pkg/room_segmentation.py:430) 要求候选具体 ID 一致率至少97%，连续三帧才能替换旧图。
4. 新 ID 不断变化，稳定计数始终重置为1，整张旧标签图被不断返回。

用保存地图重复调用原分割函数，H1 step277 的候选含1181格新区域，一致率83.7148%；H9 step104含240格新区域，一致率92.4528%。空间分割已经产生新区域，卡住的是 ID 与时间稳定确认。

原始冻结证据：[H1 map_manifest:9324](/home/ldl/outputs/interactive-nav/m1_fallback_h1_h10_1500_20260913_202839/house_1/debug/raw/map_manifest.jsonl:9324)、[H9 map_manifest:1641](/home/ldl/outputs/interactive-nav/m1_fallback_h1_h10_1500_20260913_202839/house_9/debug/raw/map_manifest.jsonl:1641)。部分 map_manifest 的 step 标签晚一帧，此处采用时间戳对应的真实采集 step。

两场 raw OCC 均为384x384、分辨率约0.1m、origin(-20,-20)，map到odom变换为零；未发现这里的坐标错位或 ROS master 串扰。H9 的分割冻结明确早于后续 move_base 崩溃。

## 4. 逐场时间线

### H1

- Door1：step643选择 `reobserve_portal`，923到观察等待，971重试，1019失败。总376步，约557秒；其中280步导航、96步观察等待。不是开门动作失败，动作未发出。
- Door2：1020选择、1189等待、1237重试、1285失败，合计265步，约403秒。返回了 `interaction_observation_unresolved` 执行失败，但没有 M1 状态结果，所以门仍 unknown。
- 新房间 OCC 不进入分割图，原因是上面的稳定标签冻结，而非视频没刷新。

### H2

- step34开始抽屉接近；274距目标0.105m，但 yaw误差1.003rad，未满足0.20rad要求；275超时换候选。
- 288距新目标0.063m，yaw误差1.677rad；后续旋转伴随位置漂移，487距目标0.3108m，最终 `interaction_approach_options_exhausted`。
- [联合到位判定](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_decision_py_pkg/behavior_execution.py:239) 要求位置和朝向同时满足。平面上看起来到点，不等于通过最终位姿条件。
- 原始 anchor 优先几何距离与角度；[anchor_priority:541](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_decision_py_pkg/behavior_execution.py:541) 对 AABB fan 偏好更小的物体表面间距。[预检:12191](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_behavior_executor.py:12191) 只验证全局 make_plan 和终点误差，不负责优化局部障碍净空。这解释了为什么没有在选择时主动将目标向更宽裕的位置调整。
- 尚未确定 H2 DWA 摆动的唯一底层原因，不能直接认定为 inflation 过大。

### H4

- 首轮前三个近端候选在保存 local costmap 中值为99；step53/63/72/82反复报 `valid control could not be found`。全局预检通过不保证局部终端位姿可执行。
- 换更远 anchor 后179进入观察，227/275/323再次等待，371以 `container_m1_evidence_inconclusive` 失败；四轮只记录一个观察视角。
- 第二轮788开始，1091观察，1139因 `container_m1_evidence_inconclusive_no_alternate_viewpoint` 失败。没有实际开冰箱动作。
- 本轮启动参数：local inflation radius=0.45m，cost_scaling_factor=10.0，robot_radius=0.25m；global inflation同为0.45m。gmapping的0.1m是另一参数，不是local inflation。见 [roslaunch.log:558](/home/ldl/outputs/interactive-nav/m1_fallback_h1_h10_1500_20260913_202839/house_4/roslaunch.log:558)。
- 未发现 H4 move_base 进程崩溃；不能声称缩小 inflation 一项即可解决全部卡住。

### H5

- Door1：277选择、323等待、371第二轮、419失败。总142步，约216秒，其中96步纯观察等待。
- 冰箱：537开始、731等待；779/883/982回到接近，835/934/1034再次等待，1082失败。共545步、约664秒，其中192步是四轮等待，只有一个记录视角，未执行物理动作。
- 未开冰箱时内部对象已入图属实：Tomato在capture step501首次出现，bbox `[306,178,778,178]`，仅1像素高但473像素；Lettuce在654以21像素出现，Egg在690以22像素出现。
- 原始 RGB：[step501](/home/ldl/outputs/interactive-nav/m1_fallback_h1_h10_1500_20260913_202839/house_5/sim_step_frames/step_000501.png)；原始语义输入：[manifest:502](/home/ldl/outputs/interactive-nav/m1_fallback_h1_h10_1500_20260913_202839/house_5/sim_step_frames/manifest.jsonl:502)。
- 普通物体接受极薄分割片段后就获得完整 GT 类别和3D box，随后 [interaction_graph_store.py:3145](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/semantic_mapping_py_pkg/interaction_graph_store.py:3145) 按空间包含建立 contains 边，不以成功开门为前提。
- 已证实内部知识提前暴露、证据阈值过宽；尚未确定像素来自真实缝隙、渲染伪影或其他几何/对齐问题。不能仅据此断言无像素依据的穿墙透视，也不能把几何 contains 当成内部已完成探索。

### H6

- Door1在130开始重观察，220 move_base明确 `SUCCEEDED/Goal reached`；268重试，316观察失败，373 no_progress结束。
- 90步导航加96步等待，未发出开门命令。门在视野内也无法绕过配帧故障；NAVIGATE观察分支不适用现有物理fallback。

### H7

- 183冰箱观察超时回到接近，随后 make_plan服务断连，move_base SIGSEGV(-11)重启。
- 186重启后的目标被取消并进入 RECALLING(7)，未正常终结。186之后机器人位置、朝向不再变化。
- 284至461累计8次frontier和99次门重观察失败，均被 `move_base_successor_not_quiescent` 阻挡；首个失败的 `attempted_goal_count=0`。后续goal尚未真正派发。
- [executor:8303](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_behavior_executor.py:8303) 将 RECALLING视为活跃目标；[executor:9434](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_behavior_executor.py:9434) 在预检和派发新目标前等待旧目标静默，失败后回到选择循环。
- 后段仍有24条非空global plan消息，但没有local plan、没有移动，因此不是严格意义上所有path消息都为空。
- SIGSEGV发生在anchor make_plan预检窗口；无C++调用栈，尚不能确定崩溃内部原因或认定为DWA reconfigure竞态。

### H8

- step650距实际目标约0.085m，但yaw误差约0.738rad，尚未完成最终朝向。
- 662取消当前goal并进入观察，这是到位后的正常阶段切换，不是当场放弃抽屉任务。
- 710/758/806仍是同一 `decision_000005`，候选index在15、16间切换；这些候选都落在当前位姿容差内，实际位姿710到854保持不变。
- [executor:9272](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_behavior_executor.py:9272) 的 `already_at_verified_approach_pose` 直接放行，[重选:9575](/home/ldl/molmospaces-exp-setting/Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/semantic_behavior_executor.py:9575) 又允许选回高优先级旧anchor。候选index不同不代表有效观察视角不同。
- 854四轮M1等待失败，855才真正切换到EXPLORE；不是650就被另一个任务抢占。

### H9

- 房间分割从102冻结，独立于交互失败。
- 冰箱三轮从145/477/884开始；297/664/1110进入观察，345/712/1158重试接近。
- 351与1166失败为 `rear_goal_local_costmap_stale`，约10.9秒旧，允许值0.75秒；720报 `container_m1_capture_dwa_reconfigure_update_failed`。
- move_base发生三次SIGABRT，时间与上述观察后重新接近窗口接近，导致局部地图与服务不可用。没有调用栈，尚不能将native崩溃归因到某一reconfigure操作。
- Door1在1167选择、1223等待、1271重试、1319失败；Door2在1320选择、1417等待，随后到达horizon。仍未执行物理交互。

## 5. 建议修复顺序与验收

1. P0：修复跨topic图像身份协议。使用不被ROS重写的capture身份和整数secs/nsecs；兼容旧浮点时间戳时明确其精度，不能回退到最新图。增加真实rospy序列化、异步乱序和同帧RGB/bbox集成测试。
2. P0：请求生命周期区分配帧失败、未入队、已派发、模型超时和已返回；不要把无结果伪装为不可见。落实物理到位后的兜底，明确观察staging和action pose的差别。
3. P0：修复房间候选ID跨帧继承，确保冻结地图样例中的新增/分裂区域三帧内可确认；不要靠每帧force_stable关闭稳定机制。
4. P1：anchor选择结合新鲜local costmap、机器人footprint和最终转向扫掠净空，不只看全局路径及物体AABB间距；按实际位姿/视角变化去重M1重试，阻止旧anchor循环复用。
5. P1：move_base重启后显式重建action客户端的目标归属与连接状态；旧RECALLING超时应进入受控重连/故障状态，停止重复选择无望派发的subgoal。保留native崩溃调用栈，独立定位C++错误。
6. P1：收紧普通物体的有效可见证据，检查极薄mask、投影/RGB一致性与多帧持续性；区分推测contains、可靠看到内部与交互后探索完成。不能一概禁止关闭容器周围所有合法可见物体。

先做保存数据的回放与最小ROS链路验证，再按同配置复测异常场景。验收需包含真实M1请求及输入图像、真实物理交互事件、room标签更新、无永久goal阻塞；不能以脚本退出0、单元测试通过或视频生成成功代替交互性能验收。

## 6. 修复进展

2026-09-13 已修改代码，以上分析保留为原始故障记录：

- 配帧不再比较 capture_step 与 ROS header.seq；精确 secs/nsecs 贯通 GT、M1、建图和录制。旧浮点时间戳只允许浮点舍入量级误差，歧义帧拒绝配对，不回退到最新 RGB。
- 检测器通过同帧 step_sync 时间戳确定真实 capture_step；M1 公开配帧阶段与入队状态，超时反馈不再编造不可见。
- 记忆门携带 INTERACT 几何合同；容器 M1 失败后使用既有 action pose 映射，经物理到位检查才调用动作，不伪造已接受的 M1 证据。
- 共享与独立 capture 阶段统一记录采样视角，按位姿差异排除候选别名，避免相邻编号循环。
- 房间候选继承上一帧未确认 ID，并将合并事件提交与稳定标签提交绑定。
- 普通物体过滤极薄/稀疏 mask，投影可见比例改用有效 mask 面积及投影交集，未用容器开关状态屏蔽真实可见物。
- 导航客户端新增有限次数的受控恢复，须确认旧进程退出或服务端旧目标终止，再接受新目标；恢复耗尽报告运行级失败，避免持续选择却不能派发。
- 容器 anchor 加入新鲜 local costmap 的中心和机器人占用检查；中心拒绝99/100，完整 footprint 仅对100与未知格做检查，避免对99膨胀层重复叠加机器人半径。local inflation 保持0.45m。

H4/H5/H8/H9 保存的真实 INTERACT 候选已做状态机回放：一次 M1 超时后进入
physical-action 导航，到位回调后生成 interact 命令。这是逻辑回放，不是物理仿真成功记录。
回归命令见 [test.md](/home/ldl/molmospaces-exp-setting/test.md:40)。尚未重跑长时间仿真。

修复后用原始冻结标签、截至 step501 的门户观测与保存 OCC 做正常三帧内存回放：
H1 有效标签格数为7255、7255、10144；H9为3169、3169、6875，确认计数均为1、2、3。
这次没有使用 `force_stable=True` 绕过确认机制。与分析阶段使用的门户时间窗口不同，
不将H1的10144与前文force_stable示例10186当作同配置性能比较。

轻量联合回归已通过722项，包含真实rospy序列化且无跳过。
仍需实际ROS仿真验证动作成功率、轨迹和吞吐；native SIGSEGV/SIGABRT的内部原因
尚无调用栈证据，当前修复的是恢复与阻塞控制，不能声称消除了原生崩溃。

## 7. 2026-09-14 对500步重跑的逐场复核

本节对应新批次 `/home/ldl/outputs/interactive-nav/batch_fixes_h1_h10_500_20260913_225013`，不是前文1500步批次。
H1/H2/H5/H6/H7/H8/H9/H10各500步，H3为382步，H4为391步；后两场为 `semantic_mission_no_progress`。
本次仅分析既有视频、无损costmap、逐步状态与代码，未改执行代码或配置，未重跑仿真。

### 7.1 证据口径

- 每场 `debug/raw/step_boundaries.jsonl` 的 `step_index` 对应视频帧；以 `semantic_execution_state.effective_goal_xyyaw` 核对正在执行的接近位，而非只看M2最初发布的 `goal_xyyaw`。
- `debug/events.jsonl`、`debug/move_base_status.csv`、`roslaunch.log` 记录命令接收侧时刻；异步回调可与相机帧或backend的 `step` 相差1步。
- 仿真结束后ROS仍会短暂运行。最后一步编号下的超时、重选不能当作视频内发生的多轮导航。H4末尾的 `navigation_step_sync_stall` 晚于mission完成请求，不是提前结束的首因。
- `debug/raw/map_manifest.jsonl` 的PNG按 `png_value_offset` 还原原始cost值，没有从视频颜色反推数值。使用逐步记录引用的local costmap快照；与executor读取快照可能相差一次地图发布。
- 分析截图位于 `/home/ldl/molmospaces-exp-setting/outputs/analysis_batch500_20260914/H{2,4,5,9,10}_timeline.png`。

### 7.2 H2：接近位预检切换和末端平移不收敛，不是重复M1

抽屉目标始终是 `chestofdrawers_fdf8ab53e7ce731f1fcbde4be80ad603_1_0_2`。

| 步数 | 实际过程 |
| --- | --- |
| 35 | M2选中抽屉，原始接近位index0为 `(2.906,0.836,-1.571)`。执行器同批预检14个位姿，选中index2。 |
| 36-214 | 实际导航到index2 `(2.735,0.819,-1.309)`。最低位置误差仍为0.1711m，没有满足0.15m到位条件。 |
| 160-214 | 位置主要停在约0.17-0.18m误差，DWA局部轨迹多次只有角度变化，没有有效平移收敛。 |
| 214-217 | `semantic_subgoal_no_progress` 触发换位，改为index3 `(2.567,0.769,-1.047)`；217步进入M1等待。 |
| 226-269 | 226步发布一次drawer_scan命令；backend在228步开始，269步完成三抽屉扫描并关闭。 |

初始预检切换有安全依据：36步引用的local map中，index0中心代价62、最近lethal格中心距0.319m，小于当前完整净空半径约0.335m；index1中心代价99；index2中心代价0、最近lethal距离0.490m。
但候选发布与执行预检分离，导致视频先画index0、随后才改成index2。这里不是M1返回不稳定。
index2到index3的移动仅约0.176m，新点与当时机器人距离约0.148m；再调整朝向后即可进入M1，不是做了第二轮抽屉物理交互。

确定的控制故障是：安全可规划的index2在位置容差外形成零平移摆动，直到179步后换位。该点局部净空约0.49m，不能解释为目标中心被障碍占据。
现有日志没有DWA各critic的逐轨迹评分，尚不能唯一归因到某个评分项；需要补充末端控制模式和轨迹评分诊断。位置锁定修复也不能单独解决一个从未进入0.15m区域的目标。

### 7.3 H4：高层目标没换，位置锁定配置没有进入实际读取路径

- 129步选中 `frontier:11:18`，到结束都没有改目标，坐标始终 `(5.150,7.050)`、yaw为2.8966rad。
- 215-390步位置误差在0.2348-0.2561m之间，对0.25m位置门槛跨越9次，176个记录中102个在门槛内。与此同时目标朝向误差仍很大，例如300步2.168rad、380步1.662rad。
- 目标附近不是无路或障碍中心：300步local map目标中心代价0，最近lethal格中心距约0.772m；DWA仍在产生局部轨迹。
- 391步因mission连续180步无有效进展提前结束。不能用收尾时的step-sync超时解释前面的长期摆动。

静态确认的配置问题：当前安装的DWA/base_local_planner为Conda标准Noetic库，包位于 `/home/ldl/conda_envs/ros-noetic/share/dwa_local_planner`，库内构建路径指向标准ROS源码；本worktree的devel/lib没有替代DWA库。
本机头文件 `base_local_planner/latched_stop_rotate_controller.h:22` 的默认构造参数为空；标准DWA构造没有给该成员传入DWA插件名。
因此构造器读取的是 `/move_base/latch_xy_goal_tolerance`，而本轮只配置了 `/move_base/DWAPlannerROS/latch_xy_goal_tolerance=true`，前者默认false。
读取逻辑见 [ROS Noetic LatchedStopRotateController源码](https://raw.githubusercontent.com/ros-planning/navigation/noetic-devel/base_local_planner/src/latched_stop_rotate_controller.cpp)，成员构造及控制分支见 [DWAPlannerROS源码](https://raw.githubusercontent.com/ros-planning/navigation/noetic-devel/dwa_local_planner/src/dwa_planner_ros.cpp)。

这解释了容差边缘为什么不能可靠保持“位置完成、只收敛yaw”的状态，反复回到普通DWA采样。参数路径错误已确认；修复后对摆动幅度与成功率的改善仍须对照回放验证，不能声称已经验证通过。

相关配置：`Interactive-Nav-SG-nav/src/nav_pkg/configs/controller/dwa_controller_params.yaml:23`、`scripts/InteractiveNav/configs/semantic_decision/semantic_interaction_nav.yaml:12`。

### 7.4 H5：接近位全部不可达后冷却，没有可执行候选

- 406步选择冰箱；407步预检选中index0，目标 `(2.242,0.649,pi)`。
- 初期目标距local map边界只有约0.249m，小于完整占用检查需要的0.335m，因此 `_container_anchor_local_clearance` 返回暂缓局部净空检查，允许先走全局路径。
- 随机器人转身、移动，新地图中原目标附近出现占据：最近lethal距离由约1.173m降至0.324/0.290/0.285m。425步以 `container_anchor_footprint_blocked` 放弃index0，试index2；426步再次失败，剩余目标中心为99。
- 426步报告7/7接近位不可达、0次targeted M1采样；位置距有效目标仍约2.55m，未到达物理交互位置。
- 失败却被归到 `container_m1_evidence_inconclusive_viewpoint_navigation` / `interaction_visual_precondition`，对象进入300step冷却。
- 450步trace明确记录 `container_anchor_step_cooldown`、`no_curated_model_candidates`；两个有效frontier簇都没有安全观测位，导航候选为0。机器人此后停在约 `(1.652,3.044,-1.435)`。

因此450步后不是M1请求挂起，也不是物理交互等待，而是IDLE等冷却。剩余不到100步，不可能走完300step冷却。
上游仍将这个被冷却的冰箱计入 `interaction_frontier_count=1`、`frontier_exhausted=false`，但curation没有可执行候选；没有产生新的恢复导航或明确的受阻终态。
新占据是否包含深度伪障碍，目前无原始逐点深度归因证据，不能直接认定为真实家具，也不能直接删掉它。

代码位置：`semantic_behavior_executor.py:6629`（净空/窗口延后）、`:10760`（动态检查失败立即换位）；`semantic_rule_decision_node.py:1001`（对象冷却）、`:2487`（冷却候选过滤）。这些文件均位于 `Interactive-Nav-SG-nav/src/semantic_decision_py_pkg/scripts/`。

### 7.5 H9：边界抖动、近目标绕转和错误的整面排除叠加

1. 157步选择冰箱index0，直到328步才换位。269-328步距离在0.1469-0.1509m之间，对0.15m门槛跨越26次；位置锁定命名空间问题同H4。
2. 328步以 `navigation_stagnation` 重试index2。新目标距机器人只有约0.068m、yaw误差约0.234rad，却仍先调用rear-goal preturn，按“通往新XY的方向”转身。329-341步先转到约-1.757rad，随后DWA又转回目标yaw1.658rad；405步才真正满足姿态条件。
3. yaw进展监控在距离大于 `final_align_max_distance_m=0.12` 时优先看最初路径lookahead，而交互位置条件为0.15m；在0.12-0.15m区间，朝向交互目标收敛不一定被识别为朝正确方向进展。位置边界抖动还使终端yaw保护反复退出。
4. 408步targeted M1正常返回：`view_state=front`、`approach_ready=true`、`confidence=0.95`。411步确实发出了交互命令；backend在412步施力前拒绝，不是完全没调用。

412步backend几何检查：

| 检查项 | 实际值 | 门槛 | 结果 |
| --- | --- | --- | --- |
| 相对选定目标位置 | 0.0660m | 0.15m | 通过 |
| 相对选定目标yaw | 0.1554rad / 8.90度 | 0.2rad / 11.46度 | 通过 |
| 相对AABB正面yaw | 0.2427rad / 13.90度 | 15度 | 通过 |
| 相对物理门板正面的位置角 | 0.2143rad / 12.28度 | 15度 | 通过 |
| 相对物理门板正面的yaw | 0.3705rad / 21.23度 | 15度 | 失败 |

关键错误不是已经证明“机器人站在背面”，而是朝向超差被归为 `interaction_wrong_face`。执行器随后把同一AABB面的index0-7全部加入 `container_m1_rejected_face_staging_indices`，立即得到 `interaction_wrong_face_options_exhausted`，失去原位朝向修正机会。
目标采用AABB轴与fan角度，backend用门板初始开启动作推导的正面，两者参考法向不同，叠加到位yaw允许误差后触发拒绝。应统一几何契约并区分“位置在错面”和“同面朝向未收敛”，而不是直接放松物理安全校验或给规划器泄露GT轴。

代码位置：`semantic_behavior_executor.py:10374`（新导航统一pre-turn）、`:6757`（XY方向判后方）、`:10787`（yaw进展参考）、`:3091`（整面排除）；`scripts/InteractiveNav/force_interaction_bridge.py:1984`、`:2030`（物理正面检查及合并错误类别）。

### 7.6 H10：第一抽屉接近位不是没尝试，而是新地图触发取消

- 381步选择 `chestofdrawers_995745317704128ee80131356cba94cc_1_0_5`，index0为 `(0.961,4.345,pi)`，预检32个位姿后仍选index0。
- 451步取消时已经向该点走了约70步；位置误差约0.178m，yaw误差约1.219rad / 69.8度，尚不满足0.15m与0.2rad合同，没有进入M1或drawer_scan。
- 450步引用的local map仍给该目标中心cost0、最近lethal距离约0.518m；452步引用的451步local map中心变成99、最近lethal距离约0.226m。executor同时明确记录 `container_anchor_center_blocked`。
- 当前代码单次动态检查失败立即取消并排除该位姿，然后预检剩余23个候选，选择index25 `(0.536,3.262,1.833)`，与机器人相距约1.09m；453-465步还先执行一次rear-goal preturn。
- 后续导航持续到500步上限，抽屉没有物理交互。499步编号下的 `navigation_step_sync_stall` 和大量新目标重试发生在仿真停止后，不是视频里已完成的抽屉操作。

这次切换有明确的局部占据触发，不是M1评价后跳过。缺陷在于：候选早期预检与近场实际占据不同；新占据一次失败即长期排除，没有记录障碍来源或短期复核；恢复仍按预排候选优先级跳到较远姿态。新占据是否真实，需要补充深度/占据来源证据，不能为了避免换点而忽略99。

### 7.7 修复优先级与附带发现

1. 先对齐实际Noetic参数命名空间，给终端XY/yaw收敛增加一致的阶段锁定、滞回和有限步数；进展监控不能在到位区域继续拿旧lookahead当最终朝向。H2还需要针对容差外零平移的局部极小值补诊断/有限恢复。
2. 已在新目标XY容差内时跳过面向XY的rear preturn，直接用同一个有界控制器对准目标yaw；不要先背离物体再转回来。
3. 将 `interaction_wrong_face` 拆分为选错面、同面位置偏差、同面朝向偏差；仅在有面错误证据时排除整面，保留原位朝向修正/有效重观察。
4. 导航不可达与M1无证据分别反馈；使用通过冷却过滤后的可执行候选判断探索是否能继续。无候选时进入有界恢复或明确受阻状态，不能仅依赖一个300step对象冷却空等。
5. 候选发布前尽量采用同一套已知净空标准；新局部占据保留安全约束，但记录来源、做有界复核，区分瞬时地图变化与持续不可达。

附带发现：H10冰箱物理执行78-82步成功，但executor在80步就发布 `interaction_timeout`，反馈显示仅经过3个task step。
`behavior_execution.py:3723` 的INTERACTING分支仍按30秒墙钟判断；`:3850` 却在有step记录时统一标为 `timeout_clock=task_steps`，且该reason对应的 `timeout_task_steps=null`。
因此本轮“物理成功9次”不代表9次executor行为均正常成功，超时归属和统计也需修复。这不是381步之后抽屉第一接近位被切换的原因。

本次未做新的仿真验证。上述证据区分了确定的事件/配置错误与需要补充DWA评分、深度来源或对照回放才能确定的更底层原因。

## 8. 2026-09-14 位置锁定与恢复修复

按用户要求实施位置完成锁定：同一决策、目标位姿、阶段及容差的接近位首次进入XY容差后，
后续只以yaw判断导航到位；停车重发保留锁定，换目标或阶段重新建立。到位复核与M1位姿绑定保留
首次到位证据，不用当前XY重新撤销完成状态。碰撞与backend真实交互面安全检查没有放宽。

- 修正Noetic位置锁定参数命名空间，公开执行状态新增 `navigation_arrival` 诊断。
- XY已到位时不执行面向新XY的rear preturn；近目标容差外空转不再延长平移进展判断。
- 动态占据先停车，最多复核3个任务步；短暂恢复允许有上限的同点重发，重选优先附近安全候选。
- backend区分同面yaw、同面位置、真实错面及未确认正面；仅真实错面允许整面排除。
- 同面yaw失败保留一次原位恢复，使用公开TF和M1已确认正面信息，不读取或返回GT纠正轴。
- 物理交互按120step计时，180秒step停流兜底；无step才使用墙钟。导航不可达不再冒充M1失败，
  导航冷却20step，持续60step无eligible候选则明确报告受阻。

### 8.1 回归暴露的短路径崩溃与防护

首次修复回归目录 `batch_latched_h1_h10_500_20260914_011500` 中，H9在388步正确返回
`interaction_orientation_misaligned`，且 `reject_selected_face=false`；但随后原位恢复仍调用inner-corridor
路径预检，同XY规划期间 `move_base` 出现 `munmap_chunk(): invalid pointer`，退出码-6。
该批次已停止，日志保留，不能作为最终500step结果。

已确定的上游风险：Noetic `FORWARDTHENINTERPOLATE` 模式对非空路径直接使用 `n-3`，
两点路径会产生负索引读写；本机库反汇编也存在该路径。见
[ROS navigation 1.17.3源码](https://raw.githubusercontent.com/ros-planning/navigation/1.17.3/global_planner/src/orientation_filter.cpp)。
这与本次同XY调用高度吻合，但没有core/backtrace，不能声称已经唯一定位崩溃栈。

两层防护已实施：原位恢复直接使用带碰撞检查的独占step旋转，不再调用路径服务；
实际 `nav.launch` 强制包装器的委托朝向模式为0，由本地 `OrientedGlobalPlanner` 安全处理切线与末端朝向。
第二、第三次短暂启动也因补齐该风险及实际参数覆盖问题主动停止，未作为完整评测结果。

### 8.2 验证与最终批次

- 决策/执行/交互桥554项通过，感知/房间分割/视频补充186项通过，总计740项。
- `git diff --check` 通过；launch展开参数已验证模式0和位置锁定true。
- 完整重跑目录：`/home/ldl/outputs/interactive-nav/batch_latched_v4_h1_h10_500_20260914_014500`。
- 配置：H1-H10、10worker、每场500step上限、house=seed、full_mllm_exploration、organized_depth、
  pointcloud stride1、初始关门/容器关闭、开启录制，ROS端口15801-15810。
- 最终批次已完成，10个runner退出码均为0；9个运行源文件/配置的SHA256与启动时一致。
  此前中止批次不与完整评测混算。

### 8.3 最终500step并行结果

下表为「成功数 / 已发送命令数」，不是仅统计已返回结果的命令。每个抽屉扫描序列计一次容器交互，
不把序列内部多次开关抽屉单独计数；导航、M1调用、reobserve不算物理交互。

| 场景 | 实际步数 | 门 | 容器 | 未返回结果 | coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1 | 500 | 0/0 | 0/0 | 0 | 0.4414 |
| H2 | 367 | 0/0 | 1/1 | 0 | 1.0000 |
| H3 | 472 | 0/0 | 0/0 | 0 | 0.2843 |
| H4 | 500 | 0/1 | 1/1 | 0 | 0.5084 |
| H5 | 500 | 0/1 | 0/0 | 0 | 0.9976 |
| H6 | 500 | 1/1 | 0/1 | 0 | 0.9940 |
| H7 | 500 | 0/2 | 1/1 | 0 | 0.5558 |
| H8 | 500 | 0/0 | 0/1 | 1 | 0.2025 |
| H9 | 500 | 0/0 | 0/1 | 0 | 0.6621 |
| H10 | 500 | 1/2 | 1/2 | 1 | 0.5804 |
| 合计 | 4839 | 2/7 | 4/8 | 2 | 均值0.6226 |

- 15条交互命令，13条已返回：6成功、7失败；另2条在500步上限时未返回，不算成功或失败。
- 7条失败包括5条门的non_articulated、H6抽屉的interaction_wrong_face、
  H9冰箱的interaction_position_misaligned。non_articulated包括观察到的静态开口，
  不等同于对一扇实际可动门施力后失败。
- H2在367步以no_executable_candidates_after_terminal_interaction_no_plan提前结束；
  H3在472步以semantic_mission_no_progress提前结束。其余8场均到500步。
- 全部10段1440x540、15fps视频首尾可解码且非空；4839帧均exact-step对齐，
  missing_sim_step_indexes和missing_raw_step_indexes均为空。组件图层遵守因果时间选择，
  不将旧组件快照描述为所有组件同一步刷新。
- 机器可读审计：`/home/ldl/outputs/interactive-nav/batch_latched_v4_h1_h10_500_20260914_014500/audit_latched_500.json`。

### 8.4 重点场景修复效果与残留

- H2：原抽屉接近位index2首次在录制步132进入锁定区间，132-141步朝向误差从0.579降到0.239rad，
  后续进入M1和drawer_scan；物理记录从158步开始，199步executor成功反馈。
  不再复现上轮直到214步才换index3的过程。之后门接近停在约0.307m，仍没有到0.15m门槛；
  345步接近候选耗尽，最终因无候选提前结束。位置锁定不能解决从未进入容差的问题。
- H4：266-301步frontier目标保持位置锁定，朝向误差1.707降到0.201rad，301步导航成功。
  末段450-499步实际累计移动约1.442m，未复现上轮同一个frontier长期跨容差摆动后391步早停。
  本轮末目标尚未到位，不能据此保证所有目标都能收敛。
- H5：450-499步累计移动约3.193m、净位移3.101m，不再是上轮450步后IDLE等待300step冷却。
  但本轮未发出冰箱交互命令，不能声称冰箱交互已成功。
- H9：262-292步冰箱目标锁定后朝向误差1.428降到0.213rad，298步backend返回同面位置不符合。
  选定目标位置误差0.120m、yaw误差0.158rad均满足导航合同，AABB正面校验也通过，
  但backend物理正面位置/yaw均未通过。这与“导航重新检查XY后未调用交互”不同。
  新分类保留同面候选，不再整面排除；恢复到另一目标后412-455步再次完成锁定旋转，
  后续更换接近目标，472步以container_approach_navigation_unreachable / rear_goal_turn_failed结束。
  本轮没有orientation-only拒绝，因此纯yaw恢复的完整ROS成功闭环尚未在最终批次中验证。
- H10：抽屉第一接近位仍为(0.961,4.345,pi)，447-471步录制状态已位置锁定，
  朝向误差1.291降到0.217rad；476步发出drawer_scan，500步时仍在执行、没有最终结果。
  450-499步累计位移仅0.019m，没有因原目标单帧占据跳去上轮约1.09m外的候选。
- H8：481步发出drawer_scan，500步时没有最终结果，与H10一样单列未完成。

最终日志共29段导航位置锁定区间、771个锁定导航帧。此次这些帧没有越出原XY容差；
因此「进入后漂移出容差仍只检查yaw」主要由针对性单元测试验证，不能把本轮录像当作该边界条件
已经充分覆盖的证据。

### 8.5 不满足整体无退化结论

上轮完整500step批次物理成功9次（门5、容器4），本轮6次（门2、容器4）。
平均coverage由0.6583降至0.6226；H9由0.9799降至0.6621，是主要下降来源。
H1、H8门成功也未在本轮重现。MLLM和异步运行会影响轨迹，且提前停止步数不同，
这只是本轮观测差异，尚未做重复对照归因；不能宣称修复后整体性能不下降。

运行中未发现move_base退出码-6/-11或Python Traceback；但所有10场日志均在
「killing on exit」之后出现分配器异常，H5/H6/H7另有publish() to a closed topic的关机竞态。
这些现象在前一批也存在，仍未解决。没有原生调用栈，不能把所有分配器异常唯一归因到
GlobalPlanner短路径缺陷，也不能用runner退出0证明原生内存安全。

后续优先级为：H9同面接近与物理正面契约、H1/H2未到位的导航恢复、H8/H10临近horizon的
未完成交互统计与调度，以及带调用栈的原生关机异常定位。此次不追加未获要求的长时间仿真。

### 8.6 本轮全部视频

H1

![H1 最新500步](/home/ldl/outputs/interactive-nav/batch_latched_v4_h1_h10_500_20260914_014500/house_0001/videos/overview_6panel.mp4)

H2（367步提前结束）

![H2 最新367步](/home/ldl/outputs/interactive-nav/batch_latched_v4_h1_h10_500_20260914_014500/house_0002/videos/overview_6panel.mp4)

H3（472步提前结束）

![H3 最新472步](/home/ldl/outputs/interactive-nav/batch_latched_v4_h1_h10_500_20260914_014500/house_0003/videos/overview_6panel.mp4)

H4

![H4 最新500步](/home/ldl/outputs/interactive-nav/batch_latched_v4_h1_h10_500_20260914_014500/house_0004/videos/overview_6panel.mp4)

H5

![H5 最新500步](/home/ldl/outputs/interactive-nav/batch_latched_v4_h1_h10_500_20260914_014500/house_0005/videos/overview_6panel.mp4)

H6

![H6 最新500步](/home/ldl/outputs/interactive-nav/batch_latched_v4_h1_h10_500_20260914_014500/house_0006/videos/overview_6panel.mp4)

H7

![H7 最新500步](/home/ldl/outputs/interactive-nav/batch_latched_v4_h1_h10_500_20260914_014500/house_0007/videos/overview_6panel.mp4)

H8

![H8 最新500步](/home/ldl/outputs/interactive-nav/batch_latched_v4_h1_h10_500_20260914_014500/house_0008/videos/overview_6panel.mp4)

H9

![H9 最新500步](/home/ldl/outputs/interactive-nav/batch_latched_v4_h1_h10_500_20260914_014500/house_0009/videos/overview_6panel.mp4)

H10

![H10 最新500步](/home/ldl/outputs/interactive-nav/batch_latched_v4_h1_h10_500_20260914_014500/house_0010/videos/overview_6panel.mp4)

## 9. H2门接近点净空与摆动专项复核

仍对应第8节最终V4批次。本次仅离线分析，未修改运行代码、未追加仿真。

### 9.1 不是多次更换subgoal

- 201步选中reobserve_portal:door_0001，目标G0=(2.282046,4.285137,0)，202-345步实际目标始终不变。
- 260步距目标0.363m；270-345步一直停在0.3047-0.3084m，未进入0.15m容差。
  这76个记录的朝向在64.12-99.47度间摆动，目标朝向为0度；累计XY微移0.0482m，净位移仅0.0024m。
- 因为从未到位，position_latched始终false，没有进入到位M1或发出门交互命令。
- 345步semantic_subgoal_no_progress结束接近，367步以无可执行候选终止场景。

### 9.2 目标贴软膨胀边缘，但并非硬碰撞点

使用300步记录绑定的local_costmap_full:1356、0.05m栅格，还原png_value_offset后计算；
距离是到最近lethal栅格中心的距离，不是精确mesh表面距离。该批实际local inflation=0.45m、
robot_radius=0.25m；参考容器完整净空检查半径为0.25+0.05+0.05/sqrt(2)=0.3354m。
这个参考检查本轮并未用于门，不能把图上的红圈误读成门已经通过该执行检查。

| 候选 | XY | 中心cost | 最近lethal格中心距离 |
| --- | --- | ---: | ---: |
| G0（唯一执行） | (2.282,4.285) | 14 | 0.4399m |
| G1（未执行） | (2.032,4.285) | 14 | 0.4399m |
| G2（未执行） | (1.782,4.285) | 0 | 0.4420m |

230-345步G0附近地图没有出现上轮H10那种突然变成99的情况；250步以后最近lethal距离稳定约0.44m。
所以这里不能说目标被实体障碍直接占据、或单帧新障碍强制换位。G0处于软代价区，点本身仍有
大于参考0.3354m的净空，但不保证整个0.15m到位区域都安全；保守覆盖整个区域需约0.4854m净空。

### 9.3 三层原因

1. 候选生成缺少横向避障自由度和净空优选。behavior_candidates.py:3162的portal分支只沿同一
   法向增加0/0.25/0.50m站距，tangent固定0。三点y都为4.285，不能明显远离上方横向墙面占据带。
   执行器semantic_behavior_executor.py:6702的净空检查对非container直接返回True；
   :10251首次make_plan可达便选中，实际只预检G0一次，没有比较G1/G2的终点代价。
2. 直接观测到的是DWA末端XY不收敛。机器人本身约在(2.32,3.98)，距最近lethal约0.744m，
   同时local_plan仍新鲜。不是已经挤在硬障碍上不能转动，而是距离剩约0.306m时持续零净平移摆动。
   障碍软代价与路径/目标评分冲突是待验证因素；现有记录缺少DWA逐轨迹critic得分，
   无法唯一归因到某个评分项。不能仅凭视频把该控制问题全归为目标距墙太近。
3. 门重试存在确定缺口。semantic_behavior_executor.py:11618调用通用接近重试，但
   behavior_execution.py:1285的允许原因集合遗漏semantic_subgoal_no_progress；同参数离线调用：
   INTERACT、index0、已尝试1次、上限4、候选3时，该原因返回None，navigation_stagnation则返回1。
   随后semantic_behavior_executor.py:13267把单点无进展归一化为interaction_approach_options_exhausted，
   导致还剩两个候选却报告耗尽。反馈中的preflight_batch_goal_count=1和单条attempt与此一致。

### 9.4 建议修改边界

- 给portal复用与container一致的终点/转向净空检查，远场依据全局图，接近后依据新鲜local图复核；
  将净空作为候选排序依据，并检查可接受到位区域，不只看目标中心是否非lethal。
- 在同一门正面角约束内允许小范围切向候选，不是任意侧移或走到另一面。本图离线示例：
  G0向-y偏0.10/0.15m，中心cost均为0，最近lethal分别约0.540/0.590m；
  对门中心的偏角约5.95/8.88度，小于本候选11.46度位置角约束。偏0.20m则约11.77度，已超约束。
  这些只是离线候选净空与AABB角度检验，不代表已通过路径、M1或backend物理验证；
  yaw容差也必须避免与切向偏角叠加超出正面要求。
- 把单subgoal无进展纳入有界换位；只有所有有效候选尝试失败后才能报exhausted，mission级停滞仍终止。
  对容差外的接近进度用距离收敛判断，避免原地yaw摆动延长等待；不建议用继续加timeout或缩小inflation掩盖问题。

测量脚本与JSON：outputs/analysis_latch500_20260914/h2_door_analysis.py、h2_door_metrics.json。

![H2门目标净空、距离与朝向证据](/home/ldl/molmospaces-exp-setting/outputs/analysis_latch500_20260914/h2_door_clearance.png)

## 10. 门净空与切向候选实现及测试

按用户要求实现第9.4节前两项，未修改第3项无进展重试白名单，也未启动新的500step仿真。

- 新生成的可见门和记忆门接近候选均启用完整到位区域净空检查。远场使用新鲜全局planning OCC，
  近场以新鲜local costmap为准；局部图过期不能用全局图绕过。未知和lethal格按完整圆形占用范围拒绝，
  99只检查目标中心，避免重复膨胀。已锁定位置时只检查实际机器人占用范围，不重新核验目标XY容差。
- 同一正面增加有限的+/-0.10m、+/-0.15m切向选点。门法向保持不变；对每个候选，
  预留整个到位圆和yaw误差区间所占用的正面角，收紧对应导航/交互容差；余量不足0.05m或0.05rad则剔除。
- 对所有通过净空检查的门候选做路径预检，按净空优先选择可达点，不再遇到第一个可达点就停止。
  远场已下发选中候选的准确到位容差，进入local窗口不因切换地图来源单独取消重发。
- 参数：candidate.portal_tangent_offsets_m=[0.10,-0.10,0.15,-0.15]，可配置为空列表关闭切向候选。

验证：决策执行包和交互桥574项测试通过；覆盖同面约束、容差绑定、完整到位圆、99不重复膨胀、
未知格、局部图过期、全局图兜底、位置锁定后实际占用检查，以及批量净空排序。

另使用真实H2第300步的local_costmap_full:1356离线运行新生成器与执行器净空函数：

| 项目 | 原G0 | 净空优选候选 |
| --- | ---: | ---: |
| XY | (2.282046,4.285137) | (1.782046,4.135137) |
| 目标yaw | 0 | 0.102382rad |
| 中心cost | 14 | 0 |
| 最近障碍格中心距离 | 0.439920m | 0.591425m |
| 完整到位区检查半径 | 0.485355m | 0.471092m |
| XY到位容差 | 0.15m | 0.135737m |
| yaw到位容差 | 0.195rad | 0.092618rad |
| 净空结果 | 拒绝 | 通过 |

13个几何候选中5个通过该帧净空检查。这里只验证历史图上的几何和净空筛选，
没有调用新ROS make_plan或执行物理开门，不能宣称H2交互已成功。
原V4视频仍是修改前记录，不标为新版本结果。

原始离线结果：
[h2_portal_fix12_replay.json](/home/ldl/molmospaces-exp-setting/outputs/analysis_latch500_20260914/h2_portal_fix12_replay.json)。
