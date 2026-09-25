# Go2 实物扩展与公共算法的边界

2026-09-25。调整前存档：`5619e193`；公共算法合并基点：
`origin/codex/exp-setting` 的 `2d159711`，合并提交 `cfaef906`。

## 目录与依赖方向

| 层 | 权威源码 | 职责 |
| --- | --- | --- |
| 公共算法 | `Interactive-Nav-SG-nav/src/semantic_*_py_pkg`、`explore_py_pkg` | 建图、图状态、M1/M2、候选和执行状态机 |
| 实物扩展 | `scripts/InteractiveNav/physical_nav` | Go2/D435i/IMU、采集时间同步、YOLO 接入、速度安全、实物交互后端、可选网页 |
| ROS 接入外壳 | `Interactive-Nav-SG-nav/src/physical_nav` | ROS 包声明、薄启动器、catkin 资源安装；不再维护第二套配置/算法 |

依赖只允许「实物扩展 → 公共接口」，公共算法不导入 Go2 节点、网页或实物脚本。
同仓库维护，不复制 exp-setting 算法，不新建大型节点子类或 monkey-patch。

`config/`、`launch/` 的权威文件仍在实物脚本目录，旧 ROS 路径是相对符号链接。
以前 ROS 副本曾保留 13° 外参、旧检测阈值和旧探索配置；现在两个入口均读取同一份。
安装时从权威目录生成普通文件，不把指向源码树的链接带进安装目录。
原有代码/配置仍可从存档提交恢复；未清理用户的 `.bak`、性能记录或论文文件。

ROS 包的 `.runtime` 是私有实现资源：源码树中为链接，install 空间中为真实文件。
名称以点开头是为了避免 ROS1 递归资源查找把实现和兼容入口认成两个同名节点/launch。
`physical_nav_runtime.py` 通过 ROS 包定位运行资源，不再依赖 `parents[4]` 猜仓库位置。
source、catkin devel 和搬迁后的 install 空间使用同一启动器；legacy sibling imports
仅在启动入口内兼容，恢复原 `sys.path`，不注入公共算法模块。

## 已抽取的接口

| 接口 | 输入/输出 | 状态所有者 |
| --- | --- | --- |
| `ProgressClock.context` | 原始采集 step → 进度 tick + 原采集身份 | 候选节点；仿真默认采集步，实物 YAML 选择 monotonic |
| `portal_approach_pose` | OBB、机器人位置、距离、上次门侧 → 位姿与新门侧 | 候选生成器保存门侧记忆；函数无隐藏状态 |
| `semantic_evidence.persistence_action` | 已接收 M1 证据、观测次数 → latch/clear/keep | GraphStore 唯一负责写图，保留两帧 + M1 门槛 |
| `capture_geometry` | 采集位姿/IMU/标定 → 相机变换、因果位姿匹配 | 实物传感器桥与 YOLO 共用纯 NumPy/Python 函数 |
| 实物交互后端 | ROS interaction_command/cancel/result | 现有 physical_interaction_policy；不通过网页完成交互 |
| `PostOpenMapPolicy.evaluate` | 原始/规划 OCC、全量/增量 costmap 的不可变计数快照 → 放行状态 | 原执行器保存接收计数、首次通过的屏障与条件锁 |
| `ExitObservationSweep.advance` | 单调时间、朝向、归属/取消标记 → 角速度意图或终态 | 原执行器持有局部 sweep，校验 decision + run token 后发布 |
| `container_approach` | 两阶段候选、已采样/不可达索引、M1 正面证据 → 视点顺序与物理接近位姿 | 原状态机保存当前阶段、采样历史与重试预算 |
| `container_evidence` | 公开 M1 更新、请求采集步、采集位姿 → 合同校验原因与正面方向 | 原执行器检查 request/decision 归属、读取 TF、保存证据 |
| `build_interaction_command` | 候选、调用方分配的序号/基础 ID、episode 回退值 → 独立的公开命令字典 | 原执行器分配序号、绑定返回结果、发布 ROS；函数不发送命令 |

这次抽取不修改几何公式、OBB 方向、外参、相机模式或速度安全阈值。
所有权、episode/command fence、M3 状态封口、地图新鲜度屏障仍由公共执行器维护，
不能复制到实物层另存一份状态。latest-only、时间戳配对和异步房间分割是公共能力，
不能为了缩小 diff 而撤掉。

统一配置使旧 ROS 探索副本遗漏的三项设置生效：OCC 消费周期 0.5 s、关闭
ExplorePy 的独立 make_plan 预检、关闭独立 external recovery 预检。
数值未新增/调优，来源是已有权威 `explore_physical_override.yaml`，
目的是让语义执行器继续独占规划/恢复职责。仿真默认配置不变。

## 网页不再是健康判断依赖

主链：Go2 → sensor ROS bridge → ROS RGB-D/TF/points → YOLO/OCC → 公共算法。
网页只接收镜像与发布受安全门控制的命令，不拥有原始帧、图状态或运动使能状态。

监督进程的 watchdog 现在读取：

- `/physical_nav/capture_context`：已映射到本地 ROS 时钟的采集时间和 seq。
- `/physical_nav/yolo/heartbeat`：`std_msgs/Header`，仅在成功发布真实 YOLO report
  后发送；空检测也是有效推理结果，模型预热不是。

它不拉取图像、mask、大点云或网页 `/api/health`。同时间戳重发/乱序不能刷新
健康计时；排队旧帧的采集年龄也计入超时；本地收包间隔用 monotonic 计算。
传感器重启后允许 seq 归零，前提是传感器桥输出的归一化采集时间继续前进。
这是数据进度判断，不是直接查询网络 socket 的连接状态。
`--source http --url ...` 仅保留显式旧诊断兼容；默认 ROS。
关闭网页仍启动 ROS watchdog，且不再默认向不存在的网页发送 MLLM trace。
网页 HTTP 就绪也不再是 ROS/YOLO 启动的前置条件，网页启动失败不会因此退出算法栈。
watchdog 保持 `--keep-running` 的原有恢复策略，不因一次数据暂停直接杀整个栈。

## 启动、安装与验证边界

旧命令不变：`bash scripts/InteractiveNav/physical_nav/physical_nav_all.sh start`。
本机 supervisor 内部统一使用 `roslaunch physical_nav physical_nav_readonly.launch`。
无网页时设置 `PHYSICAL_NAV_START_WEB=0`。这些命令会启动服务；本次离线验收没有执行它们。

catkin 安装覆盖 ROS 适配器、共享实现及配置/launch，验证了 install 目录中的传感器
启动器无需源码路径即可导入和显示帮助。完整 Go2 SSH 编排、模型权重、conda 环境、
uni_control 与网页渲染器仍是部署依赖，**不是已打包好的单独 pip 产品**。
安装不会下载模型或依赖，也不会启动相机/机器人。测试命令统一见根目录 `test.md`。

第一阶段离线验收：调整前 1942 项通过，调整后含新增边界测试 1960 项通过；
`slam_gmapping`、`oriented_global_planner`、`path_follower` 构建通过，
无网页 launch 参数解析及 devel/临时 install 启动器导入通过。

## 第二阶段：执行器中的开门后流程

新增公共模块 `semantic_decision_py_pkg/post_open_maps.py` 与 `exit_observation.py`，
均不导入 ROS、实物扩展、网页、线程或网络库，也不自行读取时钟。

地图模块包含原有三类屏障数据结构、新鲜度函数、部署接线策略和诊断字段生成。
`behavior_execution.py` 保留原符号的兼容导出，类/函数身份不变。
`PostOpenMapPolicy(direct_raw_costmap=False)` 默认要求 raw OCC → planning OCC →
global costmap；实物配置已有的 `post_interaction_costmap_fast_path_enabled=true`
选择 raw OCC → global costmap。后者仍要求真正后到的 costmap 回调，不能凭
本地 now 时间戳的变大放行。只复制少量标量计数，不复制 OCC 数组。

执行器保留唯一的 map condition、基线缓存、首次放行缓存和等待循环；模块不另起
worker，也不维护第二份图状态。M3 终态、事件 ID、规划请求和地图发布频率均未改变。
原有 `post_open_*` / `costmap_fresh` 等诊断字段、各阶段超时原因保持兼容。

门外观察改为纯 tick 状态机：旋转 → 停稳观察 → 下一视角 → 完成/超时/取消。
左右角度、旋转速度、回正开关、停稳时长与各视角超时仍读取原配置；仍只在固定
门外导航点成功到达后触发一次，不增加探索 subgoal，不改变 M2 打分。
每个 tick 的意图由原 ROS 执行器应用，没有新增线程、队列或外部 RPC。

同时补齐三个安全边界：

- 地图新鲜与任务取消同时发生时，归属检查优先，旧等待不能继续放行。
- 观察在停稳阶段也每至多 0.05 s 检查取消；每次发布（包括 finally 停止）都在
  同一把执行器锁下检查 decision 和 navigation run token，旧 worker 不覆盖新命令。
- 观察途中 TF 缺失/朝向非有限时返回零角速度，避免沿用上一条非零旋转；ROS shutdown
  进入 canceled，不把它记成正常完成。超时或异常仍由当前所有者发送停止。

新增纯策略和执行器边界测试覆盖两种地图接线、旧时间戳/假新 seq、分阶段超时、
地图与取消并发、角度跨 ±π、左右/回正、停稳取消、同 decision 新 worker、异常停止。
以上是离线功能验证；尚未重启真机，不代表现场延迟/运动表现已经验收。
第二阶段全量回归：1997 项通过（比上一阶段增加 37 项），保留 4 项 ROS 依赖
弃用警告；ROS 执行器本体净减少 315 行，原通用执行模块的地图合同兼容导出已验证。

## 第三阶段：容器两阶段策略、证据与命令合同

新增三个公共纯 Python 模块，不依赖 ROS 节点、实物扩展或网页：

- `container_approach.py`：外层 M1 视点排序、已采样位姿去重、同一面的索引分组、
  外层采集点到内层动作点的映射，以及 M1 确认正面后的接近几何。
- `container_evidence.py`：公开 bbox/采集步检查、侧向裁切判定、可见且新鲜的 M1
  正面证据门槛，以及由采集位姿/选定面冻结正面方向。直接正视开关显式传入。
- `interaction_commands.py`：抽屉视觉合同检查及后端命令组装。缺少视觉证据的抽屉
  命令在节点和纯接口两处均拒绝；门户只透传公开 aperture 字段。

`behavior_execution.py` 继续兼容导出原两阶段函数；角度归一化移至共用几何模块，
旧导入仍指向同一函数。原执行器私有校验入口保留薄委托，调用位置与状态推进不变。
视点顺序、同面重试、几何公式、接近距离、M1 门槛和超时参数没有调整。

任务/请求归属校验、TF 读取、采集证据缓存、停稳保持、异步 M1 接收、状态转换和
ROS 发布仍由原执行器/状态机负责。纯模块不新增线程、队列、时钟或网络调用，也不
复制图像/点云/OCC；命令构造仅复制小型公开字段，避免返回值与候选嵌套字典共享。
每次物理尝试的序号仍由执行器加锁分配，发送前绑定 command/event/episode，
没有让构造器另存一套命令生命周期或重新编号。

第三阶段验收：相关回归 388 项、全量离线回归 **2041 项通过**（新增 44 项），
保留 4 项既有 ROS 依赖弃用警告。额外对照存档 `5619e193`：15 个容器几何函数
AST 未改变，4 个纯证据方法执行体未改变（忽略文档缩进）；门、冰箱、柜子、抽屉
结合开/关、opaque 模式、episode 来源的 40 组命令 JSON 逐字段一致。
日志位于 `/tmp/physical-addon-phase3-YAWe0V/`。未启动真机，不将离线通过解释为
现场性能或运动效果已经验收。

## 后续合并约定与尚未完成部分

1. exp-setting 的算法变化先合入公共模块；实物参数只改 addon 的配置。
2. 修改公共算法时，优先扩展上述窄接口/通用机制；用离线测试固定公共默认行为。
3. 本分支已有的通用修复尚未反向合入远端 exp-setting，因此当前不是「未改动的
   exp-setting + 可任意拔插的插件」。仍需要按主题整理公共补丁并逐批上游化。
4. 开门后策略、容器两阶段纯策略、证据合同和后端命令组装已抽取；M1 请求路由与
   采集证据缓存、导航执行编排仍在大型节点内，可继续按边界拆分。后续继续保留
   唯一状态所有者与命令取消/过期结果隔离，不整块复制执行器或增加独立控制状态。
5. GMapping/C++ 导航差异不能由 Python 接口替代。本次未改 C++；导航 gitlink 仍为
   `5cff686805546680575a70b1c5433ca6861367f3`，exp-setting 对应 `b9278e8d`。
   合并时单独审计导航补丁，不自动把它替换成上游指针。
6. 本次不重启服务、不启用运动、不推送远端。离线回归和构建通过不等于已经证明
   真机帧率/延迟或运动行为等价；须在安全现场另做无网页→网页观察的 A/B 验收。
