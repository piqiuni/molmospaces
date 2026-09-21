# Full GT mixed 轨迹采集

仿真器基线：`interactive-nav/sim` 的 `02b2bbe4`；迁移分支：
`codex/sim-full-collection`。原型来自 `codex/instr-goal-full-collection`，
只迁移采集与 GT 几何工具，不引入实验分支的导航算法。

入口 `scripts/InteractiveNav/collect_full_gt.py` 以原始 V3 benchmark 的 oracle plan
为任务顺序，在真实 MuJoCo 场景中生成导航与交互的 GT 运动学示范。
本入口不修改 sim 的 evaluator、任务成功条件或机器人控制协议。
GT occupancy/路径工具独立放在 `collection/gt_map.py`，不依赖实验分支探索脚本。

## 数据语义与边界

- `execution_mode=gt_kinematic`：底盘和关节按参考时间表赋值，经过 MuJoCo
  forward kinematics 后读取状态与渲染；不是机械臂接触控制或真实动力学执行。
- 导航输出世界坐标 `x,y,yaw`；操作点和末端目标输出世界坐标
  `x,y,z,roll,pitch,yaw` 及 `x,y,z,qw,qx,qy,qz`。长度为米、角度为弧度，
  RPY 为 extrinsic xyz。四元数用于避免欧拉角奇异点的歧义。
- 操作点坐标系固定在实际移动 link 上；末端目标沿其局部 z 轴退让
  `--grasp-offset` 米，默认 0.03 m。`ee_target` 不是机器人实际 TCP，
  未进行 IK 可达性、接触抓取、碰撞控制验证。
- 先选择显式把手或小型叶子几何；合并网格内按 OBJ 连通面片识别细长把手；
  再尝试该关节的资产抓取库。没有有效候选时报告失败，不用柜体中心替代。
  `plan.json` 保存来源及 `physical_handle_center_verified`，几何推断不冒充人工确认。
- 有独立 handle joint 的门先转动把手，再打开门板；无独立关节时不制造假动作。
  当前没有物理锁舌约束，始终记录 `latch_simulated=false`。
- 推/拉方向由当前接近侧与关节运动方向计算；不能保证 benchmark 给定的接近侧
  一律可以拉开。`pull` 写入交互元数据，规则指令据此区分“拉开”和“打开”。
- 抽屉中的 benchmark 目标以显式 `oracle_rigid_support` 跟随其支撑 link；
  这是运动学附件关系，不是摩擦/碰撞仿真。关系写入 `plan.json/attachments`。
- 外部相机为跟随底盘朝向的**后方斜上方**视角，隐藏 wall/ceiling 命名的几何以减少遮挡；
  第一视角保留场景。相机参数逐帧保存。
- 第一视角恢复 benchmark 冻结的 head camera 挂载 body、局部位姿及 FOV，
  不直接使用新版 MJCF 的默认安装姿态。外部剖视图的墙体隐藏不影响第一视角。
- 初始化使用与 sim evaluator 数值一致的 `ROS_NAVIGATION_ARM_QPOS` 收臂姿态，
  头部/躯干归零；冰箱的 benchmark `drawer_low_view` 被明确覆盖为水平观察。
  原任务仍保留在 `episode.json`，实际覆盖写入 `plan.json`。

## 停车位置调整（v2）

不覆盖原 benchmark。每次交互前，沿目标把手方向的反向至少后退
`--parking-backoff`（默认 0.45 m），必要时增加距离或侧移，并始终朝向目标把手。
候选点须在当前膨胀 occupancy 内可达；同时预览实际关节从当前状态到目标状态的运动，
以不超过 2° / 1 cm 的间隔获取移动几何的世界 AABB，检查其扫掠区域与停车位的间隙。
同一停车位连续开门/拉抽屉时，顺序预览所有关节，取完整交互序列的扫掠并集。
采用至少 0.4 m 的底盘圆形包络加 0.1 m 余量；无候选时报告失败，不退回不安全原位。
这是离散、保守的平面几何检查，不是整机/机械臂动力学无碰撞证明。

`plan.json/parking_adjustments` 保存原位置、实际 XY/yaw、回退/侧移距离、
扫掠采样数、原/新最小间隙。`paths` 同时保存 `original_plan_step` 和实际 `plan_step`。
由于停车位置变远，末端目标的 IK 可达性仍需后续验证，不能据此宣称实机可以开门。

## 固定速度与固定频率

时间轴为 `t_k=k/hz`，不依赖墙钟和 GPU 吞吐。导航按当前关节状态的膨胀
occupancy 规划，沿路径转向/直行，到达交互位姿后停止底盘并操作物体。
支持多个门、多个容器关节以及交互后的终点导航。全程使用一个场景实例，
不通过重建任务拼接跨模型 qpos。

输入包括 `--speed`（m/s）、`--yaw-speed`（rad/s）、`--hinge-speed`（rad/s）、
`--slide-speed`（m/s）、`--handle-speed`（rad/s）及 `--hz`。
每个直行/转向/交互阶段内部匀速；阶段末端不足一个周期的剩余位移会在下一个
固定采样时刻到达终点，标记 `endpoint_interval=true`，速度不超过输入值。
这避免任意路径长度与固定采样周期不整除时，跳过终点或改变数据频率。
转弯、停留、视角调整分别标注，不计为匀速直行。

## 运行

```bash
conda activate mlspaces
MUJOCO_GL=egl OMP_NUM_THREADS=1 python scripts/InteractiveNav/collect_full_gt.py \
  --benchmark scripts/InteractiveNav/benchmarks/interactive_nav_v3_procthor10k_val_release_v1_2/mixed.json.gz \
  --output outputs/full_gt_mixed10_new_run \
  --count 10 --seed 20260921 --speed 0.5 --hz 5 --videos
```

默认根据容器交互类型轮流抽样，并优先不同房屋。`manifest.json` 固定索引、
case ID、源 SHA256、运行参数和采集代码指纹。`--indices 279 718` 可精确复现
指定失败样本。输出目录必须为空；失败样本不会被静默替换。
使用 sim 的统一加载器读取 JSON 或 JSON.GZ；源 SHA256 指解压后的内容，
另存压缩包 SHA256。已有资产位于其他安装时，通过 `MLSPACES_ASSETS_DIR`
指向该资产目录，避免为新 worktree 重复下载。

脚本默认输出网页；独立重新生成网页：

```bash
python scripts/InteractiveNav/render_full_gt.py outputs/full_gt_mixed10_new_run
```

直接用浏览器打开输出 `index.html`，无需网络或第三方 CDN。左键旋转，右键或
Shift+拖动平移，滚轮缩放；仰角滑块支持从地面下方至正俯视，Z 输入抬高观察中心。
提供斜上方、俯视 XY、正视 XZ、侧视 YZ 快捷视角。右手坐标 +Z 朝上，
俯视时 +X 向右、+Y 向上，yaw 从 +X 转向 +Y 为正。
可选择 episode、拖动时间轴和播放，同帧显示第一视角和后上方视角；另有
640×640 终态场景俯视图及完整 GT 路径，不将静态总览误标为逐帧图像。
分享时保留输出目录相对结构，不能只复制 HTML 而丢弃图片。

## 产物

```text
run/
  manifest.json, summary.json, index.html
  episode_NNNN/
    episode.json                 原始任务
    trajectory.h5                同步图像、动作、状态、具名 GT 位姿
    frames.json                  逐帧位姿、阶段、相机参数
    plan.json                    路径、阶段范围、操作点、附件关系
    instruction.json             规则指令及语句与帧区间对应
    model_layout.json            关节名称和 qpos/qvel 索引
    quality.json                 完整性、速度和对齐检查
    images/head/NNNNNN.png
    images/external/NNNNNN.png
    head.mp4, external.mp4       指定 --videos 时生成
    topdown_scene.png            终态场景俯视图
    topdown_trajectory.png       俯视图 + 完整 GT 路径 / 起终点
    topdown.json                 世界坐标到图像的投影参数与路径像素
```

H5 `steps/` 保留现有 full recorder 格式。新增 `gt/` 包含
`base_xy_yaw`、`operation_valid`、`operation_point_xyz_rpy`、
`operation_point_xyz_quat_wxyz`、`ee_target_xyz_rpy`、
`ee_target_xyz_quat_wxyz` 和 `joint_value`。
非交互帧操作点数组使用 NaN，并由 `operation_valid=false` 标记；JSON 对应 null。
`qvel` 是相邻 GT 状态的有限差分速度，不是动力学积分读数。
`overview/` 额外保存俯视原图和轨迹叠图，`projection_json` 保存投影信息。

规则 instruction 从实际记录的移动、转向、把手转动、开合及观察阶段生成，
不调用 LLM/VLM。v2 使用 0.35 m 容差简化**叙述路径**，只明确描述 ≥45° 的大转向；
小幅转向合并为直行，实际行走距离写在括号内，不输出精确转角。
把手转动与开门合并为一句，仍保存句子所覆盖的帧范围。
此压缩不改变采集路径或逐帧姿态。图像与姿态由同一个采样状态生成，不插帧。

## 验证

```bash
MUJOCO_GL=egl python -m pytest -q \
  mlspaces_tests/data_generation/test_full_gt_collection.py \
  mlspaces_tests/data_generation/test_full_gt_sim_migration.py
```

每次真实采集检查 PNG 与 H5 每帧像素完全一致、时间均匀、位姿有限、四元数
单位长度、输入速度、起终点、开合关节终值、交互顺序及指令帧范围。
v2 还检查后上方相机、全程收臂、冰箱水平视线、停车扫掠间隙及俯视 PNG/H5 一致性。
`passed` 表示这些数据合同检查通过，不代表机械臂物理执行通过。
终点目标距离和可见比例另外记录，不把数据合同检查冒充导航策略成功率。
