# Full GT 采集迁移到 sim：验证记录

## 范围

- 目标基线：`interactive-nav/sim`，`02b2bbe4cdeea314260413666b1feccdfa580fd5`。
- 独立开发分支：`codex/sim-full-collection`。
- `e16467b8`：场景加载 API、场景镜像资源路径兼容，以及非 RBY1 机器人的可选 Warp 依赖延迟导入。
- `c2201db0`：独立 Full GT 采集、GT 地图几何工具、数据记录、网页、单测及文档。
- 不合并 exp-setting 历史，不迁移 ROS 工程、MLLM、交互图感知或在线导航算法。
- sim 原有 evaluator、任务逻辑、RBY1 控制、robot views、benchmark schema 和 v1.2 benchmark 文件未改变。
- 原实验 worktree 的未提交修改保留；测试产物不进入 Git。

## 自动测试

以下 11 个测试文件共 **106 passed**，耗时 40.97 s：

```bash
PYTHONPATH=. MUJOCO_GL=egl python -m pytest -q \
  mlspaces_tests/data_generation/test_full_gt_collection.py \
  mlspaces_tests/data_generation/test_full_gt_sim_migration.py \
  mlspaces_tests/test_holo_base_yaw_control.py \
  mlspaces_tests/test_rby1_holo_base_range.py \
  mlspaces_tests/test_simulator_scope_parity.py \
  mlspaces_tests/test_nav_task_visibility_cache.py \
  mlspaces_tests/data_generation/test_interactive_nav_v3_benchmark_evaluation.py \
  mlspaces_tests/data_generation/test_interactive_nav_v3_benchmark_cli.py \
  mlspaces_tests/data_generation/test_interactive_nav_benchmark_eval_wrapper.py \
  mlspaces_tests/data_generation/test_benchmark_interaction_executor.py \
  mlspaces_tests/data_generation/test_episode_topdown.py
```

新 worktree 默认资产路径不同，首次未设置已有资产目录时测试收集失败；
显式设置 `MLSPACES_ASSETS_DIR` 后通过，未安装或下载新依赖/资产。
保留 3 条现有警告：2 条 Pydantic 废弃用法、1 条 Objaverse 资产版本提示。
`compileall` 与 `git diff --check` 通过。

## 真实 MuJoCo 采集

本机 Python 3.11 / MuJoCo 3.4 / EGL，读取 sim 自带的
`benchmarks/interactive_nav_v3_procthor10k_val_release_v1_2/mixed.json.gz`。
解压后 SHA256：`07b0c27c7664fd14c633ec042f8eda31ef1844fd4911fc04cce1c0729187a1a8`。

```bash
PYTHONPATH=. MUJOCO_GL=egl OMP_NUM_THREADS=1 \
python scripts/InteractiveNav/collect_full_gt.py \
  --benchmark scripts/InteractiveNav/benchmarks/interactive_nav_v3_procthor10k_val_release_v1_2/mixed.json.gz \
  --output outputs/full_gt_sim_mixed10_20260921 \
  --count 10 --seed 20260921 --speed 0.5 --hz 5 --videos
```

使用同一批索引：765、809、279、168、696、468、981、328、718、883。
结果：**10/10 数据合同检查通过**；2829 帧，5658 张逐帧图片、20 个视频、
10 张俯视轨迹图。累计 episode 墙钟 443.02 s。

与原型最终结果 `full_gt_mixed10_20260921_v2_final` 对比：

- 每条帧数一致，依次为 182、304、203、195、443、371、265、180、320、366。
- 逐帧底盘、操作点/末端目标 XYZ/RPY 在绝对容差 1e-9 下全部一致。
- 阶段、时间轴及完整 instruction JSON 一致。
- 20 个 MP4 经 ffprobe 检查，均为 5 FPS，帧数与轨迹一致。
- 每次运行内部 PNG/H5 像素完全对齐；跨运行存在极少量像素通道差异，
  最大为 1/255，每条相机流最大变化通道比例 3.01e-7，不宣称跨运行字节一致。
- 运行 manifest 保存的采集源码 SHA256 与当前提交内容匹配。
  采集开始时功能文件尚未提交，因此 manifest 的 Git HEAD 为兼容性提交；
  源码指纹才是本轮功能代码的准确标识。

原始证据在输出目录的 `summary.json`、`migration_audit.json` 和各 episode 的
`quality.json`，不提交大型产物。

## 浏览器验收：通过

2026-09-21，执行权限恢复后完成 Chromium 无头浏览器验证：

- 10 条 episode 的第一视角、外部视角、俯视轨迹图均正常加载，末帧图像可切换。
- 俯视 +X 向右、+Y 向上，正视 +Z 向上。
- 正负仰角切换、观察中心抬高至 Z=3、右键平移、左键旋转、滚轮缩放通过。
- 播放推进帧、移动端布局通过，JavaScript 错误为 0。

证据保存为输出目录的 `browser_audit.json` 与 `viewer_preview.png`。
此前审核服务 HTTP 503 仅导致命令未启动，本轮已解除该阻断。
集成目标为 `interactive-nav/sim`，使用独立合并提交保留开发分支历史；
不将原实验分支整体合入，也不推送采集产物。

## 仍存在的边界

- 本轮终点目标可见比例仍全部为 0，与原型一致；不把数据合同通过解释为导航成功。
- 末端为物体把手/操作点对应的目标位姿，不是真实机械臂 TCP 执行轨迹；
  尚未验证 IK、抓取接触、动力学操作或物理锁舌解锁。
- RPY 某些姿态存在欧拉角奇异性；同时保存四元数，避免姿态表达歧义。
- 这里只验证采集迁移与 simulator 协议，未运行 ROS 导航算法六场性能回归，
  不宣称算法性能无下降。
