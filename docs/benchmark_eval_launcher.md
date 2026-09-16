# Benchmark 批量评测入口

从仓库根目录运行：

```bash
python scripts/InteractiveNav/run_benchmark_eval.py
```

默认配置为 `scripts/InteractiveNav/configs/evaluation/benchmark_batch.json`：
10 worker，共 10 场（Channel 10–12、Container 1010–1012、Mixed 2010–2013），dynamic 模式、2000 步硬上限，
Qwen 8010，完整录制。启动器本身只使用 Python 标准库；仿真解释器由配置中的
`python_bin` 和 `conda_env` 显式指定，不依赖当前激活的 Conda 环境。

```bash
# 只预览命令，不启动或创建输出目录
python scripts/InteractiveNav/run_benchmark_eval.py --dry-run
# 临时调整并发，其他参数沿用配置
python scripts/InteractiveNav/run_benchmark_eval.py --workers 30
# 指定场景、并发、预算、录制和输出位置（输出目录必须尚不存在）
python scripts/InteractiveNav/run_benchmark_eval.py \
  --episode-indices 10 11 12 1010 1011 1012 2010 2011 2012 2013 \
  --workers 10 --max-steps 200 --recording \
  --output-dir /home/ldl/outputs/interactive-nav/my-eval
# 关闭完整录制
python scripts/InteractiveNav/run_benchmark_eval.py --no-recording
# 使用另一份配置
python scripts/InteractiveNav/run_benchmark_eval.py --config /absolute/path/config.json
```

终端立即显示配置和输出目录，之后每 10 秒刷新整体进度与已完成场景的均值。
“完成”包括评测及视频收尾完成，不等于导航成功；“运行异常”指 runner 或产物失败。
进度读取 batch summary，不依赖录制的 trajectory.csv；未完成和运行异常场景不进入完成均值。
初始化、仿真及视频收尾均归入“运行/收尾”。每场实际动态预算可在 eval.log 的
`episode-step-budget` 记录中核对。当前观察轮次倍率为 1，观察轮次也受场景预算约束。

每次运行创建独立输出目录，保存 `launch_config.json` 和 `batch.log`。
详细日志位于 `episode_*/attempt_*/runner.log`、`eval.log`、`roslaunch.log`。
最终汇总和视频仍由原有 batch 推理脚本生成。

结束时终端自动显示完成/异常/未报告数量、正式/任务/导航成功率、平均 SPL、
步数、实际及 GT 路径、必要交互和序列成功率、终止原因分布、批次及单场耗时、
循环加权秒/step、已完成场景循环数除以批次墙钟的吞吐量、主机 CPU/内存与 GPU 峰值、视频及指标位置。
同一报告保存为 `completion_report.txt`。评分来自 batch 已生成的结果，不重复运行评测；
每项成功率显式列出有效样本分母，缺失数据显示 N/A。中断或运行异常标注为部分结果，
不把未报告场景当作已完成的导航失败。资源统计包含同机模型服务及其他进程。

Ctrl+C 或 SIGTERM 会停止本轮 batch 及继承了本轮唯一标记的子进程，
包括脱离原父进程的 ROS 节点。正常或异常退出也执行同样清理。
首次 Ctrl+C 发出 TERM，最多等待 8 秒，然后对残留进程发出 KILL 并复查；
再次 Ctrl+C 跳过剩余宽限时间。清理失败会报告残留 PID，不会宣称清理成功。
清理不匹配通用进程名，不停止外部模型服务，不删除产物。
直接 SIGKILL 启动器或机器掉电无法执行清理逻辑。

动态预算 v2：基础 200；GT 路径超过 3 米的部分每米加 40；每次必要通道交互加 200；
每次必要容器交互加 250；存在必要容器交互时，目标容器每个记录的关节另加 50（含第一个）。
未知交互类型按 250 计。初始目标可见时取消交互相关加成。
总和向上取整到 50 的倍数，再受 `--max-steps` 硬上限限制；GT 路径缺失时使用硬上限。
例如动态计算为 700，传 `--max-steps 2000` 实际预算仍为 700；传 200 则预算为 200。
