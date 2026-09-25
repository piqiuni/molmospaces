# 仅重启 Go2 相机并切换采集模式

在仓库根目录运行；本地完整服务和 Go2 相机桥需已启动：

```bash
python3 scripts/InteractiveNav/physical_nav/restart_go2_camera.py \
  --rgb 1280x720@30 --depth 848x480@30 --publish-fps 10
```

- `--rgb`：RGB 分辨率和硬件采集 FPS。
- `--depth`：深度分辨率和硬件采集 FPS。
- `--publish-fps`：向本机发送的目标频率，不改变 YOLO 配置频率。
- 省略任意模式或发送频率，保留当前运行进程的对应参数；全部省略时重启当前模式。
- `--dry-run`：通过 SSH 查看目标进程与拟执行命令，不重启或写入运行状态。

恢复原来的 10 FPS 采集模式：

```bash
python3 scripts/InteractiveNav/physical_nav/restart_go2_camera.py \
  --rgb 1280x720@10 --depth 848x480@10 --publish-fps 10
```

脚本只向精确匹配的机载相机桥发送 TERM，再运行新命令。保留原有 align、
IMU、连接地址和其他未指定参数，不重启本地 ROS、YOLO、网页、Qwen 或动作桥。
完整启动器使用的相机 PID 与所有权记录同步更新。相机断开期间会短暂无新图像、
深度及该桥提供的遥测；应在机器人停止移动时切换模式。

模式必须受当前设备/USB 支持。新进程立即退出时重新拉起原命令，返回失败并打印
日志路径；这不是原模式已恢复输出的保证。启动超时则报告未确认，不宣称成功。
`camera_started=true` 表示检测到相机启动日志，ROS 数据恢复仍应查看实时图像或
话题。当前默认远端日志为 `/home/unitree/physical_nav/go2_readonly_sensor_bridge.log`。

`--ssh-target`、`--bridge-path` 和 `--log` 可覆盖默认连接与路径。脚本不部署机载
代码，不提供曝光/增益参数。提高硬件 FPS 后，即便发送仍为 10 Hz，机载编码负载
仍可能增加。

离线验证：

```bash
conda run -n mlspaces python -m pytest -q \
  scripts/InteractiveNav/physical_nav/tests/test_restart_go2_camera.py
```

测试使用临时假相机进程，覆盖参数保留、精确进程识别、只读预览、失败回退与
运行状态更新，不连接或重启 Go2。
