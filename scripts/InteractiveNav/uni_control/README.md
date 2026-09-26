# Go2 统一导航、状态与语音控制

本目录实现一条统一的 policy → WebSocket → Go2 控制链路，面向 VLN、Habitat 离散动作、
ROS `move_base` 连续速度和交互导航语音提示。

当前支持：

- `discrete`：Habitat/VLN 风格的原子移动和转向动作。
- `continuous`：ROS `geometry_msgs/Twist` 连续速度，默认订阅 `/cmd_vel`。
- `lidar`：开启、关闭或切换 Unitree LiDAR。
- `posture`：显式卧倒和站立。
- `speak`：固定英文开门/冰箱/抽屉请求优先播放预生成 WAV，启动时预上传 AudioHub；
  其它中文请求使用本地 Matcha，英文请求不送入中文 Matcha，而使用英文 Edge TTS。

固定音频使用本机已有 FFmpeg/libflite 离线生成，无需下载模型：

```bash
python scripts/InteractiveNav/uni_control/prepare_interaction_audio.py
paplay scripts/InteractiveNav/uni_control/audio/open_door_en.wav
```

部署时必须同时复制 `speech_assets.py` 和 `audio/*.wav` 到 Go2 对应目录。
缓存键包含音频内容哈希，避免复用旧中文模型对英文生成的无效音频。
若本机 PulseAudio 只有 `auto_null`，播放命令成功不代表硬件出声。
- `telemetry`：回传 `rt/sportmodestate`、控制状态和语音后端状态。

运动、遥测和语音队列彼此隔离。语音合成或 AudioHub 播放失败不会阻塞运动 watchdog；
WebSocket 断开、TTL 过期或 bridge 退出都会使速度归零。

交付边界：本项目自行维护的两端接口与运行代码均在本目录中；Unitree SDK2、CycloneDDS、
WebRTC 驱动和语音模型是外部开源依赖，不复制进仓库，其固定版本和从零部署方法见
[SDK_DEPLOYMENT.md](./SDK_DEPLOYMENT.md)。

## 1. 系统结构

正常部署涉及两台机器：

```text
zgca_gpu / policy 机
  policy_control_server.py
  WebSocket: 127.0.0.1:12333
              ⇅
  Go2 主动建立 SSH 本地转发
  -L 127.0.0.1:12333:127.0.0.1:12333
              ⇅
Go2 Jetson
  start_go2_control.py
    ├── SSH tunnel supervisor
    └── go2_control_bridge.py
          ├── Unitree SDK2 / CycloneDDS: eth0
          ├── rt/sportmodestate
          ├── ObstaclesAvoidClient / SportClient / LiDAR switch
          ├── Matcha TTS，Edge TTS fallback
          └── WebRTC AudioHub: 192.168.123.161
```

`127.0.0.1:12333` 在 Go2 上不是直接指向 policy 进程，而是 SSH 隧道的本地入口。当前 Go2
上的 `zgca_gpu` SSH alias 已验证为 `user@10.100.5.3:22022`，可以免密登录。

launcher 建立的等价隧道为：

```bash
ssh -N -T \
  -o BatchMode=yes \
  -o ExitOnForwardFailure=yes \
  -o ConnectTimeout=5 \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:12333:127.0.0.1:12333 \
  zgca_gpu
```

launcher 同时监控 tunnel 和 bridge。任一进程退出时，它会先停止 bridge、归零速度并释放
API 控制权，再关闭隧道，默认等待 2 秒后重新建立整条链路。

## 常驻控制桥的运动切换

实物导航已支持在同一个 Go2 控制桥进程内开启/暂停运动输出：

```bash
bash scripts/InteractiveNav/physical_nav/physical_nav_all.sh start_control enable_motion
bash scripts/InteractiveNav/physical_nav/physical_nav_all.sh stop_control
```

六面板网页的“开启运动／暂停运动”按钮调用上述接口。切换保留相机、YOLO、OCC、
语义图、WebSocket 和语音进程，不清空地图。首次部署新版桥需要重新加载一次控制桥；
SDK 首次初始化或重新申请控制权限仍可能耗时，失败时输出门保持关闭。

控制桥通过与 ready 文件同目录、权限为 `0600` 的 `go2-motion-<pid>.sock` 接收
SSH 转发的本地请求。`motion_switch.py` 需与 `go2_control_bridge.py` 一起部署。
暂停时清零并释放 API 运动权限；暂停/切换期间的速度消息直接丢弃，开启后等待新指令。
启动参数 `--enable-motion` 仅表示初始状态；运行状态应查询此 socket 或总脚本的 `status`。
原来的网页 `Stop` / Esc / S 仍可直接中断本机控制传输。

## 2. 文件与部署位置

| 文件 | 运行位置 | 用途 |
|---|---|---|
| `start_go2_control.py` | Go2 | 建立 SSH 隧道，生成 bridge 参数并监控两个子进程 |
| `go2_control_bridge.py` | Go2 | 协议分发、运动控制、遥测、LiDAR、姿态和语音队列 |
| `go2_voice_intercom.py` | Go2 | Matcha/Edge 合成、WebRTC AudioHub、麦克风采集和本地 Whisper |
| `policy_control_server.py` | policy 机 | WebSocket 服务；键盘、stdin 或 ROS 输入 |
| `control_protocol.py` | 两端 | `v=1` 消息定义、验证、动作枚举和序列化 |
| `benchmark_tts.py` | Go2 | 无播放地比较本地和在线 TTS 延迟 |
| `inspect_go2_interfaces.py` | Go2 | 只读检查 SDK、相机、VUI、服务和音频端点 |
| `audio_io_test.py` | Go2 | 麦克风统计和显式扬声器测试 |
| `test_speech_control.py` | 开发机 | 协议、队列与 Matcha→Edge 回退单元测试 |
| `requirements_go2.txt` | Go2 | Go2 端附加 Python 依赖 |
| `requirements_policy.txt` | policy 机 | policy 端 Python 依赖 |
| `SDK_DEPLOYMENT.md` | 两端 | SDK2、WebRTC、TTS/Whisper 模型来源、固定版本和换机部署方法 |

开发机源码目录：

```text
/home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control
```

Go2 持久部署目录：

```text
/home/unitree/uni_control
```

Go2 上的模型和隔离依赖位于：

```text
/home/unitree/uni_control/vendor/python       # aiortc、PyAV、edge-tts、sherpa-onnx
/home/unitree/uni_control/vendor/local_tts    # Aishell3、Matcha、MeloTTS、Vocos
/home/unitree/uni_control/vendor/whisper.cpp  # ARM64 whisper-cli、ggml-base.bin
```

## 3. 快速启动

### 3.1 第一步：在 policy 机启动服务

离散键盘控制：

```bash
python3 /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/policy_control_server.py \
  --control-mode discrete \
  --source keyboard
```

`--source auto` 在离散模式下等价于 `keyboard`，在连续模式下等价于 `ros`。

### 3.2 第二步：在 Go2 启动安全模式

```bash
python3 /home/unitree/uni_control/start_go2_control.py
```

默认行为：

- 读取真实 `rt/sportmodestate`。
- 初始化 LiDAR 和姿态辅助接口。
- 不初始化速度运动客户端，因此移动命令不会驱动机器人。
- 以 4 个 CPU 线程加载 Matcha + Vocos，并常驻复用模型。
- Matcha 加载或单句合成失败时启用 Edge TTS 回退。
- 建立到 `zgca_gpu` 的 SSH 隧道并持续重连 WebSocket。

以下日志表示本地语音后端准备完成：

```text
Matcha TTS loaded in ...s (4 threads); Edge fallback enabled=True
```

sherpa-onnx 加载中文词典时可能打印 `Unknown token: shei2`；当前实机测试中，只要随后出现
上述 `Matcha TTS loaded`，模型仍能正常合成。

### 3.3 确认安全链路后开启运动

确保 Go2 周围无人员、台阶和障碍物，并已验证 `STOP`、断开 WebSocket 和停止 launcher 都会
归零后，重新启动：

```bash
python3 /home/unitree/uni_control/start_go2_control.py --enable-motion
```

`--enable-motion` 会初始化 `ObstaclesAvoidClient`、开启避障服务并调用
`UseRemoteCommandFromApi(True)`。launcher 退出时会下发零速度并恢复为 `False`。

### 3.4 停止

在 Go2 launcher 终端按 `Ctrl+C`。launcher 会先中断 bridge，让 bridge 归零、取消离散原语、
停止语音线程并释放运动控制，再关闭 SSH 隧道。随后停止 policy 端服务。

不要通过强制断电代替正常退出；不要同时运行旧控制脚本和本目录的 bridge。

## 4. 键盘、stdin 与 ROS 输入

### 4.1 键盘映射

离散模式：

| 按键 | 动作 |
|---|---|
| `w` / `s` | 前进 / 后退一个原子步 |
| `a` / `d` | 左移 / 右移一个原子步 |
| `q` / `e` | 左转 / 右转一个原子步 |
| `x` | 立即停止并取消当前原语 |
| `l` | 切换 LiDAR；状态未知时第一次固定发送 `OFF` |
| `,` / `.` | `StandDown()` / `StandUp()` |
| `t` | 暂停按键读取，输入一整行待播报文本 |

每个非 `STOP` 离散动作都是原子的。上一个原语未完成时，新离散动作不会排队，bridge 返回
`applied=false`，防止键盘自动重复变成连续多步。

连续键盘模式：

| 按键 | 动作 |
|---|---|
| `w` | 前进 |
| `a` / `d` | 左转 / 右转 |
| `s`、空格或 `x` | 停止 |
| `q` | 停止并退出 policy 服务 |
| `l`、`,`、`.`、`t` | 与离散模式相同 |

按 `t` 后，输入期间会暂停遥测摘要打印。默认只等待语音排队 ACK，朗读时仍可继续控制；如需
阻塞键盘直至播放结束：

```bash
python3 /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/policy_control_server.py \
  --control-mode discrete \
  --source keyboard \
  --speech-volume 4 \
  --wait-speech-completion
```

`--speech-volume` 范围为 Go2 VUI 的 `0–10`；省略则保留当前音量。

### 4.2 Habitat/VLN stdin

stdin 每行接受动作名、动作编号或 JSON：

```bash
python3 /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/policy_control_server.py \
  --control-mode discrete \
  --source stdin \
  --wait-discrete-completion
```

示例输入：

```text
MOVE_FORWARD
2
{"action":"MOVE_RIGHT"}
{"type":"speak","text":"我到门口了，请帮我开门","volume":4,"wait":true}
0
```

`--wait-discrete-completion` 使 stdin 调用在收到原语最终 `status` 后再读取下一行，适合
Habitat/VLN 的一步一观测循环。代码内可直接复用：

```python
move_result = await server.publish_discrete(
    "MOVE_FORWARD",
    ttl_ms=6000,
    wait_for_completion=True,
)

speech_result = await server.publish_speech(
    "我到门口了，请帮我开门",
    volume=4,
    wait_for_completion=True,
)
```

### 4.3 ROS `move_base`

```bash
source /opt/ros/noetic/setup.zsh
source /home/user/ldl/molmospaces/Interactive-Nav-SG-nav/devel/setup.zsh
python3 /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/policy_control_server.py \
  --control-mode continuous \
  --source ros \
  --cmd-vel-topic /cmd_vel
```

ROS 输入类型固定为 `geometry_msgs/Twist`。如果实际话题是 `TwistStamped`，不能只修改 topic
名称；应使用 move_base 原生 `/cmd_vel`，或增加明确的 `TwistStamped → Twist` relay。

policy 端仅保留最新一条 ROS 速度，队列大小为 1。bridge 断开时丢弃发送；正常退出 ROS
source 时会尽力发送一次零速度。

实物交互 policy 可以复用同一条 ROS source 发布语音，不需要建立第二条
WebSocket 连接。指定 `--speech-request-topic` 后，`std_msgs/String` 的纯文本或
`{"text":"...","wait":true,"volume":4}` JSON 会被转换为 `speak` 消息并沿当前
Go2 bridge 播放：

```bash
python3 policy_control_server.py --control-mode continuous --source ros \
  --cmd-vel-topic /physical_nav/actuated_cmd_vel \
  --speech-request-topic /physical_nav/speech_request
```

## 5. WebSocket 协议 `v=1`

所有命令共用一个严格递增的 `seq`。bridge 每次建立新的 WebSocket 连接时将接收序列重置为
`-1`。运动、LiDAR、姿态和语音不能各自维护独立序列。

### 5.1 离散动作

| action_id | action | 默认执行方式 |
|---:|---|---|
| 0 | `STOP` | 立即零速度并取消当前原语 |
| 1 | `MOVE_FORWARD` | 闭环前进 0.25 m |
| 2 | `TURN_LEFT` | 有界开环左转，目标参数 30° |
| 3 | `TURN_RIGHT` | 有界开环右转，目标参数 30° |
| 4 | `MOVE_BACKWARD` | 闭环后退 0.25 m |
| 5 | `MOVE_LEFT` | 闭环左移 0.10 m |
| 6 | `MOVE_RIGHT` | 闭环右移 0.10 m |

其中 0–3 保持 Habitat/MemoryVLN 的原映射，4–6 是当前键盘和实物测试使用的扩展动作。

```json
{"v":1,"type":"control","seq":1,"control_mode":"discrete","ttl_ms":6000,"action":"MOVE_FORWARD"}
```

平移动作使用起始姿态坐标系中的 `rt/sportmodestate` 位置闭环，并用 yaw 误差修正航向。转向
默认使用 `open_loop`：按目标角度、最大角速度和启动余量计算有界持续时间；当前默认约
0.60 秒，硬超时 1 秒。这是因为现场 yaw 反馈会低估实际转角。可用 launcher 参数
`--turn-control-mode closed_loop` 切回 yaw 闭环。

bridge 先返回 `ack`，随后返回 `completed`、`failed` 或 `cancelled` 的 `status`：

```json
{"v":1,"type":"status","seq":1,"control_mode":"discrete","action":"MOVE_FORWARD","state":"completed"}
```

policy 默认离散 TTL 为 6000 ms；bridge 最大接受 7000 ms。原语自己的 timeout 与消息 TTL
取更小值。

### 5.2 连续速度

```json
{"v":1,"type":"control","seq":2,"control_mode":"continuous","ttl_ms":350,"velocity":{"linear_x":0.1,"linear_y":0.0,"angular_z":0.2}}
```

字段对应 `Twist.linear.x`、`linear.y`、`angular.z`。bridge 默认限幅：

- `|linear_x| <= 0.25 m/s`
- `linear_y = 0 m/s`
- `|angular_z| <= 0.40 rad/s`

默认连续 TTL 为 350 ms，bridge 最大接受 500 ms。消息过期后本地控制循环归零。新的连续
命令会取消正在执行的离散原语；新的离散命令也会清空连续速度。

兼容旧消息：带 `type=cmd` 和 `velocity` 的消息会被推断为 `continuous`，但新代码应显式
发送 `control_mode`。

### 5.3 LiDAR 与姿态

```json
{"v":1,"type":"lidar","seq":3,"action":"toggle"}
{"v":1,"type":"posture","seq":4,"action":"stand_down"}
{"v":1,"type":"posture","seq":5,"action":"stand_up"}
```

LiDAR `action` 可为 `toggle`、`on`、`off`，向 `rt/utlidar/switch` 发布 `ON/OFF`。bridge
重启后无法读取实际开关状态，默认记为 `unknown`，因此第一次 `toggle` 固定发送 `OFF`。

姿态切换前会归零、取消当前原语并暂停速度控制。`stand_up` 完成后会调用
`BalanceStand()`，再恢复避障服务和 API 运动控制权。姿态命令在未启用速度运动时仍可用，
所以必须由用户显式按键或发送 JSON，bridge 不会自动改变姿态。

### 5.4 文本播报

```json
{"v":1,"type":"speak","seq":6,"text":"我到门口了，请帮我开门","voice":"zh-CN-XiaoxiaoNeural","volume":4,"ttl_ms":30000}
```

字段：

- `text`：非空，bridge 默认最多 300 个字符。
- `voice`：最长 80 个字符；仅 Edge 主后端或回退时生效。
- `volume`：可选，范围 `0–10`；省略时保持机身当前音量。
- `ttl_ms`：排队有效期，协议默认 30000 ms，bridge 接受 1000–120000 ms。

policy 键盘默认使用 60000 ms；语音队列容量默认为 3。收到消息后 bridge 立即返回
`state=queued` 的 ACK，独立线程依次处理。Matcha 模型只在 bridge 启动时加载一次；每句本地
音频从 22.05 kHz 转为 AudioHub 已验证的 44.1 kHz WAV。AudioHub 上传记录在播放后删除，
VUI 音量也会恢复原值。

成功状态示例：

```json
{"v":1,"type":"status","seq":6,"command_type":"speech","state":"completed","text_chars":12,"elapsed_s":7.2,"requested_synthesis_backend":"matcha","synthesis_backend":"matcha","total_synthesis_seconds":0.64}
```

`elapsed_s` 包含合成、WebRTC 连接、上传和实际播放；`total_synthesis_seconds` 只表示合成。
如果 Matcha 单句失败并成功回退，状态还包含 `fallback_from=matcha` 和 `fallback_reason`。
播放失败返回 `state=failed + detail`，排队超时返回 `state=expired`。

### 5.5 hello 与遥测

连接建立后 bridge 先发送 `type=hello`，声明：

- `supported_control_modes`：`discrete`、`continuous`
- `supported_auxiliary_commands`：`lidar`、`posture`、`speak`
- 当前 LiDAR、运动、原语和语音后端状态

bridge 默认每 0.2 秒发送一条 `type=telemetry`。`sport_mode_state` 包含：

- `position`、`velocity`、`yaw_speed`
- IMU `quaternion`、`gyroscope`、`accelerometer`、`rpy`、`temperature`
- `mode`、`gait_type`、`progress`、`body_height`、`foot_raise_height`
- `range_obstacle`、`foot_force`、`error_code`
- `battery`（来自 `rt/lowstate`，回调内节流到 5 Hz 刷新）：`soc` 剩余电量、
  `voltage`/`current` 实时电压电流、`power` 实时功率（`voltage * current`）、
  `bms_current` 原始 BMS 电流计数、`cycle` 循环次数、`cell_vol` 15 节电芯分压、
  `bq_ntc`/`mcu_ntc` 温度、`temperature_ntc1/2`、`fan_frequency`

外层状态包含：

- `active_mode`、`primitive`、`motion_suspended`
- `lidar_enabled`
- `speaker_state`、`speaker_active_seq`、`speaker_queue_depth`
- `speaker_synthesis_backend`、`speaker_fallback_backend`
- `speaker_last_synthesis_backend`、`speaker_backend_error`

`speaker_state` 可能为 `loading`、`idle`、`speaking`、`disabled` 或加载失败时的 `failed`。
policy 端将完整消息保存在 `latest_telemetry`，终端只按
`--telemetry-print-period` 输出摘要。

## 6. launcher 与语音配置

### 6.1 常用 launcher 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--enable-motion` | 关闭 | 初始化真实速度客户端 |
| `--interface` | `eth0` | Unitree DDS 网卡 |
| `--ssh-target` | `zgca_gpu` | Go2 的 SSH alias |
| `--local-port` / `--server-port` | `12333` | 隧道两端端口 |
| `--telemetry-period` | `0.20` | 遥测发送周期，秒 |
| `--forward-distance` | `0.25` | 前后原子步，米 |
| `--lateral-distance` | `0.10` | 横移原子步，米 |
| `--turn-angle-deg` | `30.0` | 转向目标参数，度 |
| `--turn-control-mode` | `open_loop` | `open_loop` 或 `closed_loop` |
| `--restart-delay` | `2.0` | 子进程失败后的重启间隔，秒 |
| `--no-restart` | 关闭 | 子进程退出后不重启 |
| `--print-command` | 关闭 | 只打印 tunnel 和 bridge 命令，不启动 |

查看默认展开命令：

```bash
python3 /home/unitree/uni_control/start_go2_control.py --print-command
```

不在 launcher 顶层暴露的 bridge 参数可以使用重复的 `--bridge-arg` 传入，例如：

```bash
python3 /home/unitree/uni_control/start_go2_control.py \
  --enable-motion \
  --bridge-arg=--max-vx \
  --bridge-arg=0.20
```

### 6.2 TTS 后端

默认配置等价于：

```bash
python3 /home/unitree/uni_control/start_go2_control.py \
  --speech-primary-backend matcha \
  --speech-fallback-backend edge \
  --tts-threads 4
```

只使用在线 Edge，不加载 Matcha：

```bash
python3 /home/unitree/uni_control/start_go2_control.py \
  --speech-primary-backend edge
```

严格离线：模型加载失败时 bridge 退出并由 launcher 重试；单句合成失败时返回失败，不联网：

```bash
python3 /home/unitree/uni_control/start_go2_control.py \
  --speech-fallback-backend disabled
```

模型路径可通过 `--matcha-model-dir` 和 `--matcha-vocoder` 覆盖。`--tts-threads` 接受 1–8。

## 7. 依赖与验证

本节给出当前环境摘要。换机、重装或复现实机依赖时，使用完整的
[SDK_DEPLOYMENT.md](./SDK_DEPLOYMENT.md)；SDK 和大型模型属于外部资产，不直接提交到本目录。

### 7.1 当前 Go2 环境

- Ubuntu / Python 3.8.10 / ARM64 / 8 CPU
- `unitree_sdk2py`：`/home/unitree/unitree_sdk2_python`
- `cyclonedds==0.10.2`
- `websocket-client==1.8.0`
- `aiortc==1.9.0`
- `edge-tts==7.2.8`
- `sherpa-onnx==1.13.2`
- `pyrealsense2==2.55.1.6486`，外接 RealSense D435i

Unitree SDK2 和 CycloneDDS 由机器人镜像/SDK checkout 提供。不要为本模块重新安装或替换
这两项。其余附加依赖已隔离部署在 `/home/unitree/uni_control/vendor/python`。

policy 端建议 Python 3.11+，安装：

```bash
python3 -m pip install -r \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/requirements_policy.txt
```

ROS source 还需要 ROS 1 Noetic 的 `rospy` 和 `geometry_msgs`。

### 7.2 单元测试和语法检查

```bash
python3 -m py_compile \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/start_go2_control.py \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/go2_control_bridge.py \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/go2_voice_intercom.py

python3 -m unittest discover \
  -s /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control \
  -p 'test_*.py'
```

当前测试覆盖语音协议、policy ACK/status、非阻塞队列和 Matcha 加载失败时选择 Edge 回退。

## 8. 音频、TTS 与接口实测

### 8.1 TTS 基准

2026-08-26 在 Go2 上以 4 线程、每项 3 次进行了无播放基准。短、中、长文本分别为 12、
45、110 个字符：

| 后端 | 采样率 | 短文本 | 中文本 | 长文本 | 中文本平均 RTF |
|---|---:|---:|---:|---:|---:|
| Aishell3 VITS | 8 kHz | 0.179 s | 0.526 s | 1.254 s | 0.069 |
| Baker Matcha + Vocos | 22.05 kHz | 0.400 s | 1.321 s | 3.004 s | 0.153 |
| MeloTTS zh/en | 44.1 kHz | 1.962 s | 6.795 s | 16.860 s | 0.870 |
| Edge Xiaoxiao，联网 | MP3 | 1.505 s | 1.971 s | 2.495 s | 0.202 |

Matcha 在当前 bridge 中实测加载约 3.45 秒，常驻后短提示可在亚秒级生成。Whisper-base
回读表明 Matcha、Melo 和 Edge 的中文本正文基本完整，Aishell3 错字更多。综合常用短提示
延迟、采样率和可懂度，当前默认使用 Matcha。

复现基准，不调用 AudioHub 或扬声器：

```bash
PYTHONPATH=/home/unitree/uni_control/vendor/python \
python3 /home/unitree/uni_control/benchmark_tts.py \
  --backends aishell3,matcha,melo,edge \
  --repeats 3 \
  --threads 4
```

结果位于：

```text
/home/unitree/uni_control/benchmarks/tts_benchmark.json
/home/unitree/uni_control/benchmarks/samples/
```

### 8.2 Go2 交互导航相关接口

| 能力 | 当前状态 | 接口/证据 |
|---|---|---|
| 高层速度控制 | 可用 | `ObstaclesAvoidClient.Move` |
| 避障开关和 API lease | 可用 | `SwitchGet/SwitchSet`、`UseRemoteCommandFromApi` |
| 机体里程计、IMU、速度 | 可用 | DDS `rt/sportmodestate` |
| 前置原生相机 | 可用 | `VideoClient.GetImageSample()`，约 104 KB JPEG |
| 外接 RGB-D | 已连接 | RealSense D435i、`pyrealsense2` |
| Unitree LiDAR 开关 | 可用 | DDS `rt/utlidar/switch` |
| 低层状态 | 已订阅电量/功率 | `rt/lowstate` → 遥测 `battery` 字段（SOC、电压、电流、功率、电芯分压、温度）；电机、足端力、遥控数据仍不转发 |
| 姿态 | 可用 | `SportClient.StandDown/StandUp/BalanceStand` |
| VUI | 可用 | 开关、音量 `0–10`、亮度 |
| 机身扬声器 | 已验证 | WebRTC AudioHub 上传、播放、状态和删除 RPC |
| 机身麦克风 | 暂不可用于 ASR | WebRTC 有非零 PCM，但当前样本以宽带噪声为主 |
| Jetson ALSA/PulseAudio | 不是机身音频入口 | APE/ADMAIF 虚拟端点 |

### 8.3 麦克风和实验性语音交互

`go2_voice_intercom.py` 的命令行实验仍使用 Edge TTS 播报，再监听一句回复并调用 Go2 本地
Whisper 转写/翻译：

```bash
python3 /home/unitree/uni_control/go2_voice_intercom.py \
  --text "我到门口了，请帮我开门" \
  --volume 4
```

这与 bridge 的默认 Matcha 后端不同。当前机身麦克风录音以噪声为主；外接 USB 麦克风接入
前，不应将转写结果用于控制或决策。原始回复默认不保留，只有显式使用 `--save-reply` 才会
写入指定路径。

只读接口复查：

```bash
python3 /home/unitree/uni_control/inspect_go2_interfaces.py --interface eth0
```

扬声器测试会产生真实声音，只有现场明确允许时才运行 `audio_io_test.py --speaker ...`。

## 9. 故障排查

### policy 服务没有 bridge 连接

1. 在 policy 机确认 `127.0.0.1:12333` 正在监听。
2. 在 Go2 执行 `ssh -o BatchMode=yes zgca_gpu true` 验证免密登录。
3. 用 `start_go2_control.py --print-command` 检查端口和路径。
4. 确认没有第二个 bridge；policy 服务只允许一个活动连接。

### 能收到遥测但机器人不移动

- 默认 launcher 故意禁用速度运动；确认使用了 `--enable-motion`。
- 确认 Go2 已站立并完成 `BalanceStand`。
- 检查日志中避障服务和 `UseRemoteCommandFromApi(True)` 是否成功。
- 检查命令 TTL、`applied` ACK 和最终离散 `status`。
- 连续模式默认禁止 `linear_y`；横移只由离散 `MOVE_LEFT/RIGHT` 使用独立限幅。

### 文本没有播报或回退到 Edge

- 查看启动日志是否出现 `Matcha TTS loaded`。
- 查看遥测中的 `speaker_synthesis_backend` 和 `speaker_backend_error`。
- 查看最终状态中的 `synthesis_backend`、`fallback_reason` 和 `detail`。
- Edge 回退需要 Go2 能访问互联网；严格离线时使用 `--speech-fallback-backend disabled`。
- Go2 机身扬声器必须走内部主控 `192.168.123.161` 的 WebRTC AudioHub，不能用 Jetson
  默认 ALSA/PulseAudio 端点代替。

### 离散动作没有连续执行

这是预期行为：bridge 不排队新的非 `STOP` 原语。policy 应等待上一动作 `completed` 后再发送
下一动作；不要依赖键盘自动重复。

## 10. 安全边界

1. 首次联调先使用不带 `--enable-motion` 的安全模式。
2. 真实运动前清空机器人周围人员、台阶、线缆和易碎物。
3. 保留原厂遥控器和人工接管能力。
4. 先分别验证 `STOP`、断开 policy、停止 tunnel 和 `Ctrl+C` launcher 都能归零。
5. move_base 的地图尺度、坐标方向和速度符号必须低速、单轴验证。
6. 姿态切换可能重置避障服务和 API lease；等待 `StandUp + BalanceStand` 完成后再移动。
7. 不同时运行旧 bridge、本 bridge 或其他占用运动服务的程序。
8. 不把当前机身麦克风的噪声转写结果用于自主决策。
9. TTS、扬声器和麦克风测试涉及办公室环境时，先确认音量、时长和现场许可。
