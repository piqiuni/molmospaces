# Go2 SDK 与外部依赖部署说明

本文只说明 `uni_control` 使用的外部开源 SDK、Python 包和模型如何部署。项目自己维护的
policy、WebSocket 协议、Go2 bridge、launcher、控制逻辑和语音后端适配代码均在本目录中，
不依赖其他私有代码仓库。

本文记录的是 2026-08-26 当前实机验证基线。不要直接覆盖一台正在工作的 Go2 环境；换机或
重装时先在新目录安装并完成只读验证，再切换正式 launcher。

## 1. 依赖边界

### 1.1 项目自有代码

以下代码必须从本目录部署到 Go2 或 policy 机：

- `control_protocol.py`
- `policy_control_server.py`
- `start_go2_control.py`
- `go2_control_bridge.py`
- `go2_voice_intercom.py`
- `inspect_go2_interfaces.py`
- `audio_io_test.py`
- `benchmark_tts.py`

### 1.2 外部必需依赖

| 依赖 | 用途 | 当前实机版本/固定点 |
|---|---|---|
| Unitree SDK2 Python | DDS 状态、运动、姿态、LiDAR、VUI、相机 | commit `18b9ef8e57e69b73c5e9b301b8481fce16a44b9b` |
| CycloneDDS Python | SDK2 DDS 通信 | `0.10.2` |
| `websocket-client` | Go2 bridge WebSocket 客户端 | `1.8.0` |
| `websockets` | policy 端 WebSocket 服务 | `>=14,<17` |
| `unitree_webrtc_connect` | Go2 AudioHub/WebRTC | upstream `2.2.0`，commit `e0abc5780761539eff89a382a79319e5cf6ad1f4` |
| `aiortc` / PyAV | WebRTC 和音频转换 | `1.9.0` / `12.3.0` |
| sherpa-onnx | 本地 Matcha TTS | `1.13.2` |
| Matcha Baker + Vocos | 默认中文本地语音 | 见第 6 节哈希 |
| `edge-tts` | Matcha 失败时的联网回退 | `7.2.8` |

### 1.3 可选依赖

- whisper.cpp + `ggml-base.bin`：只用于实验性麦克风转写，不影响控制和文本播报。
- Aishell3、MeloTTS：只用于 TTS benchmark，默认 bridge 不加载。
- RealSense/`pyrealsense2`：用于外接 RGB-D，不影响本控制链路。
- ROS 1 `rospy`、`geometry_msgs`：仅 policy 的 `--source ros` 需要。

## 2. 当前实机目录

```text
/home/unitree/unitree_sdk2_python             # Unitree SDK2，editable install
/home/unitree/uni_control                     # 本项目 Go2 端代码
/home/unitree/uni_control/vendor/python       # 隔离的 WebRTC/TTS Python 包
/home/unitree/uni_control/vendor/local_tts    # 本地 TTS 模型
/home/unitree/uni_control/vendor/whisper.cpp  # 可选 Whisper 源码、CLI 和模型
```

当前 Python 通过用户 site 的 `.pth` 加载：

```text
/home/unitree/.local/lib/python3.8/site-packages/easy-install.pth
  → /home/unitree/unitree_sdk2_python
```

`go2_voice_intercom.py` 会主动把 `/home/unitree/uni_control/vendor/python` 插到
`sys.path` 首位，因此不会把 aiortc、PyAV 和 TTS 包安装进 Unitree 的系统 Python 环境。

## 3. policy 机部署

在 molmospaces 仓库根目录执行：

```bash
python3 -m pip install -r \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/requirements_policy.txt
```

离散键盘和 stdin 不需要 ROS。连续 ROS 输入需要先安装并 source ROS 1 Noetic，再确认：

```bash
python3 -c 'import rospy; from geometry_msgs.msg import Twist; print("ROS OK")'
```

## 4. Unitree SDK2 与 CycloneDDS

官方来源：

- Unitree SDK2 Python：<https://github.com/unitreerobotics/unitree_sdk2_python>
- CycloneDDS：<https://github.com/eclipse-cyclonedds/cyclonedds>

Unitree 官方 README 要求 Python 3.8+、CycloneDDS `0.10.2`、NumPy 和 OpenCV。当前 Go2
使用 Python 3.8.10。

### 4.1 安装 Python 依赖

```bash
python3 -m pip install --user \
  'cyclonedds==0.10.2' \
  'websocket-client==1.8.0'
```

如果 CycloneDDS wheel 无法安装或 SDK 报告找不到本地 CycloneDDS，按官方 FAQ 构建
`releases/0.10.x`：

```bash
git clone --branch releases/0.10.x \
  https://github.com/eclipse-cyclonedds/cyclonedds.git \
  /home/unitree/cyclonedds

cmake -S /home/unitree/cyclonedds \
  -B /home/unitree/cyclonedds/build \
  -DCMAKE_INSTALL_PREFIX=/home/unitree/cyclonedds/install

cmake --build /home/unitree/cyclonedds/build \
  --target install \
  --parallel 4
```

### 4.2 安装固定版本 SDK2

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git \
  /home/unitree/unitree_sdk2_python

git -C /home/unitree/unitree_sdk2_python checkout \
  18b9ef8e57e69b73c5e9b301b8481fce16a44b9b

CYCLONEDDS_HOME=/home/unitree/cyclonedds/install \
python3 -m pip install --user -e /home/unitree/unitree_sdk2_python
```

如果第 4.1 节的 wheel 已经满足 SDK 构建，则最后一条命令通常不需要显式
`CYCLONEDDS_HOME`。

当前实机 SDK 仓库的三个 `example/` 文件存在现场修改；`uni_control` 不依赖这些示例改动。
换机部署应使用上面的干净 commit，不要复制当前 SDK 工作树中的示例修改。

### 4.3 SDK 只读验证

```bash
python3 - <<'PY'
import cyclonedds
import unitree_sdk2py

print("cyclonedds:", cyclonedds.__file__)
print("unitree_sdk2py:", unitree_sdk2py.__file__)
PY
```

连接 Go2 DDS 时，本项目默认使用 Jetson 的 `eth0`。如果部署在外部电脑，应把接口名换成
与 Go2 同网段的实际有线网卡，并先按照 Unitree 官方 Quick Start 验证 DDS 通信。

## 5. 部署 `uni_control` 自有代码

先在 Go2 创建目录：

```bash
mkdir -p /home/unitree/uni_control
```

从当前开发机同步运行代码：

```bash
scp \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/README.md \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/SDK_DEPLOYMENT.md \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/control_protocol.py \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/go2_control_bridge.py \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/go2_voice_intercom.py \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/start_go2_control.py \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/inspect_go2_interfaces.py \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/audio_io_test.py \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/benchmark_tts.py \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/uni_control/requirements_go2.txt \
  unitree:/home/unitree/uni_control/
```

不要同步本机 `__pycache__/` 或 `recordings/`。录音不是运行依赖，也不应进入代码提交。

## 6. WebRTC、TTS Python 环境和模型

### 6.1 创建隔离 vendor 目录

```bash
mkdir -p \
  /home/unitree/uni_control/vendor/python \
  /home/unitree/uni_control/vendor/local_tts/models
```

当前核心版本：

```bash
python3 -m pip install \
  --target /home/unitree/uni_control/vendor/python \
  'aiortc==1.9.0' \
  'av==12.3.0' \
  'edge-tts==7.2.8' \
  'sherpa-onnx==1.13.2' \
  aiohttp curl_cffi pycryptodome wasmtime ifaddr packaging requests tabulate
```

Python 3.8 已停止上游支持，未来新版本 `cryptography` 或其他传递依赖可能不再兼容。换机时
不要无条件升级这些固定版本；安装后必须执行第 8 节验证。

### 6.2 安装固定版本 `unitree_webrtc_connect`

来源：<https://github.com/legion1581/unitree_webrtc_connect>

当前 Go2 的源码副本与 upstream `2.2.0`、commit
`e0abc5780761539eff89a382a79319e5cf6ad1f4` 完全一致。为保持与当前环境一致，只复制固定
commit 的 Python package，依赖由上一节安装：

```bash
git clone https://github.com/legion1581/unitree_webrtc_connect.git \
  /home/unitree/unitree_webrtc_connect

git -C /home/unitree/unitree_webrtc_connect checkout \
  e0abc5780761539eff89a382a79319e5cf6ad1f4

cp -a \
  /home/unitree/unitree_webrtc_connect/unitree_webrtc_connect \
  /home/unitree/uni_control/vendor/python/
```

当前源码集合校验值的生成方式与结果：

```bash
cd /home/unitree/uni_control/vendor/python
find unitree_webrtc_connect -type f ! -path '*/__pycache__/*' -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | sha256sum
```

```text
c0416fbd08397411067b6c375fd2657db6397f1e9c6f690b5910aa8c7ab0a899
```

上游完整安装还依赖 PortAudio、sounddevice、PyAudio、OpenCV 等。当前 bridge 只使用 WebRTC
数据通道和 Go2 音频 track，并在导入前注入 sounddevice 空模块，因此现场最小环境没有安装
PortAudio。若要直接运行上游的麦克风/桌面音频示例，应按上游 README 安装完整依赖。

### 6.3 下载默认 Matcha + Vocos

官方模型说明：
<https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese/matcha-icefall-zh-baker.html>

下载：

```bash
mkdir -p /home/unitree/tts_download

curl -fL \
  -o /home/unitree/tts_download/matcha-icefall-zh-baker.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2

curl -fL \
  -o /home/unitree/uni_control/vendor/local_tts/models/vocos-22khz-univ.onnx \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx
```

校验下载文件：

```bash
echo '20de2ec034b55562609d6362771c934905dfe11d0f41ec103d593427ad9a7efb  /home/unitree/tts_download/matcha-icefall-zh-baker.tar.bz2' \
  | sha256sum -c -

echo '0574a135aa1db2de6e181050db2ec528496cacd4a4701fc5d7faf9f9804c0081  /home/unitree/uni_control/vendor/local_tts/models/vocos-22khz-univ.onnx' \
  | sha256sum -c -
```

解压到 bridge 的默认路径：

```bash
tar -xjf /home/unitree/tts_download/matcha-icefall-zh-baker.tar.bz2 \
  -C /home/unitree/uni_control/vendor/local_tts/models
```

当前实机核心模型文件哈希：

```text
ef7ebdf5987e16a5836136a51d6f3560ca997ffd33d06a40daab5af92b4b86e5  model-steps-3.onnx
38b886d46aefa50da6322a64d72fd595d5f4fae1051adb160d647541b1e0a4a2  lexicon.txt
56209b2bf609d5ac1d66ede6dae7bf5254bd3f8aa24c4a6823713d5b884d87ba  tokens.txt
0574a135aa1db2de6e181050db2ec528496cacd4a4701fc5d7faf9f9804c0081  vocos-22khz-univ.onnx
```

Aishell3 和 MeloTTS 只用于 `benchmark_tts.py`；正常 launcher 不需要下载。

## 7. 可选 whisper.cpp

来源：<https://github.com/ggml-org/whisper.cpp>

当前实机：

```text
commit: 978113305b2ead22249b881deafa131dc8884911
whisper-cli: 1.9.3-dev
model: ggml-base.bin
```

部署：

```bash
git clone https://github.com/ggml-org/whisper.cpp.git \
  /home/unitree/uni_control/vendor/whisper.cpp

git -C /home/unitree/uni_control/vendor/whisper.cpp checkout \
  978113305b2ead22249b881deafa131dc8884911

cmake \
  -S /home/unitree/uni_control/vendor/whisper.cpp \
  -B /home/unitree/uni_control/vendor/whisper.cpp/build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build /home/unitree/uni_control/vendor/whisper.cpp/build \
  --parallel 4

/home/unitree/uni_control/vendor/whisper.cpp/models/download-ggml-model.sh base
```

官方 `ggml-base.bin` 校验：

```text
SHA1:   465707469ff3a37a2b9b8d8f89f2f99de7299dac
SHA256: 60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe
```

## 8. 安装验证

### 8.1 Python 导入

```bash
cd /home/unitree/uni_control
python3 - <<'PY'
import sys
import types

sys.path.insert(0, "/home/unitree/uni_control/vendor/python")
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))

import av
import edge_tts
import sherpa_onnx
import unitree_sdk2py
import unitree_webrtc_connect
import websocket

print("av", av.__version__)
print("edge_tts", edge_tts.__version__)
print("sherpa_onnx", sherpa_onnx.__version__)
print("unitree_sdk2py", unitree_sdk2py.__file__)
print("unitree_webrtc_connect", unitree_webrtc_connect.__file__)
print("websocket-client", websocket.__version__)
PY
```

### 8.2 无运动、无播放地加载 Matcha

下面使用模拟运动驱动和无服务端口；它只加载模型，不连接运动接口，不播放声音：

```bash
cd /home/unitree/uni_control
python3 go2_control_bridge.py \
  --state-source simulated \
  --speaker-backend go2 \
  --url ws://127.0.0.1:9 \
  --once
```

预期先看到 `Matcha TTS loaded ...`，然后因测试端口不存在打印 `Connection refused` 并正常
退出。

### 8.3 查看 launcher 最终命令

```bash
python3 /home/unitree/uni_control/start_go2_control.py --print-command
```

确认输出包含：

```text
--state-source go2
--speech-primary-backend matcha
--speech-fallback-backend edge
--matcha-model-dir /home/unitree/uni_control/vendor/local_tts/models/matcha-icefall-zh-baker
--matcha-vocoder /home/unitree/uni_control/vendor/local_tts/models/vocos-22khz-univ.onnx
```

### 8.4 最后再做实机只读检查

```bash
python3 /home/unitree/uni_control/inspect_go2_interfaces.py --interface eth0
```

完成全部只读验证后，才按照 `README.md` 的安全流程启用 `--enable-motion`。SDK、WebRTC 或模型
版本发生变化时，应重新验证 STOP、断线归零、姿态恢复、AudioHub 临时文件删除和 TTS 回退。
