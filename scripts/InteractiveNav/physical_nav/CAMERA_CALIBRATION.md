# Go2 与 D435i 相机参数记录

本文记录 2026-09-02 在当前 Go2 实物平台上读取或核对到的相机参数，供后续标定、建图误差分析和坐标系检查使用。

需要严格区分三类来源：

1. **实机 SDK 读取值**：由当前设备直接返回。
2. **官方模型名义值**：来自 Unitree 官方 URDF，不代表本机精确标定。
3. **人工测量值**：来自当前 D435i 安装位置的近似实测。

## 1. Go2 自带前置相机

### 1.1 已验证图像接口

通过 `ssh unitree` 调用 `VideoClient.GetImageSample()` 实测：

- 返回码：`0`
- 编码：JPEG，文件头 `FFD8`
- 分辨率：`1920 × 1080`
- 单帧大小：随画面变化，测试帧约为 `129–167 KB`

测试入口：

- `scripts/InteractiveNav/uni_control/inspect_go2_interfaces.py`

### 1.2 内参状态

当前尚未获得 Go2 自带前置相机的准确内参。

`VideoClient.GetImageSample()` 只返回压缩图像，不返回相机矩阵、畸变系数或标定信息。当前在可访问的狗端文件中也没有找到对应的相机标定 YAML/JSON。

因此以下参数目前未知：

```text
fx, fy, cx, cy
distortion model
k1, k2, p1, p2, k3, ...
```

仅凭 `1920 × 1080` 分辨率无法准确推导内参。若后续需要将 Go2 前置相机用于几何投影、PnP、深度融合或建图，应使用棋盘格或 ChArUco 对本机重新标定。

官方 SDK 接口参考：

- <https://github.com/unitreerobotics/unitree_sdk2/blob/main/include/unitree/robot/go2/video/video_client.hpp>

### 1.3 官方 URDF 名义外参

Unitree 官方 Go2 URDF 给出的固定关节为：

```xml
<joint name="front_camera_joint" type="fixed">
  <origin rpy="0 0 0" xyz="0.32715 -0.00003 0.04297" />
  <parent link="base" />
  <child link="front_camera" />
</joint>
```

即：

```text
base → front_camera
translation = [0.32715, -0.00003, 0.04297] m
RPY         = [0, 0, 0] rad
```

该值是官方机器人模型的名义安装位置，不是当前实机的单机标定结果；URDF 中也没有给出 `front_camera_optical_frame` 的精确变换。

官方 URDF：

- <https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf>

## 2. D435i 设备信息

当前实机读取结果：

```text
设备：Intel RealSense D435I
序列号：243722070237
固件：5.15.1.55
运行配置：640 × 480 @ 15 FPS
深度格式：Z16
彩色格式：BGR8
```

当前传感桥接代码使用：

```python
rs.align(rs.stream.color)
```

即深度图在发送和后续处理前已经对齐到彩色图。因此，当前 RGB-D 点云提升和语义投影使用的是**彩色相机内参**。

实现位置：

- `scripts/InteractiveNav/physical_nav/go2_readonly_sensor_bridge.py`

## 3. D435i 彩色相机内参

实机读取值：

```text
width  = 640
height = 480
fx = 605.3994750976562
fy = 604.801513671875
cx = 324.91314697265625
cy = 255.13192749023438
```

相机矩阵：

```text
K_color =
[605.3994751,   0,            324.9131470]
[  0,          604.8015137,   255.1319275]
[  0,            0,              1       ]
```

畸变参数：

```text
model = distortion.inverse_brown_conrady
D = [0, 0, 0, 0, 0]
```

## 4. D435i 原始深度相机内参

实机读取值：

```text
width  = 640
height = 480
fx = 391.511962890625
fy = 391.511962890625
cx = 315.64642333984375
cy = 240.26795959472656
```

相机矩阵：

```text
K_depth =
[391.5119629,   0,           315.6464233]
[  0,          391.5119629,  240.2679596]
[  0,            0,            1        ]
```

畸变参数：

```text
model = distortion.brown_conrady
D = [0, 0, 0, 0, 0]
```

## 5. D435i 深度到彩色外参

由当前 D435i 设备直接读取：

```text
R_depth→color =
[ 0.9999698400, -0.0022752765, -0.0074289530]
[ 0.0023080781,  0.9999876022,  0.0044097938]
[ 0.0074188276, -0.0044268076,  0.9999626875]

t_depth→color =
[0.0146291340, 0.0000499687, 0.0001549271] m
```

两个光学中心的主要横向间距约为 `14.63 mm`。

逆变换为：

```text
R_color→depth =
[ 0.9999698400,  0.0023080781,  0.0074188276]
[-0.0022752765,  0.9999876022, -0.0044268076]
[-0.0074289530,  0.0044097938,  0.9999626875]

t_color→depth =
[-0.0146274269, -0.0000844165, -0.0002632311] m
```

由于当前代码已经将深度对齐到彩色图，后续处理通常不需要再次手工应用这组 `depth → color` 外参。

## 6. D435i 到 Go2 base 的当前安装外参

当前外接 D435i 通过杆固定在 Go2 上。依据人工测量：

- 相对 Go2 base 向上约 `0.62 m`
- 摄像头距地约 `1.05 m`
- Go2 机身前后长度约 `0.70 m`
- 摄像头位于从后向前约 `0.38 m` 处，因此相对机身中心向前约 `0.03 m`
- 摄像头平视前方，没有额外安装角度偏移

当前配置使用：

```text
base → D435i mount
x = +0.03 m
y =  0.00 m
z = +0.62 m
roll  = 0 rad
pitch = 0 rad
yaw   = 0 rad
```

配置入口：

- `scripts/InteractiveNav/physical_nav/launch/physical_nav_readonly.launch`
- `scripts/InteractiveNav/physical_nav/physical_six_panel_server.py`

注意：这是一组人工测量的近似外参，不是通过标定板或联合优化得到的精确外参。

## 7. D435i optical frame 到 Go2 base 的坐标变换

ROS optical frame 采用：

```text
x：向右
y：向下
z：向前
```

Go2 base frame 采用：

```text
x：向前
y：向左
z：向上
```

在物理安装 RPY 为零时，当前代码使用：

```text
q_base←optical = [x=0.5, y=-0.5, z=0.5, w=-0.5]
```

对应旋转矩阵：

```text
R_base←optical =
[ 0,  0,  1]
[-1,  0,  0]
[ 0, -1,  0]
```

完整齐次变换：

```text
T_base←color_optical =
[ 0,  0,  1, 0.03]
[-1,  0,  0, 0.00]
[ 0, -1,  0, 0.62]
[ 0,  0,  0, 1.00]
```

实现位置：

- `scripts/InteractiveNav/physical_nav/physical_ros_gateway.py`

上述旋转主要是 optical frame 与 base frame 的坐标轴约定转换，不代表相机存在额外物理安装倾角。

## 8. 运行时动态相机位姿

当前建图使用的完整相机变换为：

```text
T_map←camera = T_map←base × T_base←camera
```

其中：

- `T_map←base` 来自 Go2 实时位置和 IMU 四元数。
- `T_base←camera` 是上述固定安装外参。

因此 Go2 俯身、抬头或侧倾时，D435i 的位置和朝向会随刚性连接杆一起发生变化，并通过 Go2 实时姿态反映到相机位姿中。

## 9. 后续需要进一步确认的事项

1. 使用棋盘格或 ChArUco 标定 Go2 自带前置相机内参及畸变。
2. 验证官方 URDF 的 `base → front_camera` 名义外参是否适用于当前实机版本。
3. 对 D435i 执行 `base ↔ camera` 联合外参标定，替代人工尺量值。
4. 检查 D435i 安装杆在运动中的弹性形变和振动误差。
5. 使用已知平面、直线和物体尺寸验证 RGB-D 点云、3D box 与 OCC 地图的一致性。
6. 分别记录站立、俯身、侧倾状态下的重投影误差，确认 Go2 IMU 姿态补偿方向和时间同步正确。
