# Habitat ObjectNav-v2 Stretch 相机协议核查（2026-08-19）

## 结论

Habitat 官方并没有写过一句“ObjectNav-v2 改成竖屏是因为某个单独原因”。官方明确写出的设计动机是：2023 年导航挑战为更容易进行 sim-to-real transfer，将 agent 配置切换为 Hello Robot Stretch，并同步引入连续速度动作空间。`480×640, HFOV=42°, camera height=1.31 m` 是该官方 Stretch benchmark 配置的一部分。

因此，严谨表述应是：

- 竖屏不是 HM3D-Semantics v0.2 数据集本身的属性，也不是所有使用 HM3D-v2 episodes 的论文都必须采用的输入形状。
- 它属于 2023 Habitat ObjectNav-v2 的 **Stretch embodiment / sensor protocol**。
- `480×640 + HFOV 42°` 与将常见的 `640×480` D435i 彩色相机几何旋转到纵向后（原垂直视场成为新水平视场）的结果高度吻合；但“官方因为物理旋转 D435i 而这样配置”目前只能标为强推断，不能当作 Habitat 官方原话。

## 官方材料链条

### 1. 2022 ObjectNav：旧横屏协议

2022 官方 Challenge 说明：agent 使用 RGB-D 与 GPS+Compass，并明确表示仿真中的相机视场和分辨率尝试匹配 Azure Kinect。来源：[Habitat Challenge 2022 官方说明](https://aihabitat.org/challenge/2022/)；同一官方仓库的 [2022 README](https://github.com/facebookresearch/habitat-challenge/blob/challenge-2022/README.md#task-objectnav)。

对应 Habitat-Lab 配置是：

| 项目 | V1 / 2022 值 |
|---|---:|
| RGB/Depth | `640×480` |
| HFOV | `79°` |
| 相机高度 | `0.88 m` |
| agent 高度/半径 | `0.88 m / 0.18 m` |
| 动作空间 | 离散 `v1`，前进/转向/俯仰 |
| episode 限制 | `500 steps` |

来源：[Habitat-Lab 官方 `objectnav_hm3d.yaml`](https://github.com/facebookresearch/habitat-lab/blob/challenge-2023/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml)，本机官方源码：[`objectnav_hm3d.yaml`](/home/ldl/habitat-objectnav/src/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml)。

### 2. 2023 ObjectNav-v2：切换到 Stretch 是官方明确的 sim-to-real 设计

2023 官方 Challenge 在 “New in 2023” 中明确说明：为了更容易 sim-to-real transfer，引入多项 agent 配置变化，使用 Hello Robot Stretch 配置并支持连续动作空间；ObjectNav agent 被建模为 Hello Stretch，配 RGB-D 和 GPS+Compass。官方 Action Space 说明还再次指出，从离散动作改为连续动作，是为了更容易把策略从仿真迁移到 Stretch。

来源：

- [Habitat Navigation Challenge 2023 官方页面](https://aihabitat.org/challenge/2023/#new-in-2023)
- [Habitat Challenge 官方 2023 README](https://github.com/facebookresearch/habitat-challenge/blob/main/README.md#new-in-2023)
- [官方连续动作说明](https://github.com/facebookresearch/habitat-challenge/blob/main/README.md#action-space)

### 3. V2 官方配置确实是 480×640 竖屏和 42° HFOV

| 项目 | V2 Stretch / 2023 值 |
|---|---:|
| RGB/Depth | `480×640` |
| HFOV | `42°` |
| 相机高度 | `1.31 m` |
| agent 高度/半径 | `1.41 m / 0.17 m` |
| 动作空间 | `velocitycontrol` |
| episode 限制 | `500 seconds` |
| episode dataset | `hm3d_v2` |

来源：[Habitat-Lab `challenge-2023` 固定版本官方配置](https://github.com/facebookresearch/habitat-lab/blob/challenge-2023/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch.yaml)，本机官方源码：[`objectnav_v2_hm3d_stretch.yaml`](/home/ldl/habitat-objectnav/src/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch.yaml)。

这说明当前仿真返回 `(height=640, width=480)` 的竖屏 RGB-D 是协议正确行为，并非 recorder 转置了宽高。

## 为什么恰好是 480×640 和 42°：事实与推断边界

Intel 官方 D400 系列规格给出 D435/D435i 彩色相机视场约为 `H=69°, V=42°`。来源：[Intel RealSense D400 Series 官方数据表](https://www.intelrealsense.com/wp-content/uploads/2024/10/Intel-RealSense-D400-Series-Datasheet-October-2024.pdf)（Color Camera FOV 表）。Hello Robot 官方产品资料确认 Stretch 使用头部 RGB-D 相机；其官方软件文档也把 D435i 作为 Stretch 的 3D 相机接口：[Hello Robot Stretch 产品规格](https://hello-robot.com/stretch-3-product)、[Hello Robot 官方 Stretch ROS 接口](https://github.com/hello-robot/stretch_ros2/blob/humble/stretch_core/README.md)。

由此可作如下**强推断**：

1. 原生横向彩色几何是约 `640×480, H≈69°, V≈42°`；
2. 将成像方向旋转为纵向会得到 `480×640`；
3. 原垂直视场 `42°` 变成旋转后图像的水平视场，正好对应 Habitat 配置的 `HFOV=42°`。

但当前找到的 Habitat 官方 Challenge 页面和源码配置没有明确写出“相机物理安装旋转了 90°”或“我们为了纵向安装而交换宽高”。所以应把上述三步写作硬件规格支持的解释，不应伪装成官方直接陈述。

## 对论文复现和当前录像的含义

1. 论文写“HM3D-v2”只说明其 episodes/scene semantic version；不能据此认定它严格使用官方 2023 Stretch 相机协议。
2. 论文展示横屏可能来自旧 V1/Azure-Kinect 风格配置、自定义传感器、训练前 resize/crop，或仅仅是排版；需要继续检查其 config，而不是根据视频画幅判断。
3. 若目标是与官方 2023 ObjectNav-v2 Challenge 对齐，策略输入和 recorder 都应保留完整 `480×640` 图像及 `HFOV=42°` 内参，不应裁剪成横屏或把图像旋转后仍沿用原内参。
4. recorder 可以在横向画布中 letterbox 展示竖屏相机，但必须明确黑边属于布局，不是传感器内容；更适合的统一接口是保留源图纵横比，并让 panel 布局适配不同平台。
5. 数据集版本、task/evaluation protocol、robot embodiment 和 recorder layout 应作为四个独立维度记录。例如：`HM3DSem-v0.2 + ObjectNav-v2 episodes + Stretch sensor/action config + 6-panel recorder`。

## 最终判断

“V2 变竖屏”不是一次孤立的 UI 或录像改动，而是 2023 官方将 benchmark embodiment 从旧协议迁移到 Stretch、追求 sim-to-real 的整体配置变化之一。能够直接由官方材料证明的是 **Stretch/sim-to-real 动机** 和 **480×640/HFOV42 的固定配置事实**；相机旋转与 D435i 视场交换是非常一致的硬件解释，但仍应明确标注为推断。
