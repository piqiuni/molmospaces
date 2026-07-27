# A100：RoboStack ROS1 与 MolmoSpaces 通信

该部署不使用 Docker，也不在系统层安装 ROS。两个环境均落在 A100 的
`/home/ldl`：

```text
/home/ldl/conda_envs/mlspaces    MolmoSpaces + roslibpy（没有 rospy）
/home/ldl/conda_envs/ros-noetic  RoboStack ROS Noetic + rosbridge + navigation
```

ROS 节点保留现有的原生接口：宿主发布 `/odom`、`/tf`、`/registered_scan`，
并订阅 `/cmd_vel_stamped`、`/move_base/status`。中间件模块为
`molmo_spaces.policy.learned_policy.roslibpy_bridge`；它不导入 `rospy`。

## 创建环境

```bash
export HOME=/home/ldl
export CONDA_PKGS_DIRS=/home/ldl/conda_pkgs
export TMPDIR=/home/ldl/tmp/conda

/home/ldl/miniconda3/bin/conda create -y -p /home/ldl/conda_envs/ros-noetic \
  -c conda-forge -c robostack-noetic --override-channels \
  ros-noetic-ros-base ros-noetic-rosbridge-server \
  ros-noetic-gmapping ros-noetic-navigation ros-noetic-tf2-ros \
  ros-noetic-tf2-geometry-msgs ros-noetic-message-filters \
  ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-ros-numpy \
  ros-noetic-catkin compilers cmake ninja pkg-config
```

不要在这些终端中执行 `source /opt/ros/noetic/setup.bash`。RoboStack 的
activation 会设置它自己的 ROS 环境。

### Custom openslam_gmapping overlay

RoboStack 的 `ros-noetic-gmapping` 会安装标准
`ros-noetic-openslam-gmapping 0.2.1`，但当前 `struct_mapping_pkg` 依赖仓库内
`Interactive-Nav-SG-nav/src/openslam_gmapping/` 提供的两个扩展接口：

- `GridSlamProcessor::setUseOdometryPose()`
- `GridSlamProcessor::setScanMatchingCorrectionLimits()`

它们用于 odometry-locked 建图、锁定 scan-matching yaw，以及限制单次平移
修正。编译 catkin workspace 时必须保留并构建仓库内的同名 overlay 包，使其
覆盖 RoboStack underlay 中的标准版本；仅使用标准包会导致
`struct_mapping_pkg` 编译失败。

当前先完整跟踪 vendored 源码以保证可复现。后续应将其收敛为基于 RoboStack
0.2.1 的最小三文件补丁或独立 overlay 包，并增加构建期兼容检测。该技术债已
记录在根目录 `TODO.md`。

MolmoSpaces 环境只需安装可选 rosbridge 依赖：

```bash
pip install -e ".[rosbridge]"
```

## 最小通信验收

在三个 tmux 窗口中、均激活 `ros-noetic` 后运行：

```bash
export HOME=/home/ldl
export ROS_HOME=/home/ldl/.ros
export ROS_LOG_DIR=/home/ldl/ros_logs
source /home/ldl/miniconda3/etc/profile.d/conda.sh
conda activate /home/ldl/conda_envs/ros-noetic

# 窗口 1
roscore

# 窗口 2。19090 避开该机已有的 9090 服务。
rosrun rosbridge_server rosbridge_websocket _port:=19090 __name:=molmospaces_rosbridge

# 窗口 3。仅用于本验收。
python /home/ldl/molmospaces/scripts/InteractiveNav/ros_native_smoke_node.py
```

然后在 `mlspaces` 环境执行：

```bash
export HOME=/home/ldl
export PYTHONPATH=/home/ldl/molmospaces
/home/ldl/conda_envs/mlspaces/bin/python \
  /home/ldl/molmospaces/scripts/InteractiveNav/roslibpy_ros_smoke.py \
  --host 127.0.0.1 --port 19090
```

预期输出为：

```text
ROSBRIDGE_SMOKE_OK cmd_vel=(0.25,0.00,-0.50)
```

该验收同时证明：

- MolmoSpaces Conda 环境无需系统 ROS；
- RoboStack 环境能够运行原生 `rospy` 节点与 `rosbridge_server`；
- Odom、TF、PointCloud2 与 TwistStamped 都以原 ROS message 类型跨环境传递。

## 进入完整导航栈

在 RoboStack 环境编译并 source 交互导航 catkin workspace 后，启动统一系统时
使用 `start_sim:=false`，使 ROS 侧只运行 mapping、move_base、exploration 与
semantic 节点；仿真仍在 `mlspaces` 环境中运行。每个并行实验使用独立 ROS
master、rosbridge 端口和输出目录，避免当前绝对 topic 名互相碰撞。

当前 `RoslibpyBridgeClient` 是独立传输层和 smoke-test 基线，尚未替换
`RosBridgePolicy` 中的 `rospy` 后端。完整仿真若要跨 Conda 环境运行，还需要在
模拟器策略入口中显式接入该 client；在完成这层适配前，应继续使用现有原生
ROS 环境运行 `RosBridgePolicy`。
