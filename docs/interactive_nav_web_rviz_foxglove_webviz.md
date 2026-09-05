# ROS1 实物导航网页 3D 可视化选型：Foxglove 与 Webviz

更新日期：2026-09-02

## 结论

当前 Go2 / D435i ROS1 Noetic 链路可以直接使用 Foxglove 实现网页端 RViz 类可视化，且无需改变现有 ROS publisher 的消息格式。推荐在运行 ROS master 的本机启动只读、严格限流的 ROS1 `foxglove_bridge`，使用新端口 `8766`，浏览器打开 Foxglove 3D Panel；保留现有 `8765` 六面板网页。

不建议新接入 Cruise Webviz。它仍能通过 rosbridge 显示本项目所需的 ROS1 消息，也能静态自托管，但官方仓库最后一次主分支提交停留在 2022-03，README 明确表示没有商业支持、社区支持不稳定，并建议在不能满足需求时改用 Foxglove。[Webviz README](https://github.com/cruise-automation/webviz#foxglove)；[最后一次主分支提交](https://github.com/cruise-automation/webviz/commit/d13afdacaec0b48f983adcaca55b84c36c42a5f2)

## 能力对比

| 项目需求 | Foxglove | Cruise Webviz |
| --- | --- | --- |
| ROS1 实时连接 | 推荐使用高性能 C++ `foxglove_bridge`；网页通过 Foxglove WebSocket 连接 | 通过 `rosbridge_server` 连接 |
| 原始/分割点云 | 原生支持 `sensor_msgs/PointCloud2`，包括 RViz 兼容的 packed `rgb/rgba` | 3D 面板源码包含 `PointCloud2` 解码、设置与测试 |
| 3D box 与文字 | 原生支持 `visualization_msgs/Marker` / `MarkerArray` | 3D SceneBuilder 支持 `MarkerArray` |
| TF | 3D Panel 维护 TF 历史，并要求消息 frame 到显示 frame 存在变换路径 | 支持 `tf2_msgs/TFMessage`，但整体实现较旧 |
| RGB/深度图 | 支持 `sensor_msgs/Image` / `CompressedImage`；深度图还可直接投影成点云 | Image View 支持 raw/compressed image |
| OCC/costmap | 原生支持 `nav_msgs/OccupancyGrid` 和增量更新 | 有 ROS 3D/地图可视化能力，但不再优先维护 |
| 浏览器使用 | 官方 Web App；ROS1 原生直连仅桌面端，网页应走 WebSocket | Hosted app 或 Docker/static build |
| 自托管完整前端 | 普通使用直接访问官方 Web App；完整自托管 Embedded Viewer 需要定制 Enterprise 协议 | 可用 `cruise/webviz` Docker 或自行构建静态站点 |
| 当前维护状态 | ROS1 bridge 处于 maintenance mode，仅修 bug；官方仍提供 ROS1 指南 | 仓库未标记 archived，但主分支多年无提交且无商业支持 |

Foxglove 对相关类型的完整声明见官方 [3D Panel 文档](https://docs.foxglove.dev/docs/visualization/panels/3d)。Webviz 的支持证据可见官方源码中的 [PointCloud2 解码](https://github.com/cruise-automation/webviz/tree/master/packages/webviz-core/src/panels/ThreeDimensionalViz/commands/PointClouds)、[MarkerArray SceneBuilder](https://github.com/cruise-automation/webviz/blob/master/packages/webviz-core/src/panels/ThreeDimensionalViz/SceneBuilder/index.js) 和 [Image View](https://github.com/cruise-automation/webviz/blob/master/packages/webviz-core/src/panels/ImageView/index.help.md)。

## Foxglove 接入方式

Foxglove 官方对 ROS1 的推荐路径是：现有 ROS graph → ROS1 `foxglove_bridge` → WebSocket → Foxglove 网页/桌面端。Bridge 自动发现 ROS topics，不需要修改机器人端或感知/建图节点。[ROS1 接入指南](https://docs.foxglove.dev/docs/getting-started/frameworks/ros1)

当前项目可直接显示：

- `/physical_nav/points`：原始点云；
- `/physical_nav/segmented_cloud_world`：世界坐标系实例分割点云；
- `/physical_nav/boxes_3d_world`：3D box、类别与置信度文字；
- `/tf`、`/tf_static`：坐标变换；
- `/physical_nav/occupancy`：占据栅格；
- 必要时再开放相机 `Image` / `CompressedImage` 与 `CameraInfo` topic。

推荐 3D Panel 的 Fixed frame 使用 `tf_frame_map`。消息的 `header.frame_id` 必须在相应时间戳存在到该 frame 的 TF 路径，否则 Foxglove 与 RViz 一样不会正确放置点云或 box。Foxglove 也支持点云 decay time、RGB packed color、2D/3D 视角切换和 TF frame 调试。[3D Panel frame/TF 说明](https://docs.foxglove.dev/docs/visualization/panels/3d#transforms)

ROS1 bridge 只支持 Noetic；由于 ROS1 已 EOL，官方说明二进制包较旧，建议从源码构建最新版，且该仓库目前只接受 bug fix。[ROS1 bridge 官方仓库](https://github.com/foxglove/ros-foxglove-bridge)

## 页面与部署边界

最简单的第一阶段不是把 Foxglove 嵌进当前六面板，而是提供第二个链接/标签页：

- 当前六面板继续使用 `http://10.100.5.3:8765/`；
- `foxglove_bridge` 改用 `8766`，避免与现有服务端口冲突；
- 在 Foxglove Web App 中使用 Foxglove WebSocket 数据源连接 `ws://10.100.5.3:8766`；
- 保存一个项目专用 Layout，使点云、分割点云、3D box、TF、OCC 和图像默认可见。

官方文档说明 direct Foxglove WebSocket、Rosbridge 和 native ROS1 连接需要 Developer seat；当前免费方案包含少量 Developer 用户，但团队扩大时需要核对当前价格与 seat 策略。[Live connection](https://docs.foxglove.dev/docs/visualization/connecting/live)；[Seat types](https://docs.foxglove.dev/docs/security/seat-types)；[Pricing](https://foxglove.dev/pricing)

若以后要求把完整 Foxglove Viewer 嵌入本项目网页，需单独评估授权。官方当前的完全自托管 Embedded Viewer 仅通过定制 Enterprise 协议提供，且要求 HTTPS 或 localhost 的 secure context；不能把旧版开源 Foxglove Studio Docker 教程当成当前产品的通用自托管方案。[Self-hosted embedded viewer](https://docs.foxglove.dev/docs/embed/self-hosted)

Webviz 的自托管更自由：官方 README 给出 `docker run -p 8080:8080 cruise/webviz` 和静态构建命令，并通过 rosbridge 访问 ROS1。[Webviz static deployment](https://github.com/cruise-automation/webviz#running-the-static-webviz-application) 但它的长期维护、现代浏览器兼容、点云性能和故障修复风险都需要项目自行承担，因此只适合作为完全离线、必须无商业依赖时的备选。

## 网络与安全

不能将默认配置的 bridge 端口直接暴露到公网。ROS1 Foxglove bridge 默认监听 `0.0.0.0`、不启用 TLS，topic/service/parameter/client-publish 白名单默认全开放；默认 capabilities 还允许客户端发布消息、修改参数、调用服务和获取 assets。官方配置文档也特别提示 asset allowlist 配置不当可能暴露机密文件。[ROS1 bridge configuration](https://github.com/foxglove/ros-foxglove-bridge#configuration)

对当前只读实物测试，建议：

- bridge 只绑定实验网卡地址，或绑定 `127.0.0.1` 后通过 SSH/VPN 隧道访问；
- `topic_whitelist` 仅包含 TF、点云、box、OCC 和必要图像；
- capabilities 仅保留只读可视化所需能力，去掉 `clientPublish`、`parameters`、`services` 和 `assets`；
- service、parameter、client-publish whitelist 设置为不匹配任何名称；
- 不开放公网端口；跨公网优先使用 VPN/WSS，或使用有账户授权和传输加密的官方 Remote Access；
- raw 点云体积较大，先限制订阅 topic，并根据浏览器/GPU负载降低点数或频率，防止 WebSocket 积压。

上述安全建议依据官方默认配置做出的风险推论。Foxglove direct WebSocket 要求客户端能通过局域网、VPN 或端口转发直接访问 bridge；官方 Remote Access 则由设备主动连平台，适合防火墙后的网络。[Live connection network model](https://docs.foxglove.dev/docs/visualization/connecting/live)

## 推荐决策

采用 Foxglove 作为独立的网页 RViz 调试界面，第一阶段不嵌入现有六面板，也不替换当前网页。先在本机以 `8766` 启动严格只读的 ROS1 bridge，验证五类数据：原始点云、分割点云、3D box、TF、OCC。Webviz 只保留为“必须完全静态自托管且接受自行维护旧代码”的后备选项。

## 当前实机运行方式（Docker）

当前实机 Foxglove bridge 使用 Docker 运行，不使用 `foxglove-wizard`，也不要求在宿主机安装 `foxglove_bridge` ROS 包。容器使用 host 网络，因此容器内的 `localhost:11311` 就是宿主机 ROS master。

运行前确认 ROS master 已启动：

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
ss -ltn | grep ':11311'
```

当前容器参数如下：

```text
container: foxglove-bridge-ros1
image: local/foxglove-bridge-ros1:with-ros-msgs
network: host
ROS_MASTER_URI: http://localhost:11311
WebSocket: 8766
send_buffer_limit: 50000000 (50 MB)
```

启动或重启：

```bash
docker start foxglove-bridge-ros1
# 已在运行时使用：
docker restart foxglove-bridge-ros1
```

如果需要从镜像重新创建容器，使用只读 topic 白名单和 50 MB 发送缓存：

```bash
docker rm -f foxglove-bridge-ros1 2>/dev/null || true
docker run -d --name foxglove-bridge-ros1 \
  --network host --restart unless-stopped \
  -e ROS_MASTER_URI=http://localhost:11311 \
  -e ROS_HOSTNAME=localhost -e ROS_DISTRO=noetic \
  -e ROS_WS=/ros1_ws -w /ros1_ws \
  local/foxglove-bridge-ros1:with-ros-msgs \
  roslaunch --screen foxglove_bridge foxglove_bridge.launch \
  port:=8766 address:=0.0.0.0 \
  "topic_whitelist:=['^/tf$','^/tf_static$','^/physical_nav/.*$']" \
  "param_whitelist:=['^$']" \
  "service_whitelist:=['^$']" \
  "client_topic_whitelist:=['^$']" \
  "capabilities:=[connectionGraph]" \
  send_buffer_limit:=50000000
```

检查容器、端口和缓存参数：

```bash
docker ps --filter name=foxglove-bridge-ros1
docker logs --tail 100 foxglove-bridge-ros1
docker exec foxglove-bridge-ros1 \
  bash -lc 'source /opt/ros/noetic/setup.bash; rosparam get /foxglove_bridge/send_buffer_limit'
ss -ltn | grep ':8766'
```

Foxglove Web App 的连接地址为：

```text
ws://10.100.5.3:8766
```

3D Panel 的 Fixed frame 设置为 `tf_frame_map`。当前相机外参 TF 应满足：

```text
tf_frame_map → tf_frame_base_link → d435i_color_optical_frame
```

可在宿主机检查相机 TF：

```bash
source /opt/ros/noetic/setup.bash
rosrun tf tf_echo tf_frame_base_link d435i_color_optical_frame
```

容器内已补齐 `ros-noetic-map-msgs` 和 `ros-noetic-tf2-msgs`，用于解析 `OccupancyGridUpdate` 与 `TFMessage`。如果再次出现 `Send buffer limit reached`，说明 Foxglove 客户端处理速度低于点云/图像发布速度；此时先检查是否打开了多个大数据 topic，以及是否存在多个 Foxglove 客户端连接。
