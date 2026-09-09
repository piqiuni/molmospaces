# OccupancyGrid 房间分割加速调研（2026-08-02）

## 结论摘要

当前实现不应首先整体替换为 ROSE²、Voronoi Random Field 等更复杂的房间分割方法。要达到仿真与地图链路 5 Hz，优先级应为：

1. 将房间分割从 `semantic_mapping_node` 的共享锁和 OCC callback 中移出，使用单独 worker，只处理最新地图并丢弃过期中间版本。
2. 保留现有“距离变换产生 room core + portal cut + 多源扩张”的语义，但消除 Python 全图 `deque`、逐 component 的 `np.where(labels == id)`、重复 list/ndarray 转换和多次全图复制。
3. 全图基线使用 OpenCV 的并行 Connected Components（优先 BBDT/GRANA）和一次 Distance Transform；必须在目标 CPU、实际地图尺寸上验证是否能在 200 ms 内完成。
4. 如果 OCC 每次只改变小区域，中期接入 `OccupancyGridUpdate` dirty rectangle，并采用 DynamicVoronoi 增量维护距离图/Voronoi 图；房间拓扑只在门户变化或累计 dirty area 超阈值时更新。
5. 房间分割不是每个 simulator step 的硬实时依赖。规划 OCC/代价地图应以 5 Hz 就绪；room segmentation 可独立以 1–2 Hz 或事件触发运行，并携带 source revision/stamp，不能阻塞最新 OCC 发布。

在当前代码中，2.85 秒的 `occupancy_room_refresh` 是整条 room refresh，而非单一 OpenCV 调用。它包含语义 overlay、全图分割、多源波前填充、稳定 ID remap、裁剪网格和 room graph 更新。开源算法只能减少其中一部分；共享锁和 Python 数据路径必须同时处理。

## 当前实现的主要成本

当前 `RoomSegmenter.segment()` 的主要阶段为：

- 将 ROS `OccupancyGrid.data` 转换为 NumPy，并创建 free/occupied/segmentation mask。
- 对 occupied mask 执行 Connected Components，识别封闭障碍物；若启用 small-obstacle 清理，会再做一次 occupied CCL。
- 根据 portal hints 在全图 mask 上绘制切口。
- 对全部自由空间执行 L2 Distance Transform。
- 对 room core 执行 Connected Components。
- 对每个 component 调用 `np.where(labels == component_id)`，这会按 component 数量反复扫描完整 label 图。
- 将 component cell 转为 Python `list[int]`，再逐 cell 写入 Python `room_ids` 和 `room_conf`。
- 使用 Python `deque` 在完整自由空间上执行多源四邻域 wavefront；每访问一个 cell 都进行 Python 除法、取模、边界判断和多个 list/ndarray 访问。
- 将结果多次复制到 previous/candidate/stable state。
- `_build_cropped_room_segment_grid()` 再构造 `valid_indices`、`rows`、`cols` 和 `cropped` Python list。
- `graph_store.update_room_grid()` 随后重建/更新房间节点和关系。

因此，即使 OpenCV 的 Distance Transform 和 CCL 已在 C++ 中，外围的 Python 全图遍历、全图复制和 component-by-component 全图扫描仍可能占据大部分 wall time。下一次 profiling 应把 `room_refresh` 细分成：overlay、mask build、occupied CCL、distance transform、core CCL、component extraction、ID remap、wavefront、stabilize/copy、crop、graph update。

## 方案比较

| 方案 | 一手来源 | 许可 | 实时 5 Hz 判断 | 集成复杂度 | 对当前项目的建议 |
|---|---|---|---|---|---|
| OpenCV 并行 CCL（SAUF/Wu、BBDT/GRANA、Spaghetti） | [OpenCV Connected Components 官方文档](https://docs.opencv.org/4.3.0/d3/dc0/group__imgproc__shape.html)、[OpenCV 官方实现](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/connectedcomponents.cpp) | OpenCV 4.5+ 为 [Apache-2.0](https://opencv.org/license/) | 对中等二维地图通常是最可控的全图基线；官方实现可在并行后端可用且图像行数足够时启用并行。是否小于 200 ms 必须实测 | 低 | 立即采用。Python 可用 `connectedComponentsWithStatsWithAlgorithm` 时显式选择 `CCL_GRANA/CCL_BBDT`；同时检查 OpenCV 构建是否有 pthreads/TBB/OpenMP，并避免多次 CCL 和逐 label 全图扫描 |
| OpenCV Distance Transform | [OpenCV Distance Transform 官方文档](https://docs.opencv.org/4.5.0/d7/d1b/group__imgproc__misc.html) | Apache-2.0（OpenCV 4.5+） | 单次全图 EDT 可作为 5 Hz 候选；多阈值重复分割或多次 EDT 不适合每帧执行 | 低 | 保留“一次 EDT 产生 room core”。先基准比较 `DIST_L2, mask=3` 与当前 `mask=5` 的耗时和分割差异，不能在未验证语义质量时直接切换 |
| OpenCV/原生 C++ 多源 wavefront 或 watershed | [scikit-image Watershed 官方文档](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed)说明 marker-based flooding 使用优先队列并由 C 实现；OpenCV 也提供 [Watershed 官方教程](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html) | scikit-image BSD；OpenCV Apache-2.0 | 用 C/C++ 实现替换 Python `deque` 后有较高机会满足 5 Hz；但 watershed 的边界语义与当前“从 core 向所有 free cell 扩张”需要回归验证 | 中 | 短期可先写 NumPy/C++ 多源传播，保持现有输出语义；watershed 作为实验分支，不直接替换主线 |
| Fraunhofer IPA morphological segmentation | [官方仓库与说明](https://github.com/ipa320/ipa_coverage_planning/tree/noetic_dev/ipa_room_segmentation)、[ICRA 2016 官方论文入口](https://publica.fraunhofer.de/entities/publication/0bf23149-75d5-4601-bfce-992d91698862) | package metadata 写明“LGPL for academic and non-commercial use；商业用途联系 Fraunhofer”，并非无条件宽松许可 | 官方 README 明确称 morphology/distance 是其最快方法，但它们包含迭代 erosion/threshold 和 wavefront；大地图仍可能慢，不能假设 5 Hz | 中 | 可借鉴算法和 benchmark，不建议直接复制代码，需先解决许可问题。迭代 erosion 会重复全图操作，不如当前单次 EDT core 路线适合在线 5 Hz |
| Fraunhofer IPA distance segmentation | 同上；[package.xml 原始元数据](https://raw.githubusercontent.com/ipa320/ipa_coverage_planning/noetic_dev/ipa_room_segmentation/package.xml) | 同上 | 算法对 distance map 迭代多个 threshold，并在每轮寻找 component；更适合离线/低频，而非每个 OCC revision | 中 | 不推荐替换当前在线算法；可用于离线对照和质量评估 |
| Fraunhofer IPA Voronoi / Voronoi Random Field | 同上 | 同上 | 官方 README 指出大地图算法可能运行数分钟，特别是 semantic 和 VRF；不适合直接进入 5 Hz 链路 | 高 | 不用于硬实时主链。VRF 还需要训练与 CRF 推断；可做后续研究 baseline |
| DynamicVoronoi 增量 EDT/GVD | [ROS 官方文档](https://docs.ros.org/fuerte/api/dynamicvoronoi/html/index.html)、[ROS Index](https://index.ros.org/p/dynamicvoronoi/)、[官方仓库](https://github.com/frontw/dynamicvoronoi)；算法论文为 Lau、Sprunk、Burgard，IROS 2010 | BSD（ROS Index）；仓库状态为 unmaintained | 最符合“地图局部变化、持续更新”的 5 Hz 场景。API 支持 `occupyCell`、`clearCell` 后增量 `update()`，避免每次全图 EDT | 中高 | 中期首选。用增量距离场替代全图 EDT，并从 GVD/portal hints 更新拓扑；需封装 C++ ROS node 或 Python binding，且自行维护新旧 OccupancyGrid 的 cell diff |
| Free-space + occupied-space Voronoi doorway detection | [ICINCO 2022 原论文](https://www.scitepress.org/publishedPapers/2022/111416/pdf/index.html) | 论文版权为 SCITEPRESS；未确认有可复用开源实现 | 同时构建两套 Voronoi 并分析窄通道/突起，语义上适合门检测，但全图执行不适合作为未经优化的 5 Hz 路径 | 高 | 适合作为 portal detector 的研究候选，不作为近期性能修复 |
| ROSE² 结构化房间重建 | [原论文](https://arxiv.org/abs/2203.03519)、[作者公开实现](https://github.com/goldleaf3i/declutter-reconstruct) | 需以仓库具体 LICENSE 为准，本文不确认 | 方法包括结构清理、Hough line、DBSCAN、墙线/face 构建与拓扑修正，目标是鲁棒质量而非每帧低延迟 | 高 | 可作为低频或离线高质量 room graph 校正器，不进入每个 OCC update 的就绪屏障 |

## 增量与局部更新

ROS `map_msgs/OccupancyGridUpdate` 原生表达一个矩形更新区域 `x/y/width/height/data`，见 [ROS 1 消息定义](https://docs.ros.org/en/melodic/api/map_msgs/html/msg/OccupancyGridUpdate.html) 和 [ROS 2 消息定义](https://docs.ros.org/en/rolling/p/map_msgs/msg/OccupancyGridUpdate.html)。这可直接作为 dirty ROI 的来源。

局部 room segmentation 不是简单裁剪后重新分割。ROI 必须：

- 向外扩张至少 `room_core_clearance_cells`、portal cut 宽度和障碍物清理 ring 的最大影响半径。
- 在 ROI 边界读取旧 room labels 作为 seeds，避免每次产生新 room ID。
- 对跨越 ROI 边界的 component 做 union/reconciliation。
- 当 doorway/portal 的 open-close 状态变化时，强制扩大到该 portal 两侧相邻 room，必要时触发全图重分割。
- 当 map origin、resolution、width/height 变化时，放弃局部更新并执行一次全图初始化。

OpenCV 没有提供动态 Connected Components 状态对象，因此“dirty ROI + CCL”需要项目自行实现边界 label 合并。DynamicVoronoi 已经提供占据 cell 加入/移除后的增量距离/Voronoi 更新，因而比自行实现动态 EDT 更合适；但最终 room component 和 stable ID 仍需维护。

## 推荐实施路线

### 阶段 A：先让链路不再被 room segmentation 阻塞

目标不是先把 2.85 秒优化到 200 ms，而是保证它即使耗时 2.85 秒也不阻塞 OCC、planning grid、publisher 和 ready barrier。

- OCC callback 在短锁内只更新 latest raw/planning occupancy、revision/stamp 和 dirty information。
- room worker 在锁内取得不可变 snapshot/reference，立即释放锁，在锁外执行 overlay + segmentation。
- worker 采用 latest-only/coalescing：若运行期间收到 revision 101、102、103，完成旧任务后只处理 103，不排队重算 101/102。
- 计算完成后仅在短锁内提交结果；若 source geometry 已变化则丢弃旧结果。
- room ready 与 mapping ready 分开。正常 simulator step 的 5 Hz 屏障不等待 room segmentation；只有确实依赖 room topology 的决策阶段才请求一个不旧于指定 revision 的 room-ready。
- graph update 也应在 room worker 内构造差异或 snapshot；共享 graph commit 必须短锁化，不能再次把数秒计算放回主锁。

这一阶段应立即消除之前观测到的约 5 秒累计锁等待。

### 阶段 B：优化现有算法但保持输出语义

- 给每个内部阶段加独立耗时与 cell/component 计数。
- 全程使用 ndarray 保存 `room_ids`/`room_conf`，直到 ROS message 序列化时才 `.ravel().tolist()`。
- 使用 NumPy 有效区域 bounding box 替换 `_build_cropped_room_segment_grid()` 的多个 Python list。
- 避免 `for component: np.where(labels == component)`；对 label 图一次排序/分桶，或直接用 vectorized label-to-stable-ID lookup table 生成完整 room ID ndarray。
- 用编译实现替换 Python `deque` wavefront。优先做一个保持四邻域、confidence 衰减规则一致的 C++/Cython/Numba-independent 实现，避免新增重量级运行依赖。
- 显式测试 OpenCV `CCL_GRANA/CCL_BBDT`，记录 OpenCV parallel backend 与 thread count。
- occupied CCL 结果在封闭障碍物清理与 small-obstacle 清理间复用；若某功能关闭，完全跳过对应扫描。
- 对稳定性比较使用 ndarray，避免每帧多份 Python list 深复制。

验收目标：实际 house map 上 room worker 的 P95 小于 200 ms；若暂时不能达到，至少保证 OCC/planning 链路维持 5 Hz，room 结果按较低频率异步更新。

### 阶段 C：增量距离场与事件驱动拓扑

- 比较前后 OccupancyGrid 或消费 update topic，提取 changed cells/dirty rectangle。
- 集成 DynamicVoronoi，首次全图初始化；之后只调用 `occupyCell`/`clearCell` 和 `update()`。
- portal hints、door state 或 dirty region 与 GVD 窄通道相交时，更新相邻 room partition。
- 周期性（例如每 5–10 秒）或累计 dirty ratio 超阈值时执行全图校正，防止局部误差累积。
- 可在离线 recorder 保存的 PNG+JSON 上重放相同 update sequence，比较全图版本与增量版本的 label IoU、stable ID 变化和 latency。

## 不推荐的做法

- 不要让 simulator 每个 step 都等待完整 room segmentation；房间结构的更新频率与局部规划 OCC 的更新频率不是同一需求。
- 不要简单将 room refresh 降频但继续在共享锁内运行；低频时仍会周期性冻结整条链路。
- 不要仅增加 ROS subscriber queue。慢 consumer 会积压旧 OCC，使 room 结果越来越落后；应使用 queue size 1/latest-only 语义并记录 skipped revisions。
- 不要直接把 GMapping/OCC 提升到更高发布频率来掩盖问题；在锁和 room worker修复前只会增加积压。
- 不要把 ROSE²/VRF 等高质量离线算法作为 5 Hz ready barrier 的前置条件。

## 建议 benchmark

在同一组原始 OccupancyGrid PNG+JSON 上分别运行：

1. 当前 Python 实现。
2. 锁外异步但算法未优化的实现。
3. ndarray + OpenCV BBDT + 编译 wavefront 的全图实现。
4. DynamicVoronoi 增量实现。

至少记录：

- map width/height、known/free/occupied cell 数。
- dirty cells、dirty rectangle area 和全图占比。
- 每阶段 average/P50/P95/max。
- source revision 到 room-ready 的 wall latency。
- coalesced/skipped revision 数。
- room label IoU、room 数、portal 两侧是否正确分离、stable room ID churn。
- OCC/planning costmap 的独立 5 Hz 达标率，确保 room worker 不再影响主链。

最终选择标准应是：主规划链路稳定 5 Hz、room segmentation 不持有共享锁、room topology 的质量不低于当前实现；之后才比较 room worker 本身能否达到 5 Hz。
