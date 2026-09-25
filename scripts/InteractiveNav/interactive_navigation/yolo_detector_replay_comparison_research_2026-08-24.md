# Habitat 历史原始感知帧的 YOLO 检测模型对比方案（2026-08-24）

## 结论

### 单模型部署约束（当前 Habitat 感知默认）

当实验配置选择“单模型”时，必须选择同时提供检测框（box）和实例分割（segmentation mask）的权重：当前默认固定为 `yolo26x-seg.pt`。它由同一个模型、同一次 RGB 推理同时产生 box、类别、置信度和 mask；下游可用 box 做目标关联/导航，用 mask 做可视化与占据区域核验。

`yolo26x.pt`、`yolo11x.pt` 等 detect-only 权重只能作为明确标注的 box-only 对照，不得作为单模型主路径；`yoloe-26x-seg-pf.pt` 也满足 box+seg，但属于 prompt-free 开放词汇模式，只有在实验明确选择 YOLOE-PF 时使用。任何需要 box+seg 的单模型配置都不得把 detector 和独立 segmentation 模型拼接后冒充“单模型”。

这组历史 Habitat RGB 帧应同时跑两类基线，而不是只横向替换当前的 `yoloe-26x-seg-pf.pt`：

1. **闭集 COCO 检测/实例分割**：`yolo26x-seg.pt`、`yolo11x-seg.pt`（若已下载）；它们同时输出 box+mask。纯检测权重 `yolo26x.pt`、`yolo11x.pt` 仅作为 box-only 对照，不得作为单模型 box3d/seg 主路径。当前 episode 的目标是 television；`tv` 属于 COCO 80 类。
2. **文本零样本/开放词汇检测**：`yoloe-26x-seg.pt`（text prompt）与 `yolov8x-worldv2.pt`。两者都固定使用同一目标词表。
3. **视觉样本提示**：仍使用 `yoloe-26x-seg.pt`，但用独立的 reference frame/crop 提供 visual prompt。参考样本不能取自被评分帧本身。
4. **无提示开放词汇基线**：保留当前 `yoloe-26x-seg-pf.pt`，以其内置 4,585 类词表运行，不调用 `set_classes()`。

上述 `x` 规模组成“近似同容量、偏准确率”的主表。若还要决定在线服务部署规格，再补充 `s` 规模的 `yolo26s`、`yolo11s`、`yoloe-26s`、`yolov8s-worldv2` 作为“偏吞吐量”副表。不要把 `s` 与 `x` 的延时差异解释成架构本身的差异。

## 候选模型与官方能力

| 模型 | 检测范式 | 本实验输入 | 输出 | 官方接口 | 适合回答的问题 |
|---|---|---|---|---|---|
| YOLO26 Detect (`yolo26x.pt`) | COCO 闭集 | RGB，不输入提示 | box/class | `YOLO("yolo26x.pt")(image)` | box-only 对照，不用于 box+seg 主路径 |
| YOLO26 Seg (`yolo26x-seg.pt`) | COCO 闭集 | RGB，不输入提示 | box/mask/class | `YOLO("yolo26x-seg.pt")(image)` | 单模型 box+seg 主基线 |
| YOLO11 Detect (`yolo11x.pt`) | COCO 闭集 | RGB，不输入提示 | box/class | `YOLO("yolo11x.pt")(image)` | 与项目更成熟的 Ultralytics 稳定代际对照 |
| YOLO-Worldv2 (`yolov8x-worldv2.pt`) | 开放词汇、文本零样本 | RGB + 固定 class strings | box/class | `YOLOWorld(...); model.set_classes([...]); model(image)` | 较早但成熟的实时文本开放词汇基线 |
| YOLOE-26 Text (`yoloe-26x-seg.pt`) | 开放词汇、文本提示 | RGB + 固定 class strings | box/mask/class | `YOLOE(...); model.set_classes([...]); model.predict(image)` | prompt-free 召回不稳定是否来自无提示词表/分类头 |
| YOLOE-26 Visual (`yoloe-26x-seg.pt`) | one-shot 视觉提示 | RGB + 独立参考框 | box/mask，临时 `objectN` 类 | `predict(..., visual_prompts={"bboxes": ..., "cls": ...})` | 给定同类 TV 外观样本能否改善 Habitat 域内召回 |
| YOLOE-26 PF (`yoloe-26x-seg-pf.pt`) | prompt-free 开放词汇 | 只有 RGB | box/mask/内置类名 | 直接 `predict`，不得调用 `set_classes()` | 当前系统原样基线 |

官方资料说明：

- [YOLO26 官方文档](https://docs.ultralytics.com/models/yolo26)将其定义为 2026 年发布的原生端到端、默认 NMS-free 模型；五个 Detect 规模均支持 train/val/inference/export。官方报告 COCO mAP 40.9–57.5、T4 TensorRT 延时 1.7–11.8 ms，但这些数字只能作为官方硬件上的背景，不能替代本机重放实测。
- [YOLO11 官方文档](https://docs.ultralytics.com/models/yolo11)提供 `n/s/m/l/x` 五个 COCO Detect 权重和统一 `YOLO()` 接口；官方表中 `x` 为 54.7 COCO mAP50–95、11.3 ms T4 TensorRT10。YOLO11 没有文本或视觉提示接口。
- [YOLO-World 官方文档](https://docs.ultralytics.com/models/yolo-world)确认其为 YOLOv8-based 开放词汇 box detector，通过 `set_classes()` 设置文本类别并可保存 offline vocabulary；v2 的 `s/m/l/x` 均支持 export，官方也明确更推荐 v2。它不提供 YOLOE 式 visual prompt、prompt-free 词表或实例 mask。
- [YOLOE 官方文档](https://docs.ultralytics.com/models/yoloe)明确区分三种互不等价的模式：text、visual、prompt-free。文本与视觉模式使用 `*-seg.pt`；prompt-free 使用 `*-seg-pf.pt` 和内置 4,585 类词表，且会拒绝 `set_classes()`。视觉提示输出 `object0/object1/...`，实验代码必须自行映射回目标类。YOLOE-26 text 第一次 `set_classes()` 还会获取约 254 MB 的文本编码器，应将这一冷启动成本单列，不混进逐帧延时。
- YOLOE 原论文为 [YOLOE: Real-Time Seeing Anything](https://arxiv.org/abs/2503.07465)，YOLO26/YOLOE-26 的技术描述见 [Ultralytics YOLO26 论文](https://arxiv.org/abs/2606.03748)。

## 本场景的固定提示协议

主实验只检测 ObjectGoal 任务目标，避免不同模型因 prompt 数量不同而产生不公平开销。

### 单类主实验

- 闭集模型：从 COCO 输出中只取 `tv`，但推理过程保持模型原生 80 类，不通过结果过滤伪装成更低延时。
- YOLOE text 和 YOLO-Worldv2：均设置 `['television']`。
- YOLOE prompt-free：不设置提示；输出中的 `tv`、`television` 等标签在评分阶段映射为统一 canonical class。
- YOLOE visual：用**不属于评分帧集合**的一张 Habitat TV 参考图，在其中给出一个紧致 bbox；同一个 reference embedding 用于所有评分帧。

### ObjectGoal 六类扩展实验

固定 canonical 词表：`['chair', 'bed', 'potted plant', 'toilet', 'tv', 'couch']`。Habitat 的 `plant/sofa/television` 仅在输入输出适配层映射到上述 canonical 名，不能对某个模型额外添加更多同义词。若要评价同义词 prompt pack，应作为独立消融，并对所有文本提示模型使用相同列表和相同 duplicate-box merge 规则。

不得在最终评分帧上试 prompt 后再挑出最好的词。prompt、置信度阈值和 visual reference 都应在独立 calibration 帧上冻结。

## 准确率标注与指标

历史 RGB 图本身没有自动提供“正确框”。要声称检测准确率，必须建立 GT：

1. 首选：如果保存了同步 Habitat semantic observation、目标 instance id 和相机位姿，投影目标实例 mask，并从 mask 生成 bbox。
2. 次选：对保存的 RGB 帧人工标注目标 TV 的 bbox 与 `visible/ignore`；严重遮挡、画面边缘截断等规则在标注前固定。
3. 不可接受：把某一个 detector 的输出当成 GT；也不能用“导航最后离目标更近”代替检测准确率。

建议主指标：

- bbox `AP50` 与 `AP50:95`；样本量较小时至少报告 `Precision/Recall/F1 @ IoU 0.5`。
- 可见帧 target recall、不可见帧 false-positive rate、每 100 帧 FP 数。
- 第一次正确检出帧、连续漏检最长 gap、可见期间 detection persistence。
- 对导航 tracker 额外报告：满足其置信度/面积/深度门限的有效命中率，以及是否能形成连续两次独立观测。该项是工程指标，不能替代 AP/F1。
- YOLOE mask 可另报 mask AP；主对比仍必须统一到 bbox，因为 YOLO11/YOLO26/YOLO-World Detect 只输出 box。

至少同时给出两个 operating point：现有在线阈值，以及 calibration set 上选定的 F1 最优阈值。完整 confidence sweep 建议从 0.05 到 0.50，步长 0.05。

## 延时、显存与公平性协议

所有模型必须使用同一张 GPU、同一进程隔离策略、相同软件版本、`batch=1`、相同原始 RGB 顺序和 `imgsz=640`。先以 PyTorch FP16 比较在线实际路径；TensorRT/ONNX 只能作为单独部署表，不能与 PyTorch 结果混排。

每个模型记录：

- 冷启动：import、权重加载、移入 GPU、首次编译/算子初始化；文本模型另列 prompt encoder 下载（若发生）和 `set_classes()`/embedding 构建。
- 热身：至少 30 次不计入统计，并执行 `torch.cuda.synchronize()`。
- 热推理：逐帧记录 preprocess、GPU forward、postprocess，以及从解码后的 RGB 输入到结构化 detections 返回的 wall-clock end-to-end。
- 统计：mean、P50、P95、P99、max、FPS；GPU 时间使用 CUDA Events，wall time 使用 monotonic clock，计时边界前后均同步 CUDA。
- 资源：模型加载后的 idle VRAM、推理 peak allocated/reserved VRAM、进程 RSS；若测 CPU，再固定线程数并记录 CPU 型号。
- 稳定性：OOM、异常、空结果帧、被跳过帧数。禁止通过丢帧得到虚高 FPS。

官方网页上的 T4/TensorRT 数字只说明模型系列的理论位置。项目结论必须来自当前机器、当前 `ultralytics`/PyTorch/CUDA 版本和当前历史帧的重放结果。

## 推荐最终表格

主表每行是一个 `model + prompt mode`，至少包括：权重、参数规模、prompt、阈值、AP50、Recall、FP/100 frames、最长漏检 gap、首次正确命中帧、P50/P95 端到端 ms、P50 GPU forward ms、peak VRAM、RSS。

建议按以下顺序执行：

1. 先验证历史帧是否为未绘制 box 的原始 RGB，并建立 manifest（episode/step/timestamp/image path/hash）。
2. 生成或人工校验 TV GT，冻结 calibration/evaluation split。
3. 用当前 `yoloe-26x-seg-pf.pt` 重放，复现现有结果，作为 harness 验收。
4. 跑 `yolo26x.pt`、`yolo11x.pt`、`yolov8x-worldv2.pt`、YOLOE text、YOLOE visual。
5. 输出逐帧 JSONL、汇总 CSV/Markdown 和统一绘框视频；保留原图只读，不覆盖、不把绘框图混回输入目录。
6. 若 `x` 系列准确率接近，再跑 `s` 系列确定在线部署的准确率—延时 Pareto 前沿。

## 许可边界

- Ultralytics 主仓库的 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 是 AGPL-3.0；[Ultralytics 官方许可页](https://www.ultralytics.com/license)提供 AGPL-3.0 与 Enterprise 两条路径。YOLO11、YOLO26、通过 Ultralytics 使用的 YOLO-World/YOLOE 都应按这一许可边界审查。
- 原作者 [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World) 仓库标注 GPL-v3，并说明商业许可需联系作者。直接使用原仓库与使用 Ultralytics 移植版不是同一个许可判断。
- 原作者 [THU-MIG/yoloe](https://github.com/THU-MIG/yoloe) 仓库标注 AGPL-3.0。研究内部评测仍应保留版本、权重来源和许可证记录；未来若形成私有产品或机器人部署，应让项目负责人进行正式许可审查。

本文只记录官方资料和实验设计，不构成法律意见。
