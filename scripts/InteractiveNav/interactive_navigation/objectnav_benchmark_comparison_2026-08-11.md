# MP3D、HM3D-v1 与 HM3D-v2 ObjectNav 基准比较（2026-08-11）

## 结论摘要

截图中的“`HM3D-v2 > HM3D-v1 > MP3D`”是许多方法在特定实现下呈现的**常见现象**，不是三个基准从易到难的普遍定律，更不能把三列 SR 当作同一把难度尺。截图出自 [IntentNav 的 Table 1](https://arxiv.org/html/2606.08029#S4.T1)；其中 `SG-Nav-GPT` 恰好是反例：MP3D/HM3D-v1/HM3D-v2 的 SR 分别为 `40.2/54.0/49.6`，即 v1 高于 v2。

对采用单层 2D 地图的零样本方法，v2 较高的一个直接、可验证解释是：2023 Challenge 明确把 v2 episode 更新为“不必跨楼层”，而 v1 与 MP3D 中的跨楼层 episode 会使单层地图方法系统性失败。[ApexNav 原论文](https://arxiv.org/html/2504.14478#S5) 的失败分析也观察到，除主要由单层任务组成的 HM3D-v2 外，其余两个基准有超过 13% 的 ``Different Floor`` 失败。这个结论描述的是该类算法与该采样协议的耦合，**不等于 HM3D-v2 对所有 ObjectNav agent 都更容易**。

## 名称到底指什么

- **MP3D**：Matterport3D，是独立的真实室内扫描数据集。2021 Habitat ObjectNav Challenge 在 90 个 MP3D 场景上使用标准 train/val/test split，并从 40 个标注类别中筛选出 **21 个**目标类别。[官方 2021 Challenge 说明](https://github.com/facebookresearch/habitat-challenge/tree/challenge-2021#dataset)
- **HM3D-v1**：论文表格中的简称通常指 `objectnav_hm3d_v1.zip`，即基于 **HM3D-Semantics v0.1** 的 2022 ObjectNav episode 版本；2022 Challenge 使用 120 个带语义的 HM3D 场景，split 为 80/20/20，目标为 **6 类**（chair、couch、potted plant、bed、toilet、tv）。[官方 2022 Challenge](https://github.com/facebookresearch/habitat-challenge/tree/challenge-2022#dataset)
- **HM3D-v2**：通常指 `objectnav_hm3d_v2.zip`，即基于 **HM3D-Semantics v0.2** 的 2023 ObjectNav episode 版本；2023 Challenge 使用 216 个场景，split 为 145/36/35，仍是同样的 **6 类**目标，并筛成不需要跨楼层的 episode。[官方 2023 Challenge](https://aihabitat.org/challenge/2023/#dataset)

Habitat-Lab 的[官方数据清单](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets)也明确将这三份 episode 包分别映射为 `objectnav_mp3d_v1.zip`、`objectnav_hm3d_v1.zip`（HM3DSem-v0.1）和 `objectnav_hm3d_v2.zip`（HM3DSem-v0.2）。因此，论文中的 `v1/v2` 首先是 **ObjectNav episode + HM3DSem 语义资产的版本标签**；不要仅凭 Habitat 配置类名 `ObjectNav-v1` 判断数据版本。该类名可同时搭配不同的具体数据路径，实际应检查 ZIP 名、`data_path` 与 scene/semantic-config 版本。[Habitat 配置键说明](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/CONFIG_KEYS.md#dataset)

原始 **HM3D** 则是包含 1,000 个 Matterport 空间的 mesh/texture 集合；HM3D-Semantics 是加在其中部分场景上的致密语义层，而不是 MP3D 的重命名。[HM3D 官方仓库](https://github.com/matterport/habitat-matterport-3dresearch#habitat---matterport-3d-research-dataset)、[HM3DSem 原论文](https://arxiv.org/html/2210.05633#S3)

## 核心差异

| 维度 | MP3D ObjectNav | HM3D-v1 ObjectNav | HM3D-v2 ObjectNav |
|---|---|---|---|
| 语义场景来源 | MP3D；2021 Challenge 的 90 场景、21 个目标类别。[官方说明](https://github.com/facebookresearch/habitat-challenge/tree/challenge-2021#dataset) | HM3D-Semantics v0.1；2022 Challenge 的 80/20/20 train/val/test 场景、6 类目标。[官方说明](https://github.com/facebookresearch/habitat-challenge/tree/challenge-2022#dataset) | HM3D-Semantics v0.2；2023 Challenge 的 145/36/35 train/val/test 场景、6 类目标。[官方说明](https://aihabitat.org/challenge/2023/#dataset) |
| 语义标注 | MP3D 以 mesh segment 为主；HM3DSem 论文展示其与渲染 RGB mesh 存在可能错位/过度聚合。 | HM3DSem 的纹理级实例语义层，与原始几何/RGB texture 对齐。 | 同为纹理级语义层；v0.2 扩充并修正 v0.1 的标注与场景。 [HM3DSem 原论文](https://arxiv.org/html/2210.05633#S3) |
| v1→v2 的资产变化 | — | 初版 HM3DSem：80/20/20 场景。 | 扩至 145/36/35；修复语义拼写/类别错误，并在若干场景人工清除碎片以改善可导航性。[官方 changelog](https://github.com/matterport/habitat-matterport-3dresearch/blob/main/CHANGELOG.md#v02---2022-10-18) |
| canonical Challenge 机器人与动作 | 2021：0.88 m 高 RGB-D、79° HFOV、离散 ObjectNav 动作、最多 500 steps 的本地配置。[官方 config](https://raw.githubusercontent.com/facebookresearch/habitat-challenge/challenge-2021/configs/challenge_objectnav2021.local.rgbd.yaml) | 2022：同属 0.88 m、640×480、79° HFOV、离散动作配置。[官方 config](https://raw.githubusercontent.com/facebookresearch/habitat-challenge/challenge-2022/configs/challenge_objectnav2022.local.rgbd.yaml) | 2023：HelloRobot Stretch、`velocitycontrol` 连续动作、1.41 m robot / 1.31 m camera、42° HFOV、最多 500 seconds。[官方 config](https://raw.githubusercontent.com/facebookresearch/habitat-challenge/main/configs/benchmark/nav/objectnav/objectnav_v2_hm3d_stretch_challenge.yaml) |
| 楼层约束 | 不保证单层。 | 不保证单层。 | 官方明确所有 episode 都无需跨楼层。[官方 2023 Challenge](https://aihabitat.org/challenge/2023/#new-in-2023) |

两代 HM3D Challenge 的官方成功判定都采用：STOP 时距任一目标实例 1 m 内，且从该位置可由 oracle 视角看到目标；SPL 的最短路以起点最近的目标实例为基准。[2022 规则](https://github.com/facebookresearch/habitat-challenge/tree/challenge-2022#evaluation)、[2023 规则](https://aihabitat.org/challenge/2023/#evaluation)

## 为什么很多行会出现 v2 > v1 > MP3D

### 1. v2 的单层 episode 对 2D 地图方法特别友好

2023 官方变化并不只是“换成新数据”：它明确说明 v2 已将 episode 更新为无需跨楼层。[官方说明](https://aihabitat.org/challenge/2023/#new-in-2023) 对只维护单层占据图/BEV 的 agent，楼梯、楼层连通性与垂直目标本来就是额外任务；去掉这类 episode 会直接减少一种不可恢复的失败模式。ApexNav 的作者在其同一套三基准实验中也将 v1/MP3D 的大量跨楼层失败与单层 2D map 相关联。[失败分析](https://arxiv.org/html/2504.14478#S5)

这是截图里多数 map-based/VLM 方法 v2 偏高的最强解释之一；它同时解释了为何不能把这种排序推广为“v2 视觉或语义一定更简单”。带多层拓扑图、楼梯策略或不同 embodiment 的方法未必得到同样的相对收益。

### 2. v0.2 修复了标注与可行走性问题

HM3D 官方 changelog 对 v0.2 记录了两类与 ObjectNav 直接有关的修复：修正 v0.1 中的类别/拼写错误（例如 `toilet`、bed/sofa 错标），以及在一批场景中人工清除 debris 来“improve navigability”。[官方 changelog](https://raw.githubusercontent.com/matterport/habitat-matterport-3dresearch/main/CHANGELOG.md) 前者会改变目标实例及其评测语义，后者会减少局部规划/碰撞/卡死问题；二者都可能推高某些方法的 SR，但不足以单独证明整个 v2 基准更容易。

### 3. MP3D 的目标词表更大，且地图资产对 RGB-D 建图更苛刻

MP3D 的 21 类目标，是 HM3D-v1/v2 六类固定目标的 3.5 倍。目标类别数量、类别频次、检测器是否覆盖、类别间视觉相似性都会改变“看到目标并正确 STOP”的难度；所以 MP3D 与 HM3D 的 SR 不应被当作只改变场景的对照实验。[MP3D 21 类](https://github.com/facebookresearch/habitat-challenge/tree/challenge-2021#dataset)、[HM3D 六类](https://github.com/facebookresearch/habitat-challenge/tree/challenge-2022#dataset)

此外，HM3DSem 原论文说明其纹理级语义是为了避免 MP3D 自动语义 mesh 与原始 RGB mesh 的错位、边界不准和过度聚合问题。[标注格式比较](https://arxiv.org/html/2210.05633#S3) 在截图所代表的 RGB-D map 方法上，这类资产差异会影响可见目标定位和地图一致性。作为具体而非普遍的实证，ApexNav 报告 MP3D 某些场景有导致深度缺失的 ``black holes``，会造成 map mismatch 和卡住。[ApexNav 失败分析](https://arxiv.org/html/2504.14478#S5)

### 4. scene/episode 集合变了，样本数本身不规定难度方向

从 v1 到 v2，语义场景 split 从 80/20/20 扩至 145/36/35；这意味着验证场景、目标实例分布和起终点抽样都变了，而不是在同一批 episode 上只升级标注。[官方 changelog](https://github.com/matterport/habitat-matterport-3dresearch/blob/main/CHANGELOG.md#v02---2022-10-18) 以截图中复用的 ApexNav 设置为例，其作者实际评测的是：HM3D-v1 `2000 episodes / 20 scenes / 6 classes`、HM3D-v2 `1000 / 36 / 6`、MP3D `2195 / 11 / 21`。[实验设置](https://arxiv.org/html/2504.14478#S5) 这些是该论文的评测子集/包配置，不能外推成所有“v1/v2”的固定 episode 数。

增加场景数可能提高外观与布局多样性，也可能因采样策略、单层过滤、目标实例可见性和距离分布而提高 SR；仅看场景数量无法推出哪一列应更高。

## 截图的表应怎样读

1. 同一**行内**的三列可用来描述该论文所报告的跨域迁移现象，但仍须核对其 config。
2. 同一**列内**、且明确使用相同 split、episode 包、动作、传感器、步数/时间上限和成功判定的结果，才适合横比方法。
3. 不应把不同论文的同名 “HM3D-v2” 结果直接与 2023 官方 Challenge leaderboard 相比。官方 v2 Challenge 是 Stretch 连续控制；而 ApexNav 为统一三套数据，把实验设为离散 `0.25 m / 30°` 动作、500 steps、`0.2 m` success distance。[ApexNav 设置](https://arxiv.org/html/2504.14478#S5) 这与官方 v2 配置不同，数值没有自动可比性。
4. 截图的 IntentNav 表本身也不是“每行完全同协议的官方榜单”：它汇总了不同论文的已报告/复现结果。论文只声明在三个 validation benchmark 上评测，并说明 MP3D 是训练域、HM3D 是零样本迁移域；阅读时应把 SR 差异首先视作「数据 + episode + protocol + 训练域」的合成结果。[IntentNav 实验设置](https://arxiv.org/html/2606.08029#S4)

## 对本机评测的建议

如果目标是报告可复现的 Habitat v2 成功率，应固定并在结果文件中写全：

1. `objectnav_hm3d_v2.zip`、HM3D/HM3DSem 资产 release、semantic config 与其校验值；
2. `val`/`minival`/官方隐藏 test 的身份、episode JSON 的 hash 与 episode 数；
3. 目标类别表、起点—目标 geodesic 距离分布、跨楼层比例；
4. Habitat-Sim/Lab/challenge tag、robot、camera height/FOV/resolution、RGB/RGB-D/GPS/Compass、动作空间、最大 steps/seconds、碰撞与噪声设置；
5. SUCCESS 的距离/visibility 规则、STOP 策略、SPL 实现；
6. 检测器/VLM/训练数据、是否针对该 benchmark 调参，以及随机 seed。

建议优先以[官方 2023 v2 Challenge 配置](https://github.com/facebookresearch/habitat-challenge#task-objectnav)做主结果，并将其标为“HM3D-v2 / Stretch / continuous”。若要复现截图类型的数值，应另建“HM3D-v2 / LoCoBot-style discrete”的实验标签，严格按目标论文代码和 episode 包配置运行；两者不要放入同一张可直接比较的 SR 排行表。

