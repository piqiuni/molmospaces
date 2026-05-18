# 交互式导航项目讨论记录

**日期：** 2026-04-11  
**参与者：** 皮丘尼  

---

## 讨论要点

### 1. 关于论文的定位分析

皮丘尼对三篇最相关论文的精准判断：

- **论文1 (Manipulate-to-Navigate, 2508.13151)**：重点偏向操作与控制，关注机器狗的身体控制、操作位置、移动位置等操作细节
- **论文2 (Interactive Navigation, CMU, 2410.13418)**：与目标更相关，但重点同样在控制上
- **论文3 (OVMM + 3D Maps, 2406.18115)**：整体实现更接近，其评测方法与数据集可能对我们很有用

**我们的差异化定位：** 更加关注**交互图的增加与构建**，而非操作细节。目前只需要确定：
- 哪些物体有交互属性
- 是哪种交互（开关/按钮/推拉）

而非关注具体的 manipulation 控制策略。

---

### 2. 仿真环境困境

**问题：**
- 当前基础导航框架基于 **AI2-THOR** 仿真器
- AI2-THOR 不够真实，场景少
- 大规模数据集如 **Habitat** 不支持交互
- 支持交互的仿真器（如 OMNIGIBSON）场景小、少，且偏操作方向

**核心矛盾：** 导航规模大 vs 交互场景小

---

### 3. 课题立意拔高

**论证思路：**

1. **任务不可达性论证**：大量目标在物理上不可达（门关着、抽屉关着），纯导航成功率存在上界
2. **效率提升论证**：交互允许"抄近路"（穿门 vs 绕行）
3. **主动感知论证**：打开门/抽屉 = 获取新观测空间
4. **与现实需求对齐**：家庭服务机器人、仓库机器人都需要交互能力

---

### 4. MolmoSpaces 平台评估

**论文：** arXiv: 2602.11337 (2026.02) | GitHub: allenai/molmospaces  
**链接：** https://arxiv.org/abs/2602.11337

**关键特性：** 230k+ 室内环境 | 130k 对象资产 | 48k 可操作对象 | 42M 抓取姿态 | Simulator-agnostic (MuJoCo/Isaac/ManiSkill) | sim-to-real R=0.96

**Benchmark 8 个任务：** navigate-to, pick, pick-and-place, pick-and-place-next-to, pick-and-place-color, open, close, **open-door**

**导航支持：** 支持 mobile manipulation 和 navigate-to，多房间场景，但导航评估相对简单（没有 VLN/ObjectNav 级别的复杂导航 benchmark）

**对我们项目的价值：** ⭐⭐⭐⭐⭐
- open-door 任务直接对应核心场景
- 48k articulated object = 丰富的交互对象（门、抽屉等）
- 多房间场景 = 跨房间导航 + 交互
- 可以定义自定义 benchmark

**技术栈：** MuJoCo 为主引擎 | Franka Droid（移动底座）| benchmark.json 定义 episode | 支持自定义 policy

---

### 5. 论文2 (Interactive Navigation, CMU) 详解

**训练方法：**
1. 离线：PyBullet 仿真中训练 basis function（学习物体 SE(2) 动态）+ 机器人动力学模型
2. 在线：自适应动态模型集成到 MPPI 控制器，每次交互后更新物体动态参数
3. 决策切换：可推动 → adaptive pushing；不可推动 → collision-free path 绕行

**评测：** 3 类对象 × 多种物理属性组合 | 动态模型预测 MSE | 真实 Shmoobot 部署

---

### 6. 课题立意确认

**核心贡献 = 交互图的构建与感知，操作部分不作为考核点**

1. 判定哪些物体有交互属性
2. 判定交互类型（开关/按钮/推拉）
3. 将交互信息链接到场景拓扑/语义图
4. 利用交互信息改进导航规划

**操作部分处理：** 实验中判断可交互后直接运行现有交互指令完成交互（类似 oracle interaction），考核的是"是否正确识别了需要交互"和"交互后导航效率是否提升"

**framing 好处：** 贡献聚焦感知和表征 | 避免 manipulation 复杂性 | 用 MolmoSpaces open/close 作交互 oracle | 评估指标清晰

---

### 7. 仿真平台建议

**直接迁移到 MolmoSpaces**，利用其 MuJoCo 后端和丰富场景。

---

## 后续行动项

- [ ] 克隆 MolmoSpaces 仓库，搭建环境并跑通 demo
- [ ] 分析 MolmoSpaces 的 open-door 任务定义和评估方法
- [ ] 设计"交互式导航"自定义 benchmark（基于 MolmoSpaces 框架）
- [ ] 量化分析"交互属性必要性"——在 ProcTHOR 场景上统计被阻塞目标比例和路径效率差异
- [ ] 确定一阶段的具体 deliverable 和时间线
