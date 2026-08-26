> 历史文档说明
>
> 本文件保留为交互导航项目的早期初始化记录。
> 当前项目的高层主入口已转移到以下 4 个核心文档：
>
> - `/home/user/ldl/molmospaces/TODO.md`
> - `/home/user/ldl/molmospaces/readme_pi.md`
> - `/home/user/ldl/molmospaces/AGENTS.md`
> - `/home/user/ldl/molmospaces/test.md`
>
> 如果本文件与上述核心文档存在冲突，以 4 个核心文档为准。

# 交互式导航项目 (Interactive Navigation)

## 项目概述

本项目是关于**交互式导航**的研究项目。目标是在基础机器人导航能力的基础上，增加与环境的交互能力，使得机器人能够通过交互（开门、开冰箱、开衣柜等）完成更加具体的导航/探索任务，提高整体效率。

## 核心思路

- 在传统语义地图引导导航（或类似 SLAM 框架导航）基础上，增加**地图交互层**
- 实时感知周围**可交互对象**（门、冰箱、衣柜等与 manipulation 相关的物品）
- 将可交互对象链接到**场景拓扑/语义图**中
- 使机器人具备通过交互（如开门）来完成导航/探索任务的能力

## 项目信息

- **项目名称：** 交互式导航 (Interactive Navigation)
- **项目路径：** `/home/ubuntu/daily-tasks/Research/interactive_navigation/`
- **项目类型：** 研究项目
- **创建时间：** 2026-04-11

## 相关研究方向

1. **语义地图构建 (Semantic Mapping)**：open-vocabulary 3D 语义地图、场景图 (Scene Graph)
2. **可交互对象感知 (Interactive Object Perception)**：articulated object 检测与 affordance 理解
3. **移动操作 (Mobile Manipulation)**：导航+操作一体化框架
4. **Object-Goal Navigation**：基于语义信息的目标导航
5. **Affordance 理解**：可交互区域的视觉 grounding

## 关键技术组件

- **感知层**：VLM/VLM-based 开放词汇检测、3D 语义重建
- **地图层**：语义地图 + 交互属性标注（可开关、可移动等）
- **规划层**：LLM 驱动的层次化任务规划（导航 → 交互 → 继续导航）
- **执行层**：导航策略 + 交互操作策略

## 文献调研

见 `survey_2026-04-11.md`

## 待补充

- 团队成员
- 当前进度
- 实验平台信息
- 具体实验方案
