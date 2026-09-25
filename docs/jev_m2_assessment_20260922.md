# Jev 对交互导航 M2 的适用性评估

日期：2026-09-22。范围：公开资料核实与本地请求静态检查；未调用 Jev 推理 API、未修改算法或配置、未提交 Git。

## 结论

**值得作为 M2 的低延迟候选选择器进行离线对照，但当前没有证据支持直接替换 Qwen。** M2 已经由算法产生有限候选，再让模型决定选谁，与 Jev 的 Choice 接口很匹配；然而本任务需要综合目标语义、门/容器状态、空间关系、历史失败与多阶段推进，这不等同于简单分类。较稳妥的后续路线是先测同一冻结数据上的单一 Choice，再研究低置信度回退 Qwen；不要先引入多维手工加权评分，导致模型效果与规则改动混在一起。

## 来源身份与接口边界

- 用户提供的 [jevai.org](https://www.jevai.org/) 自称社区站，明确模型、基础设施与官方 API 属于 TypeSafe AI；它是自身社区封装接口的一级来源，不是模型厂商。
- [社区文档](https://www.jevai.org/docs) 的 `/api/v1/decisions` 使用 `state`、`questions`，声明请求体上限 **32 KiB**。这是该网关限制，不能推广为 Jev 模型上下文限制。
- [厂商 API](https://docs.typesafe.ai/api) 是 `POST https://api.typesafe.ai/v1/systemone`，并非 OpenAI Chat Completions；更换 `base_url/model` 不足以接入。网关模型名示例 `typesafe-ai/jev` 与厂商模型 ID 也不能混用。

## 已核实能力

| 项目 | 文档事实及对 M2 的意义 |
|---|---|
| 当前模型 | `jev-1.13.0`；应固定版本并记录返回 ID，避免 `jev-latest` 更新影响复现。 |
| 输入 | 仅文本、JSON、文本数组；不能直接替代视觉感知模块，但当前 M2 的结构化上下文可承载。 |
| 上下文 | 每请求总计 64k tokens，`state + 最长单个 question` 为 32k tokens；不是 32 KiB。 |
| 价格 | 输入 $0.042/百万 tokens，输出免费；社区代理可能有不同费用。 |
| 限流 | 公布 250,000 tokens/s、1,200 requests/min，厂商明确会动态调整；真实账户配额仍需确认。 |
| 语言 | 英文是主要训练语言；其他语言包括 CJK 可处理，但厂商要求在自身数据上验证。 |
| 定制 | 文档不提供按客户 fine-tuning/LoRA；通过 state、instructions、criteria 定制。 |

上述规格来自 [厂商 Models](https://docs.typesafe.ai/models)，截至本日。

- **Choice**：候选最多 255 项；返回一个选项、覆盖全部选项且和为 1 的概率分布及 confidence。当前冻结样本最多 16 个候选，不触及此上限。可在代码中按概率排序，生成 M2 所需排名；文档没有独立的 listwise ranking 原语，概率排序不应声称等同于对完整动作序列的规划。[Choice](https://docs.typesafe.ai/primitives/choice)
- **Score**：2–10 个有文字描述的有序等级，返回等级概率的期望值，不是连续精确测量；可以每候选一问，但不同问题的评分一致性要单独验证。[Score](https://docs.typesafe.ai/primitives/score)
- **Noul**：二元判断为真的概率，可问“这个动作是否与刚刚失败的条件相同”。它不是 choice 的可互换写法。[Noul](https://docs.typesafe.ai/primitives/noul)
- **Confidence**：由该回答的概率分布形状计算的统计量，不是“此动作物理执行成功概率”，更不是机器人最终找到目标的概率。阈值必须用本任务验证，不能直接采用示例的 0.9。[Confidence](https://docs.typesafe.ai/confidence)

## 证据强度与风险

官方发布材料宣称 70–500 ms 端到端延迟，同时说明测量通常来自美国西海岸、演示短输入有利于模型。Workflow 评测使用其他强模型的平均预测作为参考，并非机器人真值；“零幻觉”在其解释中是输出模式约束，不代表语义决策永远正确。因此不能把宣传速度直接与本地 Qwen 的 7.06 s 平均值相除，也不能由此预言 M2 接受率。[发布材料与方法说明](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

官方自一致性示例仅在一个保险案例上重复 15 次，并插入变化的 uid；它报告低方差，但这不是跨场景概率校准，更不是导航成功率证据。[自一致性示例](https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook)

厂商明确列出的弱点包括数值运算、多跳间接推理、无关细节较多的长上下文、指令和条件冲突，以及不同问法之间不保证概率恒等关系。[已知弱点](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

对本任务的推论：

1. 距离、房间包围关系、最短通路、前沿长度继续在代码中计算；不要依靠 Jev 从原始坐标精确推导。
2. “开门后应穿门”“某房间已进入但仍有未探索前沿”要提供明确状态与因果事实；模型不能修复上游互相冲突的阶段字段。
3. 保留近期失败证据，但去除确实无关的重复字段；不能为适配而暗中丢弃长难例。
4. Choice 中多个同样合理动作会分散概率，低 confidence 可能是多解而不是危险；应联合可接受动作标签分析回退，而不是机械阈值。
5. 原始选择、守卫后选择必须分开评分；仍然保留硬可行性过滤和执行安全检查。

## 当前请求的体积适配

主 agent 对本地冻结的 166 条 `full_context` 静态构造最小 Choice 封装：`state=原 request 去 instruction`，问题 instructions 使用原提示词 B，criteria 为候选 ID 到相同 ID 的映射；采用 UTF-8、`ensure_ascii=False`、紧凑 JSON，没有联网发送。

| 指标 | 请求体字节数 |
|---|---:|
| 平均 | 11,002.22 |
| 最小 | 5,484 |
| 最大 | 27,018 |
| 超过 32,768 字节 | 0/166 |

最大样本为 `episode_002002-g3608-c2210-f5c7cbdf0b8e`，16 个候选。输入来自 `/home/ldl/outputs/interactive-nav/m2-context-remote-20260922/inputs/cases.jsonl`。

这只证明**该最小封装**满足社区字节上限，不是 Jev tokenizer 实测，也不代表完整适配已验证。增加候选描述、多个问题或使用 ASCII 转义后应重新统计。正式输入必须给选项清晰描述、将 B 的 JSON 输出格式要求改为 Choice 接口要求，保留其决策语义；不能把改过的协议称为逐字不变提示词实验。

## 开源与部署

检查的 [厂商 GitHub](https://github.com/typesafe-ai) 公开 SDK、适配器、skills 等代码；未发现 Jev 权重下载、推理实现或模型权重开放许可。SDK 的 MIT 许可不等于模型权重许可。现阶段应按托管服务接入来规划，不能像现有 Qwen 那样承诺双卡本地部署。关于未公开的权重、参数规模、训练数据详情，不作推断。

## 建议的后续试验（本轮未执行）

1. 使用当前 166 条冻结案例、130 条主评分及原标签；分别报告 57 条严格图版本对齐样本和 109 条较早图重建样本，避免重建噪声掩盖差异。
2. 第一组仅换为 Jev Choice：保持候选与上下文信息不变，B 只做协议转换，不增加启发式规则。现有对照为 Qwen 108/130、GPT-5.6-Luna 105/130、Gemini 107/130，均是离线合理动作接受率而非导航 SR。
3. 同时报接受率、阶段推进、重复失败动作、原始/守卫后选择、完整请求耗时 p50/p95、错误/重试、实际 tokens 与费用。多个可接受动作的案例不能强行指定唯一标签。
4. 若准确率接近，再在开发样本校准 confidence/概率间隔回退 Qwen；报告回退率和整体时延，使用新 episode 验证，现有留出集已经被分析过。
5. 未有可信的本任务实测前，保持 Qwen 为默认 M2。Jev 首要价值假设是降低决策时延和格式错误，并非已有证据证明提高导航成功率。
