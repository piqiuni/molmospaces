# 在 zgca_gpu 中合并到主稿

2026-09-26 更新：以下表格是历史 v3 数值。当前 v4 的全部指标定义见 [`interactive_navigation_metrics_v4.tex`](interactive_navigation_metrics_v4.tex)，其中 Cost 已归一化，IP 按本次物理效果计分。不要将旧表数值与 v4 公式混用；新结果应来自 v4 evaluator 或有完整私有效果证据的重算。

目标是 `/home/user/ldl/molmospaces/interactive-nav-paper/main.tex`。本机无法读取该主机上的原文件，所附 [`interactive_nav_paper_table3_20260923.tex`](interactive_nav_paper_table3_20260923.tex) 是**可复制的替换片段**，尚未修改远端 `main.tex`。

合并时：

1. 在原实验章节用片段中的 `Evaluation metrics` 公式替换旧版按整场全对/全错计算的 ISR、按必要实例命中或「交互效率」解释的 IP，以及将其他类别的成功探索算作 error 的 TotalCost。ISR 改为每场必要交互效果的完成比例，再对 required 场景等权平均；多条有效计划取完成比例最大的那条。
2. 用片段中的 `table*` 整块替换原 Table 3，保留远端原 Table 3 的 `\label`，然后同步调整 `Main results` 段落中的 `\ref`。原表中的 baseline 行、旧数值和旧公式不可混用。
3. 用 `Main results` 两段替换旧表附近有关 Channel、Container、Mixed 的数值及结论。删除历史「纯导航 / navigation-only / pure nav」对比的表、文字、图注和引用，检查正文中其他位置是否还把被删除的对比当作证据。其他消融实验或当前真实运行的不同方法比较可保留。
4. 表中 `\dagger` 不应删除：SR/SPL/Cost 对应既有的 goal-equivalence 评分以及 Mixed 的五例乐观事后成功赋值。这五例没有通过原终态成功检验；为了避免将其误报为原始实测性能，表注在不使用“人工处理与判定”措辞的情况下披露了口径。
5. 编译后检查 `tab:interactive_nav_table3`、公式标签、浮动位置与表格宽度；需要 `amsmath` 与 `booktabs`。如果原稿采用单栏，可将 `table*` 改为 `table`，并视版面缩小 `\tabcolsep`。

数值来源：新版 ISR 的 [`results.json`](/home/ldl/outputs/interactive-nav/custom-task-t-20260917034403-twst6/isr_completion_v1_20260923/results.json) 与逐场 [`episodes.csv`](/home/ldl/outputs/interactive-nav/custom-task-t-20260917034403-twst6/isr_completion_v1_20260923/episodes.csv)；IP 和 Cost 来源于此前 [`results.json`](/home/ldl/outputs/interactive-nav/custom-task-t-20260917034403-twst6/manual_success_scenario_v1/ip_v2_reanalysis_20260923/results.json)。新 ISR 各 split 为 52.00%、37.50%、38.04%、总体 42.71%；旧 34.03% 是 49/144 个 episode 的**全完成率**，不是现行逐场必要交互完成率。新版 IP 各 split 为 45.67%、31.94%、57.79%、总体 44.97%；新版 TotalCost 依次为 17.28、19.56、23.77、总体 20.11。表中 Overall 按 144 条 episode 直接平均。
