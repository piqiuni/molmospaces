# MolmoSpaces 官方 NavToObj benchmark：步数与成功判定核对（2026-07-29）

本笔记只核对 AllenAI `molmospaces` 官方 `main`（提交 `c2f1b583f087e1d3994e1377574843b759d9d0f8`）的公开源码与官方 benchmark README；不改变本项目运行代码。

## 结论

1. **500 步是官方配置中的默认/示例值，但不是每个 benchmark episode 不可覆盖的固有规则。**
   - 通用 JSON benchmark eval 配置 `JsonBenchmarkEvalConfig` 定义 `task_horizon = 500`；NavToObj 数据生成基类也定义 `task_horizon = 500`。[通用 eval 配置](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/evaluation/configs/evaluation_configs.py#L100-L106) [NavToObj 基类](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/configs/base_nav_to_obj_config.py#L34-L38)
   - 官方 README 的评测命令也显式传入 `--task_horizon_steps 500`，说明这是建议的运行参数，而非被 JSON episode 锁死的数字。[官方 README 示例](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/evaluation/README.md#L85-L96)
   - 可以增大或减小：CLI 的 `--task_horizon_steps` / `--task_horizon_sec` 是互斥的显式覆盖参数；运行时优先使用显式覆盖。[参数定义](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/evaluation/eval_main.py#L184-L195) [优先级实现](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/evaluation/eval_main.py#L289-L353)

2. **当前官方 runner 有一个需要明确的细节：无 CLI 覆盖时，它实际读取 episode `task.task_horizon_sec`，而不是自动落回上述 500。**
   - `determine_task_horizon()` 的顺序是：显式 steps/sec 覆盖优先；否则读取所有 episode 的 `task_horizon_sec` 并按 `policy_dt_ms` 换算成步数；若该字段缺失则报错要求显式传参。[实现](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/evaluation/eval_main.py#L289-L353) [调用点](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/evaluation/eval_main.py#L594-L619)
   - 这与 `benchmark_schema.py` 顶部“task horizon 不存到每 episode、应从 eval 参数获得”的说明不完全一致；对实际运行应以 runner 代码为准。为可复现比较，应在所有方法上显式固定同一 `--task_horizon_steps`（或固定同一秒数与 policy rate），并在结果中记录它。[schema 说明](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/evaluation/benchmark_schema.py#L1-L31)

3. **原始 `NavToObjTask` 的成功判定要求目标当前可见。**
   - 它在 `head_camera` 上调用 `check_visibility()`，并只接受严格大于零的可见比例；不是“曾经看见过”即可。[可见性检查](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/tasks/nav_task.py#L237-L243)
   - 成功条件是：目标当前可见 **且** 平面距离严格小于 `succ_pos_threshold`；因为 reward 为 `max(0, 1 - distance / threshold)`，`judge_success()` 又要求 reward 大于零。默认阈值为 1.5 m。[成功实现](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/tasks/nav_task.py#L245-L279) [默认距离阈值](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/configs/task_configs.py#L171-L187)
   - 原始判定比当前 native 策略的语义门槛更宽松：官方只要求 `visibility_fraction > 0`，没有 16 pixels、0.2 visibility fraction 或连续两帧这些额外条件。

4. **是否会“走过头后遮挡”取决于 eval 配置。**
   - Pipeline 在每一步看到 `infos[0]["success"]` 后，若 `end_on_success=True` 就立刻停止并保留成功；否则循环结束时重新按最后状态调用 `judge_success()`。[rollout 实现](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/data_generation/pipeline.py#L744-L804)
   - 官方 `PiPolicyEvalConfig` 设置了 `end_on_success=True`，因此若途中某一帧已同时满足“<1.5 m + 当前可见”，官方 Pi 路径会在那一帧成功退出，不会继续接近到把物体挤出视野。[Pi eval 配置](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/evaluation/configs/evaluation_configs.py#L186-L195) [参数传递](https://github.com/allenai/molmospaces/blob/c2f1b583f087e1d3994e1377574843b759d9d0f8/molmo_spaces/data_generation/pipeline.py#L1031-L1041)

## 对当前四场景分析的含义

- 不能把当前 420/444 步的 distance-adaptive screening horizon 称为“官方 500 步的原样评测”；它是本地加速筛选策略。若要报告可比较的 benchmark 结果，应关掉动态缩短或把每个 episode 的上限显式固定为同一个值（例如 500 或经过预先声明的更大值）。
- 目标在接近后因视场、朝向或遮挡变得不可见，按原始官方 `NavToObjTask` 也会失败；因此终点应是一个能同时满足距离和 head-camera 可见性的观察位，而不是物体中心或任意最近点。
- 若中途已满足官方的最小条件，官方 Pi eval 会因 `end_on_success=True` 锁存成功。当前系统若只在 move_base 回调后的单帧复核，则与该行为不同，可能错过已经满足过的成功帧。
