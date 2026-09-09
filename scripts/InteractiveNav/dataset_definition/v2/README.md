# Historical Container Interaction Format v2

本目录归档旧容器交互 benchmark 的真实历史输出，内部 schema version 为：

```text
interactive_nav_v2
```

## 文件来源

`benchmark.json` 是以下文件的字节级直接副本：

```text
scripts/InteractiveNav/output/container_interaction_benchmark_preview_first10/benchmark.json
```

复制时没有重新运行场景、容器扫描、关节动作、路径规划、可见性检测或图片渲染。

归档内容：

- 29 条 episode。
- house：`0、1、100、101、102`。
- SHA-256：`af1221f8fa5d73a3953057cda8d3b834ec819dae1b5451331fba4c7b45ffff91`。

## 历史结构

每条 episode 的容器扩展主要包含：

```text
interactive_nav
  schema_version
  interaction_domain
  case_id
  parent_benchmark_episode_index
  target
  initial_state
  oracle_plan
  oracle_plans
  generation_validation
```

v2 已具有容器目标、multi-oracle、观察姿态和关节操作记录，但其 step/reason 字符串、interaction 表达、成功条件和验证字段尚未与通道门数据统一。
