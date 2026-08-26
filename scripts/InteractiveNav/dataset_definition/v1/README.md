# Historical Door Interaction Format v1

本目录归档旧通道门 benchmark 的真实历史输出，内部 schema version 为：

```text
door_interaction_nav_v1
```

## 文件来源

`benchmark.json` 是以下文件的字节级直接副本：

```text
scripts/InteractiveNav/output/door_interaction_benchmark_v1_front30_len05/benchmark.json
```

复制时没有重新运行场景、路径规划、门状态判断或图片渲染。

归档内容：

- 34 条 episode。
- house：`1、100、102、104、105、107、108、110`。
- SHA-256：`e7a8cd1ed6f3e9268f145f318018f6bc04ee0d9faffaa60789495340af83be5a`。

## 历史结构

每条 episode 保留原 MolmoSpaces episode 字段，并增加：

```text
interactive_nav
  schema_version
  benchmark_type
  case_id
  case_type
  parent_benchmark_episode_index
  door_state
  oracle
  paths
  diagnostics
  sampling
  plot_path
```

该格式的 `oracle.required_open_doors` 是门 ID 列表，不是 v3 的 typed `oracle_plan.steps`。它也没有 v3 的统一 `interactions`、`success_criteria` 和 articulation joint state 表达。
