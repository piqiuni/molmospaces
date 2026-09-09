# Interactive Navigation Dataset Definitions

本目录集中保存交互导航 benchmark 的历史 JSON 格式和当前统一定义。

```text
dataset_definition/
  v1/  历史 door_interaction_nav_v1 数据
  v2/  历史 interactive_nav_v2 容器数据
  v3/  当前统一的通道、容器和混合交互格式定义
```

## 版本关系

`v1` 和 `v2` 不是同一个通用 schema 的严格递进版本：

- `v1` 是通道门 benchmark 的历史格式，schema version 为 `door_interaction_nav_v1`。
- `v2` 是容器 benchmark 的历史格式，schema version 为 `interactive_nav_v2`。
- `v3` 才是第一次尝试统一通道、容器、混合交互和 Instruction 的公共格式。

因此不能通过修改 `schema_version` 直接把 v1/v2 数据视为 v3。后续需要显式迁移 interaction、oracle step、initial articulation state、success criteria 和 generation validation 字段。

## 当前状态

| 目录 | 内容 | 数据来源 | 是否真实采集 |
|---|---|---|---|
| `v1/` | 34 条历史 door episode | 旧 door builder 输出的直接副本 | 是 |
| `v2/` | 29 条历史 container episode | 旧 container builder 输出的直接副本 | 是 |
| `v3/` | Schema、格式文档和 4 个结构示例 | 手工构造的格式示例 | 否 |

v1/v2 的归档操作没有重新运行仿真。v3 示例也没有执行真实场景加载、门/容器几何判断、路径规划或可见性验证。
