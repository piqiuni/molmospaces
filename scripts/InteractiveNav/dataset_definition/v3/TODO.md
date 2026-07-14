# Interactive Navigation v3 TODO

本文件记录已经确认需要后续设计、但暂不进入当前 v3 Schema 的内容。

## 双开门与联合动作

当前一个 articulation joint 对应一个 interaction，一个 oracle action 对应一个 `open_joint` step。暂不定义以下语义：

- 双开门冰箱的左右门是否组成一个逻辑 interaction group。
- 一个目标空间是否要求多个 joint 全部打开。
- 多个 joint 是 `all`、`any` 还是固定顺序执行。
- 多 joint 联合操作是否应作为一个原子高层动作计费。
- root door、leaf articulation object 和 interaction group 的稳定 ID 关系。

后续候选设计：

```json
{
  "interaction_group_id": "fridge_main_compartment",
  "member_interaction_ids": ["left_door", "right_door"],
  "execution_mode": "all"
}
```

候选 `execution_mode`：

```text
all
any
ordered
```

在加入正式 Schema 前，需要先用真实双开门冰箱验证：

- 单开左门、单开右门和双门全开分别对目标可见性和可访问性的影响。
- 两个 leaf joint 是否能独立控制和回放。
- 一个 joint 是否足以满足 NavToObj 成功条件。
- 联合动作的 oracle cost 和 policy action 定义。

当前 builder 应继续把每个 leaf joint 保存为独立 interaction，并通过 oracle step 顺序表达实际执行过程，不得声称它们已经是一个原子联合动作。
