## 
scripts/InteractiveNav/read_scene_room_properties.py

```
conda activate mlspaces

python scripts/InteractiveNav/read_scene_room_properties.py \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 0 \
  --variant base \
  --room 2

python scripts/InteractiveNav/read_scene_room_properties.py --house_ind 1

python scripts/InteractiveNav/read_scene_room_properties.py --room 2 --background_mode bounds

```

##
scripts/InteractiveNav/explore_molmo_interactions.py

```bash
conda activate mlspaces

# 1) 枚举场景中的 door / drawer / cabinet / fridge / lights
python scripts/InteractiveNav/explore_molmo_interactions.py inspect-scene \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1

# 2) 基于 nav_to_obj 采样结果读取 GT path
python scripts/InteractiveNav/explore_molmo_interactions.py nav-gt \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --target_types Apple

# 3) 固定起终点后，优先关闭 baseline path 附近的 1 个 door 并重算路径
python scripts/InteractiveNav/explore_molmo_interactions.py door-path-study \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --target_types Apple \
  --close_doors_on_path 1 \
  --study_state closed

# 会同时输出：
# - <output_json stem>_baseline.png
# - <output_json stem>_compare.png
# 图中包含 top-down occupancy 背景、door 位置、start/goal，以及可用时的 baseline / changed GT path

# 4) 直接设置容器关节开合比例（例如 drawer/cabinet/fridge）
python scripts/InteractiveNav/explore_molmo_interactions.py set-articulation \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --object-name <ARTICULATED_OBJECT_NAME> \
  --joint-index 0 \
  --open-fraction 0.0

# 5) 导出固定 task 配置模板
python scripts/InteractiveNav/explore_molmo_interactions.py task-config-template \
  --task-kind nav_to_obj

# 6) 导出 step action schema / oracle 调用模板
python scripts/InteractiveNav/explore_molmo_interactions.py action-schema \
  --mode container_oracle

# 7) 导出 benchmark EpisodeSpec 级别 JSON 骨架
python scripts/InteractiveNav/explore_molmo_interactions.py benchmark-episode-template \
  --task-kind nav_to_obj

# 8) 导出导航如何调用 oracle / planner 交互的接线伪代码
python scripts/InteractiveNav/explore_molmo_interactions.py integration-recipe \
  --mode door_oracle_nav_loop

# 9) 检查当前环境是否具备实跑 scene 的基本条件
python scripts/InteractiveNav/explore_molmo_interactions.py env-check
```

##
scripts/InteractiveNav/molmo_interaction_interfaces.md

- 汇总 `nav_to_obj` GT path、door/container/light 控制、task config 固定入口、step action dict 接口
- 建议在继续做 `door-path-study`、`oracle interaction` 或 TODO Phase 4 扩展前先通读一遍

##
scripts/InteractiveNav/molmo_gt_workflow.md

- 按 `2.1 -> 3.2` 的目标，把 GT path、关门重算、固定 episode、oracle/planner action 串成工作流
- 更适合直接照着做实验，而不是只查字段定义

##
scripts/InteractiveNav/molmo_objective_status.md

- 把 `1 / 2.1 / 2.2 / 2.3 / 3.1 / 3.2 / 4 / 5` 逐项映射到现有脚本、文档、验证状态与剩余缺口
- 适合用来做完成度审计

##
scripts/InteractiveNav/molmo_runtime_validation.md

- 记录真实在 `mlspaces` 环境中的运行结果
- 已明确验证出：只读 cache、脏 cache、无网络三类环境 blocker

## 运行排障

- 若提示 `mujoco is not available`：
  - 先进入 `mlspaces` 环境
- 若提示 `Read-only file system` 且路径包含 `molmo-spaces-resources/.lock`：
  - 改用 `MLSPACES_CACHE_DIR=/tmp/molmo-spaces-resources`
- 若本机已有 `~/.cache/molmo-spaces-resources`，但当前线程对默认 cache 不可写：
  - 可以在 `/tmp` 建一个可写 proxy cache，把现有 `robots / scenes / objects / grasps / benchmarks / test_data` 目录软链进去
- 若提示访问 `*.r2.dev` 失败或 `ConnectionError / Max retries exceeded`：
  - 说明 `molmospaces_resources` 试图联网拉取远端 manifest / asset
  - 在当前无网络环境下，真实 scene-loading 命令需要预先缓存资源，或允许联网
- 在 Linux headless 环境中，建议显式设置：
  - `MUJOCO_GL=egl`
  - `PYOPENGL_PLATFORM=egl`

推荐运行前缀：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  MLSPACES_CACHE_DIR=/tmp/molmo-spaces-resources \
  /home/user/miniconda3/envs/mlspaces/bin/python \
  scripts/InteractiveNav/explore_molmo_interactions.py <subcommand> ...
```

已确认的真实运行结果：

- `inspect-scene` 已可在 `train_10_ceiling.xml` 上成功输出 articulation 与 light 列表
- `nav-gt` 已可跑到真实 task / occupancy-map / target sampling 阶段
- 某些 house 目前会在 `NavGoalSampler` 采样导航终点时失败；脚本现已将其改成结构化结果返回，而不是直接崩溃
