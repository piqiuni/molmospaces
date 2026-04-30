# Molmo Spaces 导航开发阶段手册

本手册用于维护 `nav_to_obj` 与 ROS 适配相关的开发流程。内容基于 `test.md` 和近期联调结论整理，按阶段执行可减少环境与资源问题。

## 阶段 0：环境与路径配置

目标：确保资源目录和运行环境正确，避免路径冲突与缺资源。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlspaces

export MLSPACES_CACHE_DIR=/home/user/.cache/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/home/user/ldl/molmospaces/assets

python -c "from molmo_spaces.molmo_spaces_constants import DATA_CACHE_DIR, ASSETS_DIR; print(DATA_CACHE_DIR); print(ASSETS_DIR)"
```

说明：
- `MLSPACES_CACHE_DIR` 和 `MLSPACES_ASSETS_DIR` 不能是同一路径。
- `export` 只对当前终端生效，长期生效需写入 `~/.zshrc`。

## 阶段 1：快速单次管线验证（run_pipeline）

目标：先验证 `nav_to_obj` 主流程可跑通。

```bash
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --scene_dataset ithor --house_inds 1 --samples_per_house 1
```

可视化版本：

```bash
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --scene_dataset ithor --house_inds 1 --samples_per_house 1 --viewer
```

默认环境版本（不显式传 `scene_dataset`）：

```bash
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1
```

## 阶段 2：使用 ProcTHOR 场景进行导航生成

目标：切换到 `procthor-10k` 做真实批量场景测试。

```bash
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --scene_dataset procthor-10k --house_inds 1 --samples_per_house 1
```

批量（每场景 1 次，前 100 个场景）：

```bash
for i in $(seq 1 100); do
  python scripts/datagen/run_pipeline.py \
    --task_type nav_to_obj \
    --policy planner \
    --robot rby1 \
    --scene_dataset procthor-10k \
    --house_inds $i \
    --samples_per_house 1 \
    --seed 2 \
    --run_name_prefix nav100 || true
done
```

## 阶段 3：按需抓取场景资源（避免缺文件）

目标：按测试区间预热资源，减少 `missing scene file`。

```bash
for i in $(seq 0 100); do
  python scripts/datagen/fetch_assets.py scene procthor-10k $i --split train --variant ceiling || true
done
```

## 阶段 4：场景筛选与“大场景”挑选

目标：选择更大、更适合传统导航评测的固定场景集合。

生成面积排序 CSV：

```bash
python scripts/datagen/rank_nav_scenes.py --scene_dataset procthor-10k --data_split train
```

同时导出 top-down 预览图：

```bash
python scripts/datagen/rank_nav_scenes.py --scene_dataset procthor-10k --data_split train --save_maps
```

输出：
- `assets/scene_rankings/procthor-10k_train_ranking.csv`
- `assets/scene_rankings/house_*_map.png`（开启 `--save_maps` 时）

## 阶段 5：ROS 仿真桥接验证

目标：运行 ROS 收发 policy，发布观测并接收动作。

```bash
python scripts/datagen/run_nav_ros_sim.py \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 0 \
  --target_types Apple \
  --samples_per_house 1 \
  --observation_topic /molmo_spaces/observation \
  --action_topic /molmo_spaces/action
```

查看图像观测 topic（当前 policy 已改为发布 `sensor_msgs/Image`）：

```bash
rostopic hz /molmo_spaces/observation
rostopic echo /molmo_spaces/observation/header
```

## 阶段 6：传统导航框架适配建议

目标：固定测试集，做可复现实验对比，而不是每次随机。

建议策略：
- 固定场景：使用 `rank_nav_scenes.py` 结果选一批 `house_inds`。
- 固定随机性：固定 `--seed`，关闭随机化开关。
- 固定目标类型：使用 `--target_types`。
- 固定每场景采样次数：`--samples_per_house 1`。

推荐执行顺序：
1. 先批量 `fetch_assets`（目标 house 区间）。
2. 再跑 `rank_nav_scenes` 选 Top-N 大场景。
3. 最后在固定场景列表上批量跑 `run_pipeline` 或 `run_nav_ros_sim`。

## 附：`test.md` 里的常用命令归档

基础生成：

```bash
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1
python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1 --scene_dataset procthor-10k
```

抓取场景：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlspaces && for i in $(seq 0 100); do python scripts/datagen/fetch_assets.py scene procthor-10k $i --split train --variant ceiling || true; done
```

场景排序：

```bash
python scripts/datagen/rank_nav_scenes.py --scene_dataset procthor-10k --data_split train --save_maps
```

ROS 导航仿真：

```bash
python scripts/datagen/run_nav_ros_sim.py \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 0 \
  --target_types Apple
```
