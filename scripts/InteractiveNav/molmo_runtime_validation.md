# MolmoSpaces 运行验证记录

本文记录 `scripts/InteractiveNav/explore_molmo_interactions.py` 在本机上的真实运行结果，用于区分：

- 代码逻辑是否可达
- 环境配置是否正确
- 外部资源/网络是否成为 blocker

## 1. 已确认可运行的轻量验证

以下命令不依赖真实 scene 资源加载，已成功运行：

```bash
python -m py_compile scripts/InteractiveNav/explore_molmo_interactions.py

python scripts/InteractiveNav/explore_molmo_interactions.py --help

python scripts/InteractiveNav/explore_molmo_interactions.py task-config-template --task-kind nav_to_obj

python scripts/InteractiveNav/explore_molmo_interactions.py action-schema --mode container_oracle

python scripts/InteractiveNav/explore_molmo_interactions.py benchmark-episode-template --task-kind nav_to_obj

python scripts/InteractiveNav/explore_molmo_interactions.py integration-recipe --mode door_oracle_nav_loop

python scripts/InteractiveNav/explore_molmo_interactions.py env-check
```

结论：

- 脚本语法正确
- CLI 子命令注册正常
- 不依赖 MuJoCo 场景加载的模板/说明类子命令可以正常使用

## 2. 已确认的真实 scene 运行尝试

### 2.1 直接使用 `mlspaces` 环境运行

命令：

```bash
/home/user/miniconda3/envs/mlspaces/bin/python \
  scripts/InteractiveNav/explore_molmo_interactions.py inspect-scene \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1
```

结果：

- 成功进入 `mlspaces` Python 环境
- 失败原因不是脚本语法，而是资源缓存目录不可写

报错关键信息：

- `Read-only file system: '/home/user/.cache/molmo-spaces-resources/.lock'`

结论：

- 默认 `MLSPACES_CACHE_DIR` 在当前环境下不可写

### 2.2 改用可写缓存目录 `/tmp/molmo-spaces-resources`

命令：

```bash
MLSPACES_CACHE_DIR=/tmp/molmo-spaces-resources \
  /home/user/miniconda3/envs/mlspaces/bin/python \
  scripts/InteractiveNav/explore_molmo_interactions.py inspect-scene \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1
```

结果：

- 绕过了只读缓存目录问题
- 但遇到了已有 cache 与 manifest 不一致的残留状态

报错关键信息：

- `Directory path exists on disk but is not recorded in the cache manifest`
- 路径：`/tmp/molmo-spaces-resources/robots/rby1/20251224`

结论：

- `/tmp/molmo-spaces-resources` 不是一个干净 cache
- 这不是脚本逻辑错误，而是本地资源缓存状态问题

### 2.3 改用全新缓存目录 `/tmp/molmo-spaces-resources-codex-fresh`

命令：

```bash
MLSPACES_CACHE_DIR=/tmp/molmo-spaces-resources-codex-fresh \
  /home/user/miniconda3/envs/mlspaces/bin/python \
  scripts/InteractiveNav/explore_molmo_interactions.py inspect-scene \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1
```

结果：

- 绕过了只读问题
- 绕过了脏 cache 问题
- 最终明确卡在远端 manifest / asset 下载

报错关键信息：

- `MolmoSpaces resource manager attempted to fetch remote manifests/assets but network access is unavailable`

底层对应现象：

- `requests.exceptions.ConnectionError`
- 访问 `*.r2.dev` 失败

结论：

- 当前环境中，真实 scene-loading 至少还需要：
  - 可写 cache 目录
  - 预缓存的本地资源，或可用网络

## 3. 当前可以下的结论

### 已被真实证据支持的部分

1. 脚本本身可执行
2. 非 scene-loading 子命令可正常工作
3. `mlspaces` conda 环境存在
4. 真实 scene-loading 的首要环境约束已被定位

## 3.1 新进展：离线 proxy cache + EGL 已打通真实 scene

在进一步检查后，发现本机默认 cache

- `/home/user/.cache/molmo-spaces-resources`

其实已经包含：

- `robots/rby1/20251224`
- `scenes/procthor-10k-train/20251122`
- `objects/thor/20251117`

因此不必依赖联网重新下载。可行做法是：

1. 在 `/tmp` 创建一个可写 proxy cache 根目录
2. 复制 `mjthor_data_type_to_source_to_versions.json`
3. 把现有 `robots / scenes / objects / grasps / benchmarks / test_data` 目录软链进去
4. 运行时设置：

```bash
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
MLSPACES_CACHE_DIR=/tmp/molmo-spaces-cache-proxy
MLSPACES_ASSETS_DIR=/tmp/molmo-spaces-assets-proxy
```

### `inspect-scene` 实跑结果

已成功运行：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
MLSPACES_CACHE_DIR=/tmp/molmo-spaces-cache-proxy \
MLSPACES_ASSETS_DIR=/tmp/molmo-spaces-assets-proxy \
/home/user/miniconda3/envs/mlspaces/bin/python \
scripts/InteractiveNav/explore_molmo_interactions.py inspect-scene \
  --house_ind 10 \
  --output_json scripts/InteractiveNav/output/inspect-scene_procthor-10k_train_10.json
```

得到的真实结果包括：

- scene: `train_10_ceiling.xml`
- 4 个 door body name
- 10 个 articulated object
- 至少 1 个 light
- articulation 类别已覆盖：
  - `Box`
  - `Dresser`
  - `Fridge`
  - `Safe`
  - `Laptop`
  - `Toilet`

### `nav-gt` 实跑结果

已成功运行到：

- scene 加载
- occupancy map 生成
- `nav_to_obj` task 采样
- target object 选定

在 `train_10` 上，当前失败点不是 scene-loading，而是：

- `NavGoalSampler` 10 次都未采到合适的 navigation goal

对应目标对象：

- `safe_5ea0563319e8e09ddd8f7b0099388eb0_1_0_5`

脚本现已更新为：

- 遇到这类失败时返回结构化字段 `nav_goal_sampling_error`
- 不再让整个 `nav-gt` / `door-path-study` 命令直接崩掉

### 尚未被真实证据支持的部分

1. `nav-gt` 在合适 house 上成功生成 live GT path
2. `door-path-study` 成功完成 door override + replan
3. candidate house 筛选结果

## 4. 推荐后续动作

### 如果继续在当前环境推进

优先级：

1. 使用可写 cache 目录
2. 优先复用本机现有资源缓存，必要时在 `/tmp` 建 proxy cache
3. 显式设置 `MUJOCO_GL=egl` 与 `PYOPENGL_PLATFORM=egl`
4. 再次运行：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
MLSPACES_CACHE_DIR=/tmp/molmo-spaces-cache-proxy \
MLSPACES_ASSETS_DIR=/tmp/molmo-spaces-assets-proxy \
  /home/user/miniconda3/envs/mlspaces/bin/python \
  scripts/InteractiveNav/explore_molmo_interactions.py inspect-scene \
  --house_ind 10
```

### 如果只做代码/接口层工作

当前已经具备：

- GT path 接口梳理
- door/container/light 状态接口梳理
- task config 模板
- benchmark episode 模板
- action schema
- integration recipe
- environment self-check
- objective status 审计
