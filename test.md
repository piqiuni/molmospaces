- [mlspaces](#mlspaces)
  - [数据流](#数据流)
  - [数据资产下载](#数据资产下载)
  - [抓取场景](#抓取场景)
    - [场景排序测试](#场景排序测试)
- [导航](#导航)
  - [仿真开启](#仿真开启)
    - [调试机械臂位置](#调试机械臂位置)
    - [Log 位置](#log-位置)
  - [可视化](#可视化)
  - [探索包](#探索包)
    - [一键启动](#一键启动)
    - [占据地图](#占据地图)
      - [配置文件](#配置文件)
    - [探索策略包](#探索策略包)
      - [配置文件](#配置文件-1)
    - [导航控制](#导航控制)
    - [地图/策略统一清空重置](#地图策略统一清空重置)
  - [检测包](#检测包)
    - [](#)
    - [检测测试](#检测测试)
    - [GT 测试](#gt-测试)



# mlspaces
## 数据流

conda activate mlspaces

python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1

python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1 --scene_dataset procthor-10k



## 数据资产下载
export MLSPACES_CACHE_DIR=/home/user/.cache/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/home/user/ldl/molmospaces/assets

python -c "from molmo_spaces.molmo_spaces_constants import DATA_CACHE_DIR, ASSETS_DIR; print(DATA_CACHE_DIR); print(ASSETS_DIR)"


## 抓取场景

source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlspaces && for i in $(seq 0 100); do python scripts/datagen/fetch_assets.py scene procthor-10k $i --split train --variant ceiling || true; done

### 场景排序测试
python scripts/datagen/rank_nav_scenes.py --scene_dataset procthor-10k --data_split train --save_maps


# 导航
## 仿真开启
python scripts/InteractiveNav/run_nav_ros_sim.py \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --target_types Apple \
  --task_horizon 3000


测试耗时
python scripts/datagen/run_nav_ros_sim.py \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --target_types Apple
  --timing_log_every_n_frames


###  调试机械臂位置

source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlspaces && \
python scripts/datagen/run_nav_ros_sim.py \
  --viewer \
  --policy_mode left_arm_debug \
  --left_arm_joint_delta 0.05 \
  --debug_loop_episodes 50 \
  --robot rby1 \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 0 \
  --target_types Apple \
  --task_horizon 2000


### Log 位置
/home/user/ldl/molmospaces/assets/datagen/nav_to_obj_ros_sim_v1


## 可视化
rviz -d ldl/molmospaces/nav_rviz.rviz

## 探索包


### 一键启动
conda activate mlspaces
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg molmospaces_nav_system.launch

source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch semantic_mapping_py_pkg semantic_mapping_debug.launch

### 占据地图
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch struct_mapping_pkg slam_gmapping.launch

#### 配置文件
Interactive-Nav-SG-nav/src/struct_mapping_pkg/config/slam_gmapping_params.yaml

### 探索策略包
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch explore_pkg explore_manager.launch

#### 配置文件
Interactive-Nav-SG-nav/src/explore_pkg/config/exploration_planner_params.yaml


### 导航控制
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg nav.launch


### 地图/策略统一清空重置
source ./Interactive-Nav-SG-nav/devel/setup.zsh
rostopic pub -1 /nav_system/reset std_msgs/Empty "{}"




## 检测包
### 

### 检测测试
conda activate yolo_world
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch semantic_mapping_py_pkg object_detection_visual_test.launch \
  rgb_path:=/home/user/ldl/molmospaces/detection_models/tum_rgbd_scribble_samples/kitchen_22_image.png \
  depth_path:=/home/user/ldl/molmospaces/detection_models/tum_rgbd_scribble_samples/kitchen_22_depth.png \
  backend:=yoloe_pf_box3d \
  provider:=yoloe_local \
  model_path:=/home/user/ldl/molmospaces/detection_models/yoloe/weights/yoloe-26x-seg-pf.pt



### GT 测试
保存文件到/home/user/ldl/molmospaces/scripts/InteractiveNav/output
python scripts/InteractiveNav/read_scene_room_properties.py \
  --scene_dataset procthor-10k \
  --data_split train \
  --house_ind 1 \
  --variant base

python Interactive-Nav-SG-nav/src/semantic_mapping_py_pkg/scripts/semantic_mapping_gt_replay.py \
  /home/user/ldl/molmospaces/scripts/InteractiveNav/output/procthor-10k_train_1_base_scene_full.json \
  --batch-size 4 \
  --publish-rate 1.0
