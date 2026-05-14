
## 数据流

conda activate mlspaces

python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1

python scripts/datagen/run_pipeline.py --task_type nav_to_obj --policy planner --robot rby1 --house_inds 1 --samples_per_house 1 --scene_dataset procthor-10k



## 下载
export MLSPACES_CACHE_DIR=/home/user/.cache/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/home/user/ldl/molmospaces/assets

python -c "from molmo_spaces.molmo_spaces_constants import DATA_CACHE_DIR, ASSETS_DIR; print(DATA_CACHE_DIR); print(ASSETS_DIR)"


## 抓取场景

source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlspaces && for i in $(seq 0 100); do python scripts/datagen/fetch_assets.py scene procthor-10k $i --split train --variant ceiling || true; done

### 排序测试
python scripts/datagen/rank_nav_scenes.py --scene_dataset procthor-10k --data_split train --save_maps


## ROS
python scripts/datagen/run_nav_ros_sim.py \
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


## 调试机械臂位置

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

## 可视化
rviz -d ldl/molmospaces/nav_rviz.rviz

## 探索包
### 占据地图

source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch struct_mapping_pkg slam_gmapping.launch

#### 配置文件
Interactive-Nav-SG-nav/src/struct_mapping_pkg/config/slam_gmapping_params.yaml

### 探索包
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch explore_pkg explore_manager.launch

#### 配置文件
Interactive-Nav-SG-nav/src/explore_pkg/config/exploration_planner_params.yaml

### 统一清空重置
source ./Interactive-Nav-SG-nav/devel/setup.zsh
rostopic pub -1 /nav_system/reset std_msgs/Empty "{}"


### 导航控制
source ./Interactive-Nav-SG-nav/devel/setup.zsh
roslaunch nav_pkg nav.launch



#### Log 位置
/home/user/ldl/molmospaces/assets/datagen/nav_to_obj_ros_sim_v1