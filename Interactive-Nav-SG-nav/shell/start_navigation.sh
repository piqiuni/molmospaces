#!/bin/bash
# AI2-THOR导航系统一键启动脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  AI2-THOR 语义导航系统启动脚本${NC}"
echo -e "${GREEN}=====================================${NC}"

# 工作空间路径
WS_PATH="/home/wxy/Experiment/Interactive-nav/interactive_nav-main"
ENV_PATH="$HOME/miniconda3/bin/activate"

# 场景配置
SCENE_TYPE="procthor"  # procthor 或 floorplan
PROCTHOR_INDEX=120
SEMANTIC_TARGET="Microwave"

# 检查conda环境
if [ ! -f "$ENV_PATH" ]; then
    echo -e "${RED}错误: 找不到conda环境${NC}"
    echo "请安装miniconda并创建ai2thor环境"
    exit 1
fi

# 进入工作空间
cd $WS_PATH || exit

# 检查是否已编译
if [ ! -d "devel" ]; then
    echo -e "${YELLOW}首次运行，正在编译工作空间...${NC}"
    catkin_make
fi

# Source工作空间
source ./devel/setup.bash

echo -e "${GREEN}启动配置:${NC}"
echo "  场景类型: $SCENE_TYPE"
if [ "$SCENE_TYPE" == "procthor" ]; then
    echo "  ProcTHOR索引: $PROCTHOR_INDEX"
fi
echo "  语义目标: $SEMANTIC_TARGET"
echo ""

# 启动AI2-THOR仿真器（新终端）
echo -e "${GREEN}[1/2] 启动AI2-THOR仿真器...${NC}"
gnome-terminal -- bash -c "
    source $ENV_PATH smartllm
    cd $WS_PATH
    source ./devel/setup.bash
    roslaunch ai2thor_pkg ai2thor_sim.launch \
        scene_type:=$SCENE_TYPE \
        procthor_index:=$PROCTHOR_INDEX \
        semantic_target_type:=$SEMANTIC_TARGET
"

# 等待仿真器启动
sleep 3

# 启动占据地图构建（新终端）
echo -e "${GREEN}[2/2] 启动占据地图构建...${NC}"
gnome-terminal -- bash -c "
    source $ENV_PATH smartllm
    cd $WS_PATH
    source ./devel/setup.bash
    roslaunch grid_map_pkg grid_mapping.launch
"

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}系统启动完成！${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo "可视化工具:"
echo "  - RViz: rosrun rviz rviz"
echo "  - 查看话题: rostopic list"
echo "  - 查看地图: rostopic echo /grid_mapping/occ_map"
echo ""
echo "按 [Enter] 键停止所有服务..."
read

# 清理
echo -e "${YELLOW}正在停止服务...${NC}"
killall gnome-terminal
killall roslaunch
killall roscore

echo -e "${GREEN}已停止所有服务${NC}"

