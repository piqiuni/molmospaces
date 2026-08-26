#!/bin/bash
# 完整系统启动脚本 - AI2-THOR + struct_mapping_pkg 建图系统

echo "========================================="
echo " 🚀 启动完整建图系统"
echo "========================================="
echo ""

cd /home/wxy/Experiment/Interactive-nav/interactive_nav-main

# 检查 roscore
if ! pgrep -x "rosmaster" > /dev/null; then
    echo "⚠️  ROS Master 未运行，正在启动..."
    gnome-terminal --title="ROS Master" -- bash -c "source /opt/ros/noetic/setup.bash && roscore; exec bash" &
    sleep 3
else
    echo "✓ ROS Master 已运行"
fi

echo ""
echo "正在启动各个组件..."
echo ""

# 1. 启动 AI2-THOR 仿真器
echo "1️⃣  启动 AI2-THOR 仿真器..."
gnome-terminal --title="AI2-THOR 仿真器" --geometry=80x24+0+0 -- bash -c "
    cd /home/wxy/Experiment/Interactive-nav/interactive_nav-main
    source /opt/ros/noetic/setup.bash
    source devel/setup.bash
    export CONDA_BASE=\$(conda info --base)
    source \$CONDA_BASE/etc/profile.d/conda.sh
    conda activate smartllm
    echo '🏠 启动 AI2-THOR 仿真器...'
    echo ''
    roslaunch ai2thor_pkg ai2thor_sim.launch
    exec bash
" &
sleep 5

# 2. 启动 struct_mapping_pkg 建图节点
echo "2️⃣  启动 GMapping 建图节点..."
gnome-terminal --title="GMapping 建图" --geometry=80x24+800+0 -- bash -c "
    cd /home/wxy/Experiment/Interactive-nav/interactive_nav-main
    source /opt/ros/noetic/setup.bash
    source devel/setup.bash
    echo '🗺️  启动 GMapping 建图节点...'
    echo ''
    roslaunch struct_mapping_pkg slam_gmapping_pr2.launch
    exec bash
" &
sleep 3

# 3. 启动键盘控制器
echo "3️⃣  启动键盘控制器..."
gnome-terminal --title="🎮 键盘控制器" --geometry=80x24+0+400 -- bash -c "
    cd /home/wxy/Experiment/Interactive-nav/interactive_nav-main
    source /opt/ros/noetic/setup.bash
    source devel/setup.bash
    export CONDA_BASE=\$(conda info --base)
    source \$CONDA_BASE/etc/profile.d/conda.sh
    conda activate smartllm
    echo '🎮 启动键盘控制器...'
    echo ''
    echo '操作说明:'
    echo '  W - 前进'
    echo '  S - 后退'
    echo '  A - 左转'
    echo '  D - 右转'
    echo '  Q - 退出'
    echo ''
    python3 src/ai2thor_pkg/script/keyboard_control.py
    exec bash
" &
sleep 2

# 4. 启动 RViz 可视化
echo "4️⃣  启动 RViz 可视化..."
gnome-terminal --title="RViz 可视化" --geometry=80x24+800+400 -- bash -c "
    cd /home/wxy/Experiment/Interactive-nav/interactive_nav-main
    source /opt/ros/noetic/setup.bash
    source devel/setup.bash
    echo '👁️  启动 RViz 可视化...'
    echo ''
    echo '配置说明:'
    echo '  固定坐标系: tf_frame_map'
    echo '  添加显示:'
    echo '    - PointCloud2: /local_scan (原始点云)'
    echo '    - PointCloud2: /filtered_pointcloud (过滤后点云)'
    echo '    - OccupancyGrid: /grid_mapping/occ_map1 (2D地图)'
    echo '    - TF (坐标系)'
    echo ''
    sleep 3
    rviz
    exec bash
" &

echo ""
echo "========================================="
echo " ✅ 所有组件启动完成！"
echo "========================================="
echo ""
echo "📋 已启动的组件："
echo "  1. AI2-THOR 仿真器"
echo "  2. GMapping 建图节点"
echo "  3. 键盘控制器"
echo "  4. RViz 可视化"
echo ""
echo "🎮 使用键盘控制机器人移动进行建图"
echo ""
echo "📊 监控命令（新终端）："
echo "  rostopic hz /local_scan"
echo "  rostopic hz /filtered_pointcloud"
echo "  rostopic hz /grid_mapping/occ_map1"
echo "  rosrun tf view_frames && evince frames.pdf"
echo ""
echo "💾 保存地图："
echo "  rosrun map_server map_saver -f my_map map:=/grid_mapping/occ_map1"
echo ""
echo "按 Ctrl+C 停止此脚本（不会关闭已启动的窗口）"
echo ""

# 保持脚本运行
wait












