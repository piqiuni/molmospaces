#!/bin/bash
# 启动BLIP服务器脚本
# 建立SSH隧道并在远程服务器上启动BLIP服务

# 方式1: 建立SSH隧道（后台运行），然后在远程服务器上执行命令
gnome-terminal --title="BLIP SSH Tunnel" -- bash -c "ssh -L 5000:localhost:5000 ycs@10.106.11.248 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate blip && cd ~/blip_server && bash start_server.sh'; exec bash"

# 或者方式2: 分离SSH隧道和服务器启动（如果需要分别管理）
# gnome-terminal --title="BLIP SSH Tunnel" -- bash -c "ssh -f -N -L 5000:localhost:5000 ycs@10.106.11.248; exec bash"
# sleep 2
# gnome-terminal --title="BLIP Server" -- bash -c "ssh ycs@10.106.11.248 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate blip && cd ~/blip_server && bash start_server.sh'; exec bash"
