#!/usr/bin/env python3
"""
BLIP客户端使用示例
"""
# ssh -L 5000:localhost:5000 ycs@10.106.11.248
from blip_client import BlipClient

# 初始化客户端（只需设置一次服务器地址）
server_url = "http://127.0.0.1:5000"  # 替换为你的服务器IP
client = BlipClient(server_url, timeout=60, connect_timeout=5)

# 先检查服务器健康状态
print(f"检查服务器状态: {server_url}")
health = client.check_health()
if health.get("status") != "healthy":
    print(f"警告: 服务器不可用 - {health.get('error', '未知错误')}")
    print("请确保服务器正在运行，或检查网络连接")
    exit(1)
print("服务器状态正常\n")

# 视觉问答
print("执行视觉问答...")
result = client.visual_qa(
    "/home/ycs/robot_ws/semantic_ws/src/semantic_mapping_pkg/test/dinning.jpg", 
    "what is in the image?"
)
if result.get("success"):
    print(f"答案: {result.get('answer', 'N/A')}")
else:
    print(f"错误: {result.get('error', '未知错误')}")

print()

# 图像描述
print("生成图像描述...")
result = client.image_caption(
    "/home/ycs/robot_ws/semantic_ws/src/semantic_mapping_pkg/test/dinning.jpg"
)
if result.get("success"):
    print(f"描述: {result.get('caption', 'N/A')}")
else:
    print(f"错误: {result.get('error', '未知错误')}")

