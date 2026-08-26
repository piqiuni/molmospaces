import os
import sys
import gzip
import json
from pathlib import Path

# 设置ProcTHOR-10K数据集本地目录
LOCAL_DATASET_DIR = "/home/wxy/Experiment/Interactive-nav/datasets/procthor-10k-data"
DATASET_CACHE_DIR = "/home/wxy/Experiment/Interactive-nav/datasets/procthor_cache"
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

# 导入prior和其他库
import prior
try:
    from prior import LazyJsonDataset
except ImportError:
    print("警告: 无法导入LazyJsonDataset，请更新prior库: pip install --upgrade prior")
    LazyJsonDataset = None

from tqdm import tqdm
import ai2thor.controller
import time
import random
import cv2
import threading
from datetime import datetime

def load_dataset_from_local(data_dir: str) -> prior.DatasetDict:
    """
    从本地目录直接加载数据集，不依赖Git LFS
    
    Args:
        data_dir: 包含train.jsonl.gz, val.jsonl.gz, test.jsonl.gz的目录
    
    Returns:
        DatasetDict包含train, val, test splits
    """
    if LazyJsonDataset is None:
        raise ImportError("需要更新prior库: pip install --upgrade prior")
    
    data = {}
    for split, size in [("train", 10_000), ("val", 1_000), ("test", 1_000)]:
        jsonl_file = os.path.join(data_dir, f"{split}.jsonl.gz")
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"未找到数据文件: {jsonl_file}")
        
        print(f"正在从本地加载 {split} split...")
        with gzip.open(jsonl_file, "rt") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        
        data[split] = LazyJsonDataset(
            data=houses, dataset="procthor-dataset", split=split
        )
        print(f"✓ {split} split加载完成: {len(houses)} 个场景")
    
    return prior.DatasetDict(**data)

# 加载ProcTHOR-10K数据集
print(f"加载ProcTHOR-10K数据集...")

# 优先使用本地数据文件
if os.path.exists(LOCAL_DATASET_DIR) and os.path.exists(os.path.join(LOCAL_DATASET_DIR, "train.jsonl.gz")):
    print(f"从本地目录加载数据集: {LOCAL_DATASET_DIR}")
    try:
        dataset = load_dataset_from_local(LOCAL_DATASET_DIR)
        print("✓ 成功从本地文件加载数据集")
    except Exception as e:
        print(f"⚠ 从本地加载失败: {e}")
        print("回退到使用prior库加载...")
        dataset = prior.load_dataset("procthor-10k")
else:
    print(f"本地数据文件不存在，使用prior库加载（会使用缓存）...")
    print(f"数据集缓存位置: ~/.prior/datasets/")
    dataset = prior.load_dataset("procthor-10k")

house = dataset["train"][120]  # 可修改
print(f"使用房屋: {house}")

# 创建控制器 - 启用RGBD传感器
c = ai2thor.controller.Controller(scene=house, height=600, width=800, renderDepthImage=True)
c.step(action="Pass")  # 初始化场景

# 初始化机器人
print("初始化机器人...")
event = c.step(action="GetMapViewCameraProperties")
event = c.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])

# 获取可达位置
reachable_positions_ = c.step(action="GetReachablePositions").metadata["actionReturn"]
reachable_positions = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]
print(f"找到 {len(reachable_positions)} 个可达位置")

# 随机放置机器人
init_pos = random.choice(reachable_positions_)
print(f"将机器人放置到位置: {init_pos}")
c.step(dict(action="Teleport", position=init_pos, agentId=0))

print("\n=== 操作说明 ===")
print("移动控制:")
print("  W - 前进  S - 后退  A - 左转  D - 右转")
print("  Q - 向上看  E - 向下看")
print("交互控制:")
print("  P - 拾取物体  U - 放下物体")
print("  O - 打开物体  C - 关闭物体")
print("  T - 开关物体  B - 破坏物体")
print("  L - 切片物体  Y - 投掷物体")
print("  R - 清洁物体  F - 填充液体")
print("  H - 推物体    G - 拉物体")
print("  K - 烹饪物体  N - 创建物体")
print("  I - 显示场景摘要  Z - 显示深度信息")
print("  X - 退出")
print("RGBD传感器:")
print("  - Agent RGB View: 显示机器人RGB视角")
print("  - Agent Depth View: 显示机器人深度视角(彩色可视化)")
print("  - Top View: 显示俯视图")
print("  - 所有图像和深度数据将自动保存到logs目录")
print("  - 程序结束时将自动生成RGB和深度视频")
print("================\n")

# 移动动作映射
movement_actions = {
    'w': lambda: c.step(action="MoveAhead", agentId=0),
    's': lambda: c.step(action="MoveBack", agentId=0),
    'a': lambda: c.step(action="RotateLeft", degrees=30, agentId=0),
    'd': lambda: c.step(action="RotateRight", degrees=30, agentId=0),
    'q': lambda: c.step(action="LookUp", degrees=30, agentId=0),
    'e': lambda: c.step(action="LookDown", degrees=30, agentId=0),
}

def get_interactive_properties(obj):
    """获取物体的交互属性"""
    properties = []
    
    # 基础交互属性
    if obj.get('pickupable', False):
        properties.append("可拾取")
    if obj.get('toggleable', False):
        properties.append("可开关")
    if obj.get('openable', False):
        properties.append("可开合")
    if obj.get('breakable', False):
        properties.append("可破坏")
    if obj.get('sliceable', False):
        properties.append("可切片")
    if obj.get('moveable', False):
        properties.append("可移动")
    if obj.get('receptacle', False):
        properties.append("可放置")
    if obj.get('isInteractable', False):
        properties.append("可交互")
    
    # 液体相关属性
    if obj.get('canFillWithLiquid', False):
        properties.append("可装液体")
    if obj.get('isFilledWithLiquid', False):
        properties.append("已装液体")
    
    # 清洁相关属性
    if obj.get('dirtyable', False):
        properties.append("可弄脏")
    if obj.get('isDirty', False):
        properties.append("已脏")
    
    # 使用相关属性
    if obj.get('canBeUsedUp', False):
        properties.append("可用完")
    if obj.get('isUsedUp', False):
        properties.append("已用完")
    
    # 烹饪相关属性
    if obj.get('cookable', False):
        properties.append("可烹饪")
    if obj.get('isCooked', False):
        properties.append("已烹饪")
    
    # 温度相关属性
    if obj.get('isHeatSource', False):
        properties.append("热源")
    if obj.get('isColdSource', False):
        properties.append("冷源")
    
    # 状态属性
    if obj.get('isToggled', False):
        properties.append("已开启")
    if obj.get('isBroken', False):
        properties.append("已破坏")
    if obj.get('isSliced', False):
        properties.append("已切片")
    if obj.get('isOpen', False):
        properties.append("已打开")
    if obj.get('isPickedUp', False):
        properties.append("已拾取")
    if obj.get('isMoving', False):
        properties.append("移动中")
    
    return properties

def print_visible_objects_with_interactions(event):
    """打印可见物体及其交互类型"""
    print("\n=== 当前视野可见物体 ===")
    visible_objects = []
    
    # 统计信息
    total_objects = len(event.metadata["objects"])
    visible_count = 0
    interactive_count = 0
    
    for obj in event.metadata["objects"]:
        if obj.get("visible", False):
            print(obj)
            visible_count += 1
            properties = get_interactive_properties(obj)
            visible_objects.append({
                'id': obj['objectId'],
                'name': obj.get('name', obj['objectId']),
                'properties': properties,
                'position': obj.get('position', {}),
                'distance': obj.get('distance', 0),
                'objectType': obj.get('objectType', ''),
                'assetId': obj.get('assetId', ''),
                # 存储原始属性用于交互检查
                'pickupable': obj.get('pickupable', False),
                'openable': obj.get('openable', False),
                'toggleable': obj.get('toggleable', False),
                'breakable': obj.get('breakable', False),
                'sliceable': obj.get('sliceable', False),
                'moveable': obj.get('moveable', False),
                'receptacle': obj.get('receptacle', False),
                'isInteractable': obj.get('isInteractable', False),
                # 液体相关
                'canFillWithLiquid': obj.get('canFillWithLiquid', False),
                'isFilledWithLiquid': obj.get('isFilledWithLiquid', False),
                # 清洁相关
                'dirtyable': obj.get('dirtyable', False),
                'isDirty': obj.get('isDirty', False),
                # 使用相关
                'canBeUsedUp': obj.get('canBeUsedUp', False),
                'isUsedUp': obj.get('isUsedUp', False),
                # 烹饪相关
                'cookable': obj.get('cookable', False),
                'isCooked': obj.get('isCooked', False),
                # 温度相关
                'isHeatSource': obj.get('isHeatSource', False),
                'isColdSource': obj.get('isColdSource', False),
                # 状态属性
                'isToggled': obj.get('isToggled', False),
                'isBroken': obj.get('isBroken', False),
                'isSliced': obj.get('isSliced', False),
                'isOpen': obj.get('isOpen', False),
                'isPickedUp': obj.get('isPickedUp', False),
                'isMoving': obj.get('isMoving', False),
                'openness': obj.get('openness', 0.0),
                'mass': obj.get('mass', 0.0),
                'temperature': obj.get('temperature', 'RoomTemp')
            })
    
    if not visible_objects:
        print("没有可见物体")
        print(f"场景中共有 {total_objects} 个物体，其中 {visible_count} 个可见")
        return []
    
    # 统计可交互物体
    interactive_objects = [obj for obj in visible_objects if obj['properties']]
    interactive_count = len(interactive_objects)
    
    print(f"场景中共有 {total_objects} 个物体，其中 {visible_count} 个可见，{interactive_count} 个可交互")
    
    # 按距离排序
    visible_objects.sort(key=lambda x: x['distance'])
    
    for i, obj in enumerate(visible_objects):
        print(f"{i+1}. {obj['name']} ({obj['id']})")
        print(f"   类型: {obj['objectType']} | 资产: {obj['assetId']}")
        print(f"   距离: {obj['distance']:.2f}m | 质量: {obj['mass']:.2f}kg | 温度: {obj['temperature']}")
        if obj['openable']:
            print(f"   开合状态: {obj['openness']:.1f} (0=关闭, 1=打开)")
        if obj['properties']:
            print(f"   交互类型: {', '.join(obj['properties'])}")
        else:
            print(f"   交互类型: 无")
        print()
    
    return visible_objects

def print_all_objects_summary(event):
    """打印场景中所有物体的摘要信息"""
    print("\n=== 场景物体摘要 ===")
    objects = event.metadata["objects"]
    
    # 按类型统计
    type_count = {}
    for obj in objects:
        obj_type = obj.get('objectType', 'Unknown')
        type_count[obj_type] = type_count.get(obj_type, 0) + 1
    
    print("物体类型统计:")
    for obj_type, count in sorted(type_count.items()):
        print(f"  {obj_type}: {count} 个")
    
    # 显示一些重要的物体类型
    important_types = ['Chair', 'Table', 'Cabinet', 'CounterTop', 'Fridge', 'Microwave', 'Sink', 'Stove', 'Window', 'Door']
    print("\n重要物体:")
    for obj_type in important_types:
        if obj_type in type_count:
            objects_of_type = [obj for obj in objects if obj.get('objectType') == obj_type]
            for obj in objects_of_type[:3]:  # 只显示前3个
                visible_status = "可见" if obj.get('visible', False) else "不可见"
                print(f"  {obj['name']} ({obj_type}) - {visible_status}")

def get_object_by_index(visible_objects, index):
    """根据索引获取物体"""
    if 1 <= index <= len(visible_objects):
        return visible_objects[index - 1]
    return None

def display_images():
    """在单独线程中显示RGBD图像"""
    global frame_counter
    while True:
        try:
            # 显示agent RGB view
            agent_frame = c.last_event.cv2img
            cv2.imshow('Agent RGB View', agent_frame)
            
            # 保存agent RGB图片
            agent_frame_path = os.path.join(agent_view_dir, f"frame_{frame_counter:05d}.png")
            cv2.imwrite(agent_frame_path, agent_frame)
            
            # 显示agent 深度图像
            if hasattr(c.last_event, 'depth_frame') and c.last_event.depth_frame is not None:
                depth_frame = c.last_event.depth_frame
                # 将深度图像转换为可视化格式
                depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_frame, alpha=255/depth_frame.max()), cv2.COLORMAP_JET)
                cv2.imshow('Agent Depth View', depth_vis)
                
                # 保存深度图像
                depth_frame_path = os.path.join(depth_view_dir, f"frame_{frame_counter:05d}.png")
                cv2.imwrite(depth_frame_path, depth_frame)
                
                # 保存深度可视化图像
                depth_vis_path = os.path.join(depth_vis_dir, f"frame_{frame_counter:05d}.png")
                cv2.imwrite(depth_vis_path, depth_vis)
            
            # 显示top view
            top_view_frame = None
            if hasattr(c.last_event, 'events') and len(c.last_event.events) > 0:
                if hasattr(c.last_event.events[0], 'third_party_camera_frames') and len(c.last_event.events[0].third_party_camera_frames) > 0:
                    top_view_frame = cv2.cvtColor(c.last_event.events[0].third_party_camera_frames[-1], cv2.COLOR_BGR2RGB)
                    cv2.imshow('Top View', top_view_frame)
                    
                    # 保存top view图片
                    top_view_frame_path = os.path.join(top_view_dir, f"frame_{frame_counter:05d}.png")
                    cv2.imwrite(top_view_frame_path, top_view_frame)
            
            cv2.waitKey(1)  # 更新窗口
            time.sleep(0.1)
            frame_counter += 1
        except:
            break

# 启动图像显示线程
display_thread = threading.Thread(target=display_images, daemon=True)
display_thread.start()

# 存储当前可见物体列表
current_visible_objects = []

# 创建保存目录
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f"logs/manual_control_{timestamp}"
agent_view_dir = os.path.join(save_dir, "agent_view")
top_view_dir = os.path.join(save_dir, "top_view")
depth_view_dir = os.path.join(save_dir, "depth_view")
depth_vis_dir = os.path.join(save_dir, "depth_visualization")

os.makedirs(agent_view_dir, exist_ok=True)
os.makedirs(top_view_dir, exist_ok=True)
os.makedirs(depth_view_dir, exist_ok=True)
os.makedirs(depth_vis_dir, exist_ok=True)

print(f"图片将保存到: {save_dir}")

# 图片计数器
frame_counter = 0

# 创建操作日志
log_file = os.path.join(save_dir, "operation_log.txt")
def log_operation(action, details=""):
    """记录操作到日志文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {action} {details}\n")

while True:
    try:
        # 获取并显示可见物体
        current_visible_objects = print_visible_objects_with_interactions(c.last_event)
        
        print("请输入动作指令: ", end='', flush=True)
        user_input = input().strip().lower()
        
        if user_input == 'x':
            print("退出.")
            log_operation("用户退出程序")
            break
            
        # 处理移动动作
        elif user_input in movement_actions:
            # 获取移动前的位置
            old_position = c.last_event.metadata["agent"]["position"]
            old_rotation = c.last_event.metadata["agent"]["rotation"]
            
            print(f"移动前位置: {old_position}")
            print(f"移动前旋转: {old_rotation}")
            
            event = movement_actions[user_input]()
            
            # 获取移动后的位置
            new_position = event.metadata["agent"]["position"]
            new_rotation = event.metadata["agent"]["rotation"]
            
            print(f"移动后位置: {new_position}")
            print(f"移动后旋转: {new_rotation}")
            
            # 检查是否有错误
            if event.metadata.get('errorMessage', ''):
                print(f"移动失败: {event.metadata['errorMessage']}")
                log_operation(f"移动失败: {user_input} - {event.metadata['errorMessage']}")
            else:
                print(f"执行移动动作: {user_input}")
                log_operation(f"移动动作: {user_input}")
            
        # 处理特殊命令
        elif user_input == 'i':  # 显示场景摘要
            print_all_objects_summary(c.last_event)
            continue
            
        elif user_input == 'z':  # 显示深度信息
            if hasattr(c.last_event, 'depth_frame') and c.last_event.depth_frame is not None:
                depth_frame = c.last_event.depth_frame
                print(f"\n=== 深度信息 ===")
                print(f"深度图像尺寸: {depth_frame.shape}")
                print(f"深度范围: {depth_frame.min():.3f}m - {depth_frame.max():.3f}m")
                print(f"平均深度: {depth_frame.mean():.3f}m")
                print(f"深度标准差: {depth_frame.std():.3f}m")
                
                # 显示中心区域的深度信息
                h, w = depth_frame.shape
                center_h, center_w = h // 2, w // 2
                center_depth = depth_frame[center_h-10:center_h+10, center_w-10:center_w+10]
                print(f"视野中心深度: {center_depth.mean():.3f}m")
                print("================\n")
            else:
                print("深度数据不可用")
            continue
            
        # 处理交互动作
        elif user_input in ['p', 'u', 'o', 'c', 't', 'b', 'l', 'y', 'r', 'f', 'h', 'g', 'k', 'n']:
            if not current_visible_objects:
                print("没有可见物体可交互")
                continue
                
            print("请输入物体编号(1-{}): ".format(len(current_visible_objects)), end='', flush=True)
            try:
                obj_index = int(input().strip())
                target_obj = get_object_by_index(current_visible_objects, obj_index)
                
                if not target_obj:
                    print("无效的物体编号")
                    continue
                    
                # 执行交互动作
                if user_input == 'p':  # 拾取
                    if target_obj.get('pickupable', False):
                        event = c.step(action="PickupObject", objectId=target_obj['id'], agentId=0, forceAction=True)
                        print(f"尝试拾取: {target_obj['name']}")
                        log_operation(f"拾取物体: {target_obj['name']} ({target_obj['id']})")
                    else:
                        print(f"{target_obj['name']} 不可拾取")
                        log_operation(f"尝试拾取失败: {target_obj['name']} (不可拾取)")
                        continue
                    
                elif user_input == 'u':  # 放下
                    event = c.step(action="PutObject", objectId=target_obj['id'], agentId=0, forceAction=True)
                    print(f"尝试放下: {target_obj['name']}")
                    log_operation(f"放下物体: {target_obj['name']} ({target_obj['id']})")
                    
                elif user_input == 'o':  # 打开
                    if target_obj.get('openable', False):
                        event = c.step(action="OpenObject", objectId=target_obj['id'], agentId=0, forceAction=True)
                        print(f"尝试打开: {target_obj['name']}")
                        log_operation(f"打开物体: {target_obj['name']} ({target_obj['id']})")
                    else:
                        print(f"{target_obj['name']} 不可打开")
                        log_operation(f"尝试打开失败: {target_obj['name']} (不可打开)")
                        continue
                    
                elif user_input == 'c':  # 关闭
                    if target_obj.get('openable', False):
                        event = c.step(action="CloseObject", objectId=target_obj['id'], agentId=0, forceAction=True)
                        print(f"尝试关闭: {target_obj['name']}")
                        log_operation(f"关闭物体: {target_obj['name']} ({target_obj['id']})")
                    else:
                        print(f"{target_obj['name']} 不可关闭")
                        log_operation(f"尝试关闭失败: {target_obj['name']} (不可关闭)")
                        continue
                    
                elif user_input == 't':  # 开关
                    if target_obj.get('toggleable', False):
                        event = c.step(action="ToggleObjectOn", objectId=target_obj['id'], agentId=0, forceAction=True)
                        print(f"尝试开关: {target_obj['name']}")
                        log_operation(f"开关物体: {target_obj['name']} ({target_obj['id']})")
                    else:
                        print(f"{target_obj['name']} 不可开关")
                        log_operation(f"尝试开关失败: {target_obj['name']} (不可开关)")
                        continue
                    
                elif user_input == 'b':  # 破坏
                    if target_obj.get('breakable', False):
                        event = c.step(action="BreakObject", objectId=target_obj['id'], agentId=0, forceAction=True)
                        print(f"尝试破坏: {target_obj['name']}")
                        log_operation(f"破坏物体: {target_obj['name']} ({target_obj['id']})")
                    else:
                        print(f"{target_obj['name']} 不可破坏")
                        log_operation(f"尝试破坏失败: {target_obj['name']} (不可破坏)")
                        continue
                    
                elif user_input == 'l':  # 切片
                    if target_obj.get('sliceable', False):
                        event = c.step(action="SliceObject", objectId=target_obj['id'], agentId=0, forceAction=True)
                        print(f"尝试切片: {target_obj['name']}")
                        log_operation(f"切片物体: {target_obj['name']} ({target_obj['id']})")
                    else:
                        print(f"{target_obj['name']} 不可切片")
                        log_operation(f"尝试切片失败: {target_obj['name']} (不可切片)")
                        continue
                    
                elif user_input == 'y':  # 投掷
                    event = c.step(action="ThrowObject", moveMagnitude=7, agentId=0, forceAction=True)
                    print(f"尝试投掷: {target_obj['name']}")
                    log_operation(f"投掷物体: {target_obj['name']} ({target_obj['id']})")
                    
                elif user_input == 'r':  # 清洁
                    if target_obj.get('dirtyable', False):
                        event = c.step(action="CleanObject", objectId=target_obj['id'], agentId=0, forceAction=True)
                        print(f"尝试清洁: {target_obj['name']}")
                        log_operation(f"清洁物体: {target_obj['name']} ({target_obj['id']})")
                    else:
                        print(f"{target_obj['name']} 不可清洁")
                        log_operation(f"尝试清洁失败: {target_obj['name']} (不可清洁)")
                        continue
                        
                elif user_input == 'f':  # 填充液体
                    if target_obj.get('canFillWithLiquid', False):
                        event = c.step(action="FillObjectWithLiquid", objectId=target_obj['id'], agentId=0, forceAction=True)
                        print(f"尝试填充液体: {target_obj['name']}")
                        log_operation(f"填充液体: {target_obj['name']} ({target_obj['id']})")
                    else:
                        print(f"{target_obj['name']} 不可填充液体")
                        log_operation(f"尝试填充液体失败: {target_obj['name']} (不可填充液体)")
                        continue
                        
                elif user_input == 'h':  # 推物体
                    event = c.step(action="PushObject", objectId=target_obj['id'], agentId=0, forceAction=True)
                    print(f"尝试推物体: {target_obj['name']}")
                    log_operation(f"推物体: {target_obj['name']} ({target_obj['id']})")
                    
                elif user_input == 'g':  # 拉物体
                    event = c.step(action="PullObject", objectId=target_obj['id'], agentId=0, forceAction=True)
                    print(f"尝试拉物体: {target_obj['name']}")
                    log_operation(f"拉物体: {target_obj['name']} ({target_obj['id']})")
                    
                elif user_input == 'k':  # 烹饪物体
                    if target_obj.get('cookable', False):
                        event = c.step(action="CookObject", objectId=target_obj['id'], agentId=0, forceAction=True)
                        print(f"尝试烹饪: {target_obj['name']}")
                        log_operation(f"烹饪物体: {target_obj['name']} ({target_obj['id']})")
                    else:
                        print(f"{target_obj['name']} 不可烹饪")
                        log_operation(f"尝试烹饪失败: {target_obj['name']} (不可烹饪)")
                        continue
                        
                elif user_input == 'n':  # 创建物体
                    # 创建物体需要指定物体类型，这里创建一个简单的物体
                    event = c.step(action="CreateObject", objectType="Apple", agentId=0, forceAction=True)
                    print(f"尝试创建苹果")
                    log_operation(f"创建物体: Apple")
                
                # 检查动作是否成功
                if event.metadata.get('errorMessage', ''):
                    print(f"动作失败: {event.metadata['errorMessage']}")
                    log_operation(f"动作失败: {event.metadata['errorMessage']}")
                else:
                    print(f"动作成功执行")
                    log_operation("动作成功执行")
                    
            except ValueError:
                print("请输入有效的数字")
                continue
                
        else:
            print("无效按键，请参考操作说明。")
            continue
            
    except KeyboardInterrupt:
        print("\n退出.")
        break
    except Exception as e:
        print(f"输入错误: {e}")
        continue

cv2.destroyAllWindows()

# 录制视频
def create_video_from_images(image_dir, output_video_path, fps=10):
    """使用ffmpeg从图片序列创建视频"""
    import glob
    import subprocess
    
    # 获取所有图片文件
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    
    if not image_files:
        print(f"在 {image_dir} 中没有找到图片文件")
        return
    
    print(f"正在创建视频: {output_video_path}")
    print(f"图片数量: {len(image_files)}")
    
    # 使用ffmpeg命令
    command_set = [
        'ffmpeg', '-i',
        f'{image_dir}/frame_%05d.png',
        '-framerate', str(fps),
        '-pix_fmt', 'yuv420p',
        '-y',  # 覆盖已存在的文件
        output_video_path
    ]
    
    try:
        subprocess.call(command_set)
        print(f"视频创建完成: {output_video_path}")
    except Exception as e:
        print(f"视频创建失败: {e}")
        print("请确保已安装ffmpeg: sudo apt-get install ffmpeg")

# 创建视频
print("\n正在创建视频...")
agent_video_path = os.path.join(save_dir, "agent_view.mp4")
top_view_video_path = os.path.join(save_dir, "top_view.mp4")
depth_video_path = os.path.join(save_dir, "depth_view.mp4")
depth_vis_video_path = os.path.join(save_dir, "depth_visualization.mp4")

create_video_from_images(agent_view_dir, agent_video_path)
create_video_from_images(top_view_dir, top_view_video_path)
create_video_from_images(depth_view_dir, depth_video_path)
create_video_from_images(depth_vis_dir, depth_vis_video_path)

print(f"\n所有文件已保存到: {save_dir}")
print(f"Agent RGB视图视频: {agent_video_path}")
print(f"Agent 深度视图视频: {depth_video_path}")
print(f"Agent 深度可视化视频: {depth_vis_video_path}")
print(f"Top视图视频: {top_view_video_path}")

def generate_video_from_saved_dir():
    """从保存的目录生成视频（类似您提供的文件中的generate_video函数）"""
    import glob
    import subprocess
    
    frame_rate = 10
    cur_path = os.path.dirname(__file__) + "/*/"
    
    for imgs_folder in glob(cur_path, recursive=False):
        if not os.path.isdir(imgs_folder):
            continue
            
        view = imgs_folder.split('/')[-1]
        if view.startswith('manual_control_'):
            print(f"处理目录: {imgs_folder}")
            
            # 检查是否有图片文件
            agent_view_dir = os.path.join(imgs_folder, "agent_view")
            top_view_dir = os.path.join(imgs_folder, "top_view")
            depth_view_dir = os.path.join(imgs_folder, "depth_view")
            depth_vis_dir = os.path.join(imgs_folder, "depth_visualization")
            
            # 创建Agent RGB视频
            if os.path.exists(agent_view_dir):
                agent_video_path = os.path.join(imgs_folder, "agent_view.mp4")
                command_set = [
                    'ffmpeg', '-i',
                    f'{agent_view_dir}/frame_%05d.png',
                    '-framerate', str(frame_rate),
                    '-pix_fmt', 'yuv420p',
                    '-y',
                    agent_video_path
                ]
                try:
                    subprocess.call(command_set)
                    print(f"Agent RGB视图视频创建完成: {agent_video_path}")
                except Exception as e:
                    print(f"Agent RGB视图视频创建失败: {e}")
            
            # 创建深度视频
            if os.path.exists(depth_view_dir):
                depth_video_path = os.path.join(imgs_folder, "depth_view.mp4")
                command_set = [
                    'ffmpeg', '-i',
                    f'{depth_view_dir}/frame_%05d.png',
                    '-framerate', str(frame_rate),
                    '-pix_fmt', 'yuv420p',
                    '-y',
                    depth_video_path
                ]
                try:
                    subprocess.call(command_set)
                    print(f"Agent 深度视图视频创建完成: {depth_video_path}")
                except Exception as e:
                    print(f"Agent 深度视图视频创建失败: {e}")
            
            # 创建深度可视化视频
            if os.path.exists(depth_vis_dir):
                depth_vis_video_path = os.path.join(imgs_folder, "depth_visualization.mp4")
                command_set = [
                    'ffmpeg', '-i',
                    f'{depth_vis_dir}/frame_%05d.png',
                    '-framerate', str(frame_rate),
                    '-pix_fmt', 'yuv420p',
                    '-y',
                    depth_vis_video_path
                ]
                try:
                    subprocess.call(command_set)
                    print(f"Agent 深度可视化视频创建完成: {depth_vis_video_path}")
                except Exception as e:
                    print(f"Agent 深度可视化视频创建失败: {e}")
            
            # 创建Top视图视频
            if os.path.exists(top_view_dir):
                top_video_path = os.path.join(imgs_folder, "top_view.mp4")
                command_set = [
                    'ffmpeg', '-i',
                    f'{top_view_dir}/frame_%05d.png',
                    '-framerate', str(frame_rate),
                    '-pix_fmt', 'yuv420p',
                    '-y',
                    top_video_path
                ]
                try:
                    subprocess.call(command_set)
                    print(f"Top视图视频创建完成: {top_video_path}")
                except Exception as e:
                    print(f"Top视图视频创建失败: {e}")

# 如果需要重新生成所有视频，可以调用这个函数
# generate_video_from_saved_dir() 