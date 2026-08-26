#!/usr/bin/env python3
"""
NPZ文件可视化脚本（纯可视化，不加载模型）
可视化npz文件中的输入数据和可选的预测结果

使用方法:
    # 只可视化输入数据（GT）
    python visualize_npz.py --npz_path /path/to/file.npz --output_dir ./output/vis

    # 可视化输入 + 外部预测结果
    python visualize_npz.py --npz_path /path/to/file.npz --output_dir ./output/vis --pred_path /path/to/predictions.npy
    
    # 批量处理
    python visualize_npz.py --npz_dir /path/to/npz_dir --output_dir ./output/vis

    python /home/lsl/robot_ws/semantic_ws/src/semantic_mapping_pkg/script/visualize_npz.py --npz_dir /home/ycs/robot_ws/semantic_ws/data/raw_npz/working --output_dir /home/ycs/robot_ws/semantic_ws/data/vis_data
    
"""

import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import glob


def visualize_semantic(gt_semantic, save_path, explored_mask=None, trajectory=None, agent_history=None, agent_pos=None):
    """
    可视化语义标签，未探索区域显示为黑色，可选择添加轨迹
    
    Args:
        gt_semantic: 语义标签图 (H, W)，值为0-9
        save_path: 保存路径
        explored_mask: 已探索区域掩码，如果提供则未探索区域显示为黑色
        trajectory: 轨迹点列表 [(x1,y1), (x2,y2), ...]，如果提供则绘制红色轨迹线
        agent_history: 智能体历史轨迹掩码 (H, W)，如果提供则绘制红色轨迹
        agent_pos: 智能体当前位置 (x, y)，如果提供则绘制蓝色当前位置
    """
    palette = np.array([
        [0, 255, 255],       # 0 客厅 - 黄色
        [255, 0, 0],         # 1 卧室 - 蓝色 (BGR)
        [0, 0, 255],         # 2 厨房 - 红色 (BGR)
        [0, 255, 0],         # 3 洗手间 - 绿色
        [255, 0, 255],       # 4 阳台 - 品红色
        [128, 0, 128],       # 5 储藏间 - 紫色
        [255, 255, 0],       # 6 门 - 蓝绿色 (BGR)
        [128, 128, 128],     # 7 墙 - 中灰色
        [0, 165, 255],       # 8 大门 - 橙色 (BGR)
        [200, 160, 120]      # 9 外部区域 - 淡蓝色 (BGR)，柔和饱和度
    ], dtype=np.uint8)
    
    color_img = palette[gt_semantic]
    
    # 如果有探索掩码，将未探索区域设为黑色
    if explored_mask is not None:
        unexplored_mask = ~explored_mask
        color_img[unexplored_mask] = [0, 0, 0]  # 黑色 (BGR)
    
    # 1. 绘制智能体轨迹（红色）- 使用agent_history掩码
    if agent_history is not None:
        trajectory_mask = agent_history > 0
        color_img[trajectory_mask] = [0, 0, 255]  # 红色轨迹
    
    # 2. 绘制智能体当前位置（蓝色）
    if agent_pos is not None:
        x, y = agent_pos
        if 0 <= x < color_img.shape[0] and 0 <= y < color_img.shape[1]:
            color_img[x, y] = [255, 0, 0]  # 蓝色当前位置
    
    # 3. 兼容原有的轨迹点列表方式
    if trajectory is not None and len(trajectory) > 0:
        for i, (x, y) in enumerate(trajectory):
            if 0 <= x < color_img.shape[0] and 0 <= y < color_img.shape[1]:
                cv2.circle(color_img, (int(y), int(x)), 1, (0, 0, 255), -1)  # 红色小圆点
                if i > 0:
                    prev_x, prev_y = trajectory[i-1]
                    if 0 <= prev_x < color_img.shape[0] and 0 <= prev_y < color_img.shape[1]:
                        cv2.line(color_img, (int(prev_y), int(prev_x)), (int(y), int(x)), (0, 0, 255), 1)
    
    cv2.imwrite(save_path, color_img)


def visualize_probability_heatmap(prob_map, save_path, vmin=0, vmax=1):
    """
    可视化概率热力图，灰色底色，高概率区域显示橙色
    
    Args:
        prob_map: 概率图 (H, W)
        save_path: 保存路径
        vmin: 最小值
        vmax: 最大值
    """
    H, W = prob_map.shape
    color_img = np.full((H, W, 3), 128, dtype=np.uint8)  # 灰色底色
    
    # 针对低概率值优化
    if vmax <= 0.02:
        adjusted_vmax = 0.02
        high_prob_threshold = 0.02
        medium_prob_threshold = 0.01
    else:
        adjusted_vmax = vmax
        high_prob_threshold = vmax * 0.8
        medium_prob_threshold = vmax * 0.4
    
    # 高概率区域：橙色
    high_prob_mask = prob_map > high_prob_threshold
    if np.any(high_prob_mask):
        color_img[high_prob_mask] = [0, 165, 255]  # 橙色 (BGR)
    
    # 中等概率区域：浅橙色
    medium_prob_mask = (prob_map > medium_prob_threshold) & ~high_prob_mask
    if np.any(medium_prob_mask):
        color_img[medium_prob_mask] = [0, 200, 255]  # 浅橙色 (BGR)
    
    cv2.imwrite(save_path, color_img)


def visualize_channel(channel_data, save_path, channel_name="", colormap='gray'):
    """
    可视化单个通道
    
    Args:
        channel_data: 通道数据 (H, W)
        save_path: 保存路径
        channel_name: 通道名称（用于标题）
        colormap: 颜色映射 ('gray', 'jet', 'hot'等)
    """
    H, W = channel_data.shape
    
    # 归一化到0-255
    if channel_data.max() > 1.0:
        normalized = channel_data.astype(np.uint8)
    else:
        normalized = (channel_data * 255).astype(np.uint8)
    
    # 应用颜色映射
    if colormap == 'gray':
        # 对于灰度图，直接保存为灰度图像（不需要颜色映射）
        # 如果需要彩色显示，可以使用 COLORMAP_BONE 或 COLORMAP_VIRIDIS
        color_img = cv2.applyColorMap(normalized, cv2.COLORMAP_BONE)
    elif colormap == 'jet':
        color_img = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    elif colormap == 'hot':
        color_img = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
    else:
        # 默认使用 BONE 颜色映射（类似灰度）
        color_img = cv2.applyColorMap(normalized, cv2.COLORMAP_BONE)
    
    cv2.imwrite(save_path, color_img)


def visualize_occupancy_map(occ_map, save_path):
    """
    可视化占用图
    
    Args:
        occ_map: 占用图 (H, W)，0=自由，100=障碍，255=未知
        save_path: 保存路径
    """
    color_map = np.zeros((occ_map.shape[0], occ_map.shape[1], 3), dtype=np.uint8)
    color_map[occ_map == 255] = [128, 128, 128]  # 未知灰色
    color_map[occ_map == 0] = [255, 255, 255]    # 自由白色
    color_map[occ_map == 100] = [0, 0, 0]        # 障碍黑色
    cv2.imwrite(save_path, color_map)


def create_video_from_frames(frame_dir, video_path, fps=5):
    """从帧图像创建视频"""
    img_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.png')])
    if not img_files:
        print("没有帧图像，无法生成视频")
        return False
    
    frame = cv2.imread(img_files[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for img_file in img_files:
        frame = cv2.imread(img_file)
        video_writer.write(frame)
    video_writer.release()
    return True


def visualize_single_npz(npz_path, output_dir, pred_data=None, target_class_id=None, target_class_name=None):
    """
    可视化单个npz文件（不加载模型）
    
    Args:
        npz_path: npz文件路径
        output_dir: 输出目录
        pred_data: 外部预测结果 (T, 10, H, W) 或 (10, H, W)，可选
        target_class_id: 目标类别ID（用于生成概率热力图）
        target_class_name: 目标类别名称
    """
    # 获取数据集名称（不含扩展名）
    dataset_name = os.path.splitext(os.path.basename(npz_path))[0]
    result_dir = os.path.join(output_dir, f'result_{dataset_name}')
    os.makedirs(result_dir, exist_ok=True)
    
    # 创建视频输出目录
    combined_semantic_frame_dir = os.path.join(result_dir, "combined_semantic_frames")
    probability_heatmap_frame_dir = os.path.join(result_dir, "probability_heatmap_frames")
    input_channels_dir = os.path.join(result_dir, "input_channels")
    os.makedirs(combined_semantic_frame_dir, exist_ok=True)
    os.makedirs(probability_heatmap_frame_dir, exist_ok=True)
    os.makedirs(input_channels_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"正在可视化数据集: {dataset_name}")
    print(f"结果保存至: {result_dir}")
    print(f"{'='*50}")

    try:
        data = np.load(npz_path)
        
        # 支持三种格式：
        # 1. 旧格式：'maps' 键，形状为 (T, C, H, W) - 多帧时间序列
        # 2. 新格式：'channels' 键，形状为 (C, H, W) - 单帧
        # 3. 时间维度格式：'semantic_timeline' 键，形状为 (T, 10, H, W) - 只有语义通道
        if 'maps' in data:
            maps = data['maps']  # (T, C, H, W)
            num_frames = maps.shape[0]
            C, H, W = maps.shape[1], maps.shape[2], maps.shape[3]
            is_multi_frame = True
            print(f"数据集包含 {num_frames} 帧，每帧 {C} 通道，尺寸 {H}x{W}")
        elif 'channels' in data:
            channels = data['channels']  # (C, H, W)
            # 转换为多帧格式（只有一帧）
            maps = channels[np.newaxis, :, :, :]  # (1, C, H, W)
            num_frames = 1
            C, H, W = channels.shape[0], channels.shape[1], channels.shape[2]
            is_multi_frame = False
            print(f"数据集包含单帧，{C} 通道，尺寸 {H}x{W}")
            # 打印元信息（如果存在）
            if 'width' in data:
                print(f"地图元信息: width={data['width']}, height={data['height']}, resolution={data.get('resolution', 'N/A')}")
        elif 'semantic_timeline' in data:
            # 时间维度格式：优先使用完整14通道（如果存在maps键），否则使用semantic_timeline
            if 'maps' in data:
                # 新格式：包含完整14通道
                maps = data['maps']  # (T, 14, H, W)
                T, C, H, W = maps.shape
                num_frames = T
                is_multi_frame = True
                print(f"时间维度数据集包含 {num_frames} 帧，每帧 {C} 通道（包含完整14通道），尺寸 {H}x{W}")
            else:
                # 旧格式：只有语义通道 (T, 10, H, W)
                semantic_timeline = data['semantic_timeline']  # (T, 10, H, W)
                T, num_classes, H, W = semantic_timeline.shape
                num_frames = T
                C = 14  # 完整14通道
                
                # 创建完整的14通道数组，前4个通道设为0，后10个通道使用semantic_timeline
                maps = np.zeros((T, 14, H, W), dtype=np.uint8)
                maps[:, 4:14, :, :] = semantic_timeline  # 填充语义通道
                
                is_multi_frame = True
                print(f"时间维度数据集包含 {num_frames} 帧，每帧 {num_classes} 个语义通道，尺寸 {H}x{W}")
                print(f"注意：前4个通道（occupancy, explored_mask, agent_position, agent_history）未提供，已设为0")
            # 打印元信息（如果存在）
            if 'width' in data:
                print(f"地图元信息: width={data['width']}, height={data['height']}, resolution={data.get('resolution', 'N/A')}")
        else:
            raise KeyError(f"NPZ文件中未找到 'maps'、'channels' 或 'semantic_timeline' 键。可用键: {list(data.keys())}")

        # 最后一帧的完整地图（用于参考，但注意这是输入数据，不是GT）
        last_frame = maps[-1]  # shape: (C, H, W)
        last_sem = last_frame[4:14]  # 语义通道部分 shape: (10, H, W)
        # 将多通道概率图转换为单通道类别标签（取最大概率对应的类别）
        # 注意：未探索区域（所有通道为0）会被标记为类别0，需要后续处理
        last_sem_label = np.argmax(last_sem, axis=0).astype(np.uint8)
        
        # 计算最后一帧的已探索区域（用于gt_complete）
        if last_frame.shape[0] >= 14 and np.all(last_frame[1] == 0):
            # 时间维度格式：使用语义通道推断已探索区域
            last_explored_mask = np.any(last_sem > 0, axis=0)
        else:
            last_explored_mask = last_frame[1] > 0
        
        # 保存完整地图（只显示已探索区域，未探索区域为黑色）
        last_sem_label_masked = np.zeros_like(last_sem_label, dtype=np.uint8)
        last_sem_label_masked[last_explored_mask] = last_sem_label[last_explored_mask]
        visualize_semantic(last_sem_label_masked, os.path.join(result_dir, 'input_complete.png'), 
                          explored_mask=last_explored_mask)
        print(f"完整输入地图中的类别（仅已探索区域）: {np.unique(last_sem_label[last_explored_mask])}")

        # 初始化轨迹追踪
        trajectory = []
        
        # 处理每一帧
        for t_idx in range(num_frames):
            print(f"\n=== 处理第 {t_idx}/{num_frames-1} 帧 ===")
            
            # 获取当前帧的探索状态
            current_frame = maps[t_idx]  # (C, H, W)
            # 对于时间维度格式，前4个通道可能为0，使用语义通道来推断已探索区域
            if current_frame.shape[0] >= 14 and np.all(current_frame[1] == 0):
                # 时间维度格式：使用语义通道推断已探索区域（至少有一个语义类别有值）
                semantic_channels = current_frame[4:14]  # (10, H, W)
                explored_mask = np.any(semantic_channels > 0, axis=0)  # 至少有一个语义类别有值
            else:
                explored_mask = current_frame[1] > 0  # 已探索区域掩码
            unexplored_mask = ~explored_mask      # 未探索区域掩码
            
            # 提取智能体位置和历史轨迹
            agent_pos = None
            agent_history = None
            if current_frame.shape[0] > 3:
                # 智能体位置通道 (第3个通道，索引为2)
                agent_pose = current_frame[2]
                if np.any(agent_pose > 0):
                    agent_pos = np.unravel_index(np.argmax(agent_pose), agent_pose.shape)
                    trajectory.append(agent_pos)
                
                # 智能体历史轨迹通道 (第4个通道，索引为3)
                agent_history = current_frame[3]
            
            explored_pixels = np.sum(explored_mask)
            total_pixels = explored_mask.size
            exploration_ratio = explored_pixels / total_pixels
            print(f"探索进度: {explored_pixels}/{total_pixels} ({exploration_ratio:.2%})")
            
            # ===== 可视化输入通道 =====
            channel_names = [
                "occupancy", "explored_mask", "agent_position", "agent_history",
                "semantic_class_0", "semantic_class_1", "semantic_class_2", "semantic_class_3",
                "semantic_class_4", "semantic_class_5", "semantic_class_6", "semantic_class_7",
                "semantic_class_8", "semantic_class_9"
            ]
            
            for ch_idx in range(min(C, len(channel_names))):
                channel_data = current_frame[ch_idx]
                channel_name = channel_names[ch_idx]
                
                # 打印通道统计信息
                non_zero_count = np.count_nonzero(channel_data)
                max_val = channel_data.max()
                min_val = channel_data.min()
                print(f"  通道 {ch_idx} ({channel_name}): 非零像素={non_zero_count}, 值范围=[{min_val}, {max_val}]")
                
                # 特殊处理占用图
                if ch_idx == 0:
                    occ_path = os.path.join(input_channels_dir, f'frame_{t_idx:02d}_channel_{ch_idx:02d}_{channel_name}.png')
                    visualize_occupancy_map(channel_data, occ_path)
                else:
                    # 其他通道使用灰度/热力图
                    colormap = 'hot' if ch_idx in [2, 3] else 'gray'  # 智能体位置和历史用热力图
                    channel_path = os.path.join(input_channels_dir, f'frame_{t_idx:02d}_channel_{ch_idx:02d}_{channel_name}.png')
                    visualize_channel(channel_data, channel_path, channel_name, colormap)
            
            # ===== 可视化输入数据（当前帧的感知输入） =====
            # 从当前帧提取语义通道（这是实际的感知输入，不是GT）
            current_sem = current_frame[4:14]  # (10, H, W) - 当前帧的语义通道
            # 将多通道概率图转换为单通道类别标签（取最大概率对应的类别）
            current_sem_label = np.argmax(current_sem, axis=0).astype(np.uint8)
            
            # 只显示已探索区域的语义标签（未探索区域显示为黑色，带轨迹）
            explored_semantic = np.zeros_like(current_sem_label, dtype=np.uint8)
            explored_semantic[explored_mask] = current_sem_label[explored_mask]
            explored_path = os.path.join(result_dir, f'frame_{t_idx:02d}_explored_input.png')
            visualize_semantic(explored_semantic, explored_path, explored_mask, trajectory, agent_history, agent_pos)
            
            # ===== 如果有外部预测结果，可视化预测 =====
            if pred_data is not None:
                # 处理预测数据维度
                if pred_data.ndim == 3:  # (10, H, W) - 单帧预测
                    if t_idx == 0:  # 只在第一帧使用
                        preds = pred_data
                    else:
                        continue  # 跳过其他帧
                elif pred_data.ndim == 4:  # (T, 10, H, W) - 多帧预测
                    if t_idx < pred_data.shape[0]:
                        preds = pred_data[t_idx]
                    else:
                        continue  # 跳过超出范围的帧
                else:
                    print(f"警告: 预测数据维度不正确: {pred_data.shape}")
                    continue
                
                # 确保preds是numpy数组
                if isinstance(preds, np.ndarray):
                    # 如果预测是概率图（0-1范围），转换为类别
                    if preds.max() <= 1.0:
                        pred_classes = preds.argmax(axis=0).astype(np.uint8)
                    else:
                        pred_classes = preds.argmax(axis=0).astype(np.uint8)
                else:
                    continue
                
                # 3. 保存完整语义预测（已探索区域显示输入，未探索区域显示预测）
                # 使用当前帧的输入语义标签，而不是最后一帧的GT
                combined_semantic = current_sem_label.copy()
                combined_semantic[unexplored_mask] = pred_classes[unexplored_mask]
                
                combined_path = os.path.join(result_dir, f'frame_{t_idx:02d}_combined_semantic.png')
                visualize_semantic(combined_semantic, combined_path, agent_history=agent_history, agent_pos=agent_pos)
                
                # 同时保存到视频帧目录
                video_frame_path = os.path.join(combined_semantic_frame_dir, f'{t_idx:04d}.png')
                visualize_semantic(combined_semantic, video_frame_path, agent_history=agent_history, agent_pos=agent_pos)
                
                # 如果指定了目标类别，生成概率热力图
                if target_class_id is not None and target_class_id < preds.shape[0]:
                    target_prob = preds[target_class_id]  # (H, W)
                    target_prob_unexplored = target_prob * unexplored_mask  # 只关注未探索区域
                    
                    print(f"{target_class_name or f'类别{target_class_id}'}预测概率范围: {target_prob.min():.4f} - {target_prob.max():.4f}")
                    
                    # 1. 保存目标类别在未探索区域的概率热力图
                    prob_heatmap_path = os.path.join(result_dir, f'frame_{t_idx:02d}_{target_class_name or f"class_{target_class_id}"}_unexplored_heatmap.png')
                    visualize_probability_heatmap(target_prob_unexplored, prob_heatmap_path, vmin=0, vmax=0.02)
                    
                    # 同时保存到视频帧目录
                    prob_video_frame_path = os.path.join(probability_heatmap_frame_dir, f'{t_idx:04d}.png')
                    visualize_probability_heatmap(target_prob_unexplored, prob_video_frame_path, vmin=0, vmax=0.02)
                    
                    # 2. 保存二值化预测
                    high_prob_threshold = 0.02
                    medium_prob_threshold = 0.01
                    
                    high_prob_mask = (target_prob_unexplored > high_prob_threshold).astype(np.uint8) * 255
                    medium_prob_mask = (target_prob_unexplored > medium_prob_threshold).astype(np.uint8) * 255
                    
                    high_prob_path = os.path.join(result_dir, f'frame_{t_idx:02d}_{target_class_name or f"class_{target_class_id}"}_high_prob_0.02.png')
                    medium_prob_path = os.path.join(result_dir, f'frame_{t_idx:02d}_{target_class_name or f"class_{target_class_id}"}_medium_prob_0.01.png')
                    
                    cv2.imwrite(high_prob_path, high_prob_mask)
                    cv2.imwrite(medium_prob_path, medium_prob_mask)
        
        # 生成视频（如果有预测结果）
        if pred_data is not None:
            # 生成combined_semantic视频
            print(f"\n正在生成Combined Semantic视频...")
            combined_semantic_video_path = os.path.join(result_dir, f'{dataset_name}_combined_semantic.mp4')
            video_success = create_video_from_frames(combined_semantic_frame_dir, combined_semantic_video_path, fps=5)
            if video_success:
                print(f"✓ Combined Semantic视频已保存到: {combined_semantic_video_path}")
            
            # 生成概率热力图视频
            if target_class_id is not None:
                print(f"\n正在生成Probability Heatmap视频...")
                probability_heatmap_video_path = os.path.join(result_dir, f'{dataset_name}_probability_heatmap.mp4')
                video_success = create_video_from_frames(probability_heatmap_frame_dir, probability_heatmap_video_path, fps=5)
                if video_success:
                    print(f"✓ Probability Heatmap视频已保存到: {probability_heatmap_video_path}")
        
        print(f"\n数据集 {dataset_name} 可视化完成！结果保存在: {result_dir}")
        
    except Exception as e:
        print(f"处理数据集 {dataset_name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='NPZ文件可视化脚本（纯可视化，不加载模型）')
    parser.add_argument('--npz_path', type=str, help='单个npz文件路径')
    parser.add_argument('--npz_dir', type=str, help='npz文件目录（批量处理）')
    parser.add_argument('--output_dir', type=str, default='./output/vis', help='输出目录')
    parser.add_argument('--pred_path', type=str, default=None, help='外部预测结果文件路径（.npy格式，形状为(T,10,H,W)或(10,H,W)）')
    parser.add_argument('--target_class_id', type=int, default=None, help='目标类别ID（用于生成概率热力图，需要提供pred_path）')
    parser.add_argument('--target_class_name', type=str, default=None, help='目标类别名称')
    
    args = parser.parse_args()
    
    # 确定要处理的文件列表
    npz_files = []
    if args.npz_path:
        if os.path.exists(args.npz_path):
            npz_files = [args.npz_path]
        else:
            print(f"错误: 文件不存在: {args.npz_path}")
            return
    elif args.npz_dir:
        if os.path.exists(args.npz_dir):
            npz_files = sorted(glob.glob(os.path.join(args.npz_dir, '*.npz')))
            print(f"找到 {len(npz_files)} 个npz文件")
        else:
            print(f"错误: 目录不存在: {args.npz_dir}")
            return
    else:
        print("错误: 必须指定 --npz_path 或 --npz_dir")
        parser.print_help()
        return
    
    if not npz_files:
        print("错误: 没有找到npz文件")
        return
    
    # 加载外部预测结果（如果提供）
    pred_data = None
    if args.pred_path:
        if os.path.exists(args.pred_path):
            try:
                pred_data = np.load(args.pred_path)
                print(f"已加载预测结果，形状: {pred_data.shape}")
            except Exception as e:
                print(f"警告: 无法加载预测结果: {e}")
                pred_data = None
        else:
            print(f"警告: 预测结果文件不存在: {args.pred_path}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个npz文件
    for i, npz_path in enumerate(npz_files):
        print(f"\n{'='*60}")
        print(f"处理进度: {i+1}/{len(npz_files)}")
        print(f"{'='*60}")
        
        # 如果提供了预测结果，需要匹配对应的预测数据
        current_pred_data = pred_data
        # 这里可以根据文件名匹配，暂时使用同一个pred_data
        
        visualize_single_npz(
            npz_path=npz_path,
            output_dir=args.output_dir,
            pred_data=current_pred_data,
            target_class_id=args.target_class_id,
            target_class_name=args.target_class_name
        )
    
    print(f"\n{'='*60}")
    print(f"所有文件处理完成！结果保存在: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
