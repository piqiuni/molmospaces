#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
场景建图到14通道格式转换节点

功能：
1. 订阅场景语义地图相关话题
2. 订阅占用地图（occupancy grid）
3. 订阅智能体位置（通过TF）
4. 记录智能体历史轨迹
5. 将所有数据转换为14通道格式并保存为npz文件

14通道格式：
- 通道0: 占用图 (Occupancy Map)
- 通道1: 探索掩码 (Explored Mask)
- 通道2: 智能体位置 (Agent Position)
- 通道3: 智能体历史轨迹 (Agent History)
- 通道4-13: 10个语义类别概率图 (Semantic Probability Maps)
"""

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
import tf2_ros
from geometry_msgs.msg import PoseStamped
from collections import deque
import threading
import os
from datetime import datetime

# 语义类别映射（根据 channel_meaning.md）
SEMANTIC_CLASSES = [
    "livingroom",    # 类别0 - 客厅 (通道4)
    "bedroom",       # 类别1 - 卧室 (通道5)
    "kitchen",       # 类别2 - 厨房 (通道6)
    "bathroom",      # 类别3 - 洗手间 (通道7)
    "balcony",       # 类别4 - 阳台 (通道8)
    "storage",       # 类别5 - 储藏间 (通道9)
    "door",          # 类别6 - 门 (通道10)
    "wall",          # 类别7 - 墙 (通道11)
    "entrance",      # 类别8 - 大门 (通道12)
    "outside"        # 类别9 - 外部区域 (通道13)
]

class SceneTo14ChannelConverter:
    def __init__(self):
        """初始化转换节点"""
        rospy.init_node('scene_to_14channel_converter', anonymous=True)
        
        # 获取参数
        self.occupancy_grid_topic = rospy.get_param("~occupancy_grid_topic", "/struct_mapping/wall_occ_map")
        self.scene_id_grid_topic = rospy.get_param("~scene_id_grid_topic", "/semantic_mapping/scene_id_grid")
        self.scene_confidence_grid_topic = rospy.get_param("~scene_confidence_grid_topic", "/semantic_mapping/scene_confidence_grid")
        self.agent_frame = rospy.get_param("~agent_frame", "base_link")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.save_rate = rospy.get_param("~save_rate", 1.0)  # Hz
        
        # 硬编码保存路径
        base_save_dir = "/home/wxy/Downloads/Interactive-Nav-SG-nav/semantic_ws/data/raw_npz"
        
        # 为每次实验创建以时间戳命名的子文件夹
        experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(base_save_dir, experiment_timestamp)
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 文件计数器
        self.file_counter = 0
        
        # 时间维度累积数组 (T, 14, H, W) - 存储所有时间步的完整14通道数据
        self.semantic_timeline = []  # 列表，每个元素是 (14, H, W) 的数组
        self.timeline_initialized = False
        self.timeline_height = None
        self.timeline_width = None
        
        # 数据缓存
        self.occupancy_grid = None
        self.scene_id_grid = None
        self.scene_confidence_grid = None
        self.lock = threading.Lock()
        
        # 智能体历史轨迹（存储最近N个位置）
        self.trajectory_history = deque(maxlen=1000)  # 最多保存1000个位置
        self.trajectory_decay = 0.95  # 轨迹衰减因子
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 订阅者
        self.occ_grid_sub = rospy.Subscriber(
            self.occupancy_grid_topic, 
            OccupancyGrid, 
            self.occupancy_grid_callback,
            queue_size=1
        )
        
        self.scene_id_sub = rospy.Subscriber(
            self.scene_id_grid_topic,
            OccupancyGrid,
            self.scene_id_grid_callback,
            queue_size=1
        )
        
        self.scene_conf_sub = rospy.Subscriber(
            self.scene_confidence_grid_topic,
            OccupancyGrid,
            self.scene_confidence_grid_callback,
            queue_size=1
        )
        
        # 定时器：更新智能体位置和轨迹，并保存14通道地图为npz文件
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.save_rate),
            self.timer_callback
        )
        
        # 定期保存时间维度数组的定时器（每60秒保存一次，避免数据丢失）
        self.timeline_save_timer = rospy.Timer(
            rospy.Duration(20.0),
            self.periodic_save_timeline
        )
        
        # 注册关闭回调，保存时间维度累积数组
        rospy.on_shutdown(self.save_timeline)
        
        rospy.loginfo("场景建图到14通道格式转换节点已启动")
        rospy.loginfo(f"  占用地图话题: {self.occupancy_grid_topic}")
        rospy.loginfo(f"  场景ID地图话题: {self.scene_id_grid_topic}")
        rospy.loginfo(f"  场景置信度地图话题: {self.scene_confidence_grid_topic}")
        rospy.loginfo(f"  智能体坐标系: {self.agent_frame}")
        rospy.loginfo(f"  地图坐标系: {self.map_frame}")
        rospy.loginfo(f"  保存频率: {self.save_rate} Hz")
        rospy.loginfo(f"  保存目录: {self.save_dir}")
    
    def occupancy_grid_callback(self, msg):
        """占用地图回调"""
        with self.lock:
            self.occupancy_grid = msg
    
    def scene_id_grid_callback(self, msg):
        """场景ID地图回调"""
        with self.lock:
            self.scene_id_grid = msg
    
    def scene_confidence_grid_callback(self, msg):
        """场景置信度地图回调"""
        with self.lock:
            self.scene_confidence_grid = msg
    
    def get_agent_pose(self):
        """获取智能体在地图坐标系中的位置"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.agent_frame,
                rospy.Time(0),
                rospy.Duration(0.1)
            )
            pose = PoseStamped()
            pose.header.frame_id = self.map_frame
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            return pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, f"TF查找失败: {e}")
            return None
    
    def world_to_grid(self, x, y, grid_info):
        """世界坐标转栅格坐标"""
        grid_x = int((x - grid_info.origin.position.x) / grid_info.resolution)
        grid_y = int((y - grid_info.origin.position.y) / grid_info.resolution)
        return grid_x, grid_y
    
    def grid_to_index(self, grid_x, grid_y, width):
        """栅格坐标转一维索引"""
        return grid_y * width + grid_x
    
    def convert_to_14channel(self):
        """转换为14通道格式"""
        with self.lock:
            # 检查数据是否完整
            if self.occupancy_grid is None:
                return None
            
            # 使用占用地图的尺寸作为基准
            width = self.occupancy_grid.info.width
            height = self.occupancy_grid.info.height
            resolution = self.occupancy_grid.info.resolution
            origin_x = self.occupancy_grid.info.origin.position.x
            origin_y = self.occupancy_grid.info.origin.position.y
            
            # 创建14通道数组 (C, H, W)
            channels = np.zeros((14, height, width), dtype=np.uint8)
            
            # ========== 通道0: 占用图 ==========
            occ_data = np.array(self.occupancy_grid.data, dtype=np.int8)
            occ_data = occ_data.reshape((height, width))
            # 转换为0-255范围：-1->0, 0->0, >0->255
            channels[0] = np.where(occ_data > 0, 255, 0).astype(np.uint8)
            
            # ========== 通道11: 墙（直接从occupancy赋值，因为occupancy已经筛选过，只剩下墙壁占据）==========
            # 语义类别索引7对应通道11
            channels[11] = channels[0].copy()  # 直接使用occupancy数据作为wall通道
            
            # ========== 通道2: 智能体位置 ==========
            agent_pose = self.get_agent_pose()
            if agent_pose:
                agent_x = agent_pose.pose.position.x
                agent_y = agent_pose.pose.position.y
                grid_x, grid_y = self.world_to_grid(agent_x, agent_y, self.occupancy_grid.info)
                
                if 0 <= grid_x < width and 0 <= grid_y < height:
                    channels[2, grid_y, grid_x] = 255
                    # 更新轨迹历史
                    self.trajectory_history.append((grid_x, grid_y))
            
            # ========== 通道3: 智能体历史轨迹 ==========
            # 绘制轨迹（带衰减）
            if len(self.trajectory_history) > 0:
                for i, (tx, ty) in enumerate(self.trajectory_history):
                    if 0 <= tx < width and 0 <= ty < height:
                        # 越新的轨迹值越大（衰减）
                        decay_value = int(255 * (self.trajectory_decay ** (len(self.trajectory_history) - i - 1)))
                        channels[3, ty, tx] = max(channels[3, ty, tx], decay_value)
            
            # ========== 通道4-13: 语义类别概率图 ==========
            # 重要：只使用实时已探索区域的语义信息（与探索掩码保持一致）
            # 这样可以确保语义通道反映的是当前时刻的实时场景属性，而不是完全探索后的最终状态
            # scene_id 直接就是语义类别索引：-1=未知, 0-9=语义类别索引（直接对应通道4-13）
            if self.scene_id_grid is not None and self.scene_confidence_grid is not None:
                scene_id_data = np.array(self.scene_id_grid.data, dtype=np.int8)
                scene_id_data = scene_id_data.reshape((height, width))
                
                confidence_data = np.array(self.scene_confidence_grid.data, dtype=np.int8)
                confidence_data = confidence_data.reshape((height, width))
                
                # 统计信息
                unique_scene_ids = np.unique(scene_id_data)
                non_unknown_ids = unique_scene_ids[unique_scene_ids != -1]
                rospy.loginfo(f"Scene ID统计: 总共有 {len(unique_scene_ids)} 个不同的scene_id, 非未知ID: {non_unknown_ids.tolist()}")
                
                filled_count = 0
                skipped_not_explored = 0
                invalid_ids = set()
                for y in range(height):
                    for x in range(width):
                        # 关键修改：只处理已探索的区域（与探索掩码保持一致）
                        # 如果该位置在occupancy_grid中是未知（-1），则跳过，即使scene_id_grid中有值
                        if occ_data[y, x] == -1:
                            skipped_not_explored += 1
                            continue
                        
                        scene_id = int(scene_id_data[y, x])
                        conf_val = int(confidence_data[y, x])
                        
                        # 跳过未知区域
                        if scene_id == -1 or conf_val == -1:
                            continue
                        
                        # scene_id 直接就是语义类别索引（0-9对应通道4-13）
                        if 0 <= scene_id <= 9:
                            # 置信度转换为0-255范围（conf_val是0-100）
                            confidence_uint8 = int(conf_val * 255 / 100)
                            channel_idx = 4 + scene_id  # 直接使用scene_id作为语义类别索引
                            channels[channel_idx, y, x] = confidence_uint8
                            filled_count += 1
                        else:
                            invalid_ids.add(scene_id)
                
                if invalid_ids:
                    rospy.logwarn(f"以下Scene ID超出有效范围[0-9]: {sorted(invalid_ids)}")
                rospy.loginfo(f"语义通道填充统计: 填充了 {filled_count} 个像素点, "
                            f"跳过了 {skipped_not_explored} 个未探索区域的像素点")
            else:
                rospy.logwarn_throttle(5.0, f"语义地图数据缺失: scene_id_grid={self.scene_id_grid is not None}, "
                                          f"scene_confidence_grid={self.scene_confidence_grid is not None}")
            
            # ========== 通道1: 探索掩码 ==========
            # 基于通道4-13的语义信息计算：如果通道4-13中至少有一个有值，则认为是已探索
            # 这样可以处理"占据但场景属性未知"的情况，这种情况应该当作未知处理
            semantic_channels = channels[4:14, :, :]  # 提取通道4-13 (10个语义通道)
            # 对每个位置，检查10个语义通道中是否有非零值
            explored_mask = np.any(semantic_channels > 0, axis=0).astype(np.uint8) * 255
            channels[1] = explored_mask
            
            return channels, self.occupancy_grid.info
    
    def timer_callback(self, event):
        """定时器回调：转换并保存14通道地图为npz文件"""
        result = self.convert_to_14channel()
        if result is None:
            return
        
        channels, grid_info = result
        
        # 打印通道统计信息
        rospy.loginfo(f"14通道数据统计 (文件 #{self.file_counter}):")
        channel_names = [
            "occupancy", "explored_mask", "agent_position", "agent_history",
            "semantic_0", "semantic_1", "semantic_2", "semantic_3",
            "semantic_4", "semantic_5", "semantic_6", "semantic_7",
            "semantic_8", "semantic_9"
        ]
        for ch_idx in range(14):
            ch_data = channels[ch_idx]
            non_zero = np.count_nonzero(ch_data)
            max_val = ch_data.max()
            rospy.loginfo(f"  通道 {ch_idx} ({channel_names[ch_idx]}): 非零={non_zero}, 最大值={max_val}")
        
        # 生成文件名（只需序号，因为已经在时间戳文件夹中）
        filename = f"14channel_map_{self.file_counter:06d}.npz"
        filepath = os.path.join(self.save_dir, filename)
        
        # 保存为npz文件
        # channels: (14, H, W) uint8数组
        # 同时保存地图元信息
        np.savez_compressed(
            filepath,
            channels=channels,  # (14, H, W) uint8
            width=grid_info.width,
            height=grid_info.height,
            resolution=grid_info.resolution,
            origin_x=grid_info.origin.position.x,
            origin_y=grid_info.origin.position.y,
            origin_z=grid_info.origin.position.z,
            map_frame=self.map_frame
        )
        
        # 累积时间维度数据：保存完整的14通道数据
        full_channels = channels.copy()  # (14, H, W)
        
        # 初始化时间维度数组（第一次）
        if not self.timeline_initialized:
            self.timeline_height = grid_info.height
            self.timeline_width = grid_info.width
            self.timeline_initialized = True
            rospy.loginfo(f"初始化时间维度数组: (T, 14, {self.timeline_height}, {self.timeline_width})")
            # 添加第一帧数据
            self.semantic_timeline.append(full_channels)
        else:
            # 检查尺寸是否一致
            if full_channels.shape[1] != self.timeline_height or full_channels.shape[2] != self.timeline_width:
                rospy.logwarn(f"通道尺寸不匹配: 期望 ({self.timeline_height}, {self.timeline_width}), "
                            f"实际 ({full_channels.shape[1]}, {full_channels.shape[2]})，跳过本次累积")
            else:
                self.semantic_timeline.append(full_channels)
        
        self.file_counter += 1
        rospy.loginfo(f"已保存14通道地图到: {filepath}, 尺寸: {grid_info.width}x{grid_info.height}, "
                     f"时间维度累积: {len(self.semantic_timeline)} 帧")
    
    def periodic_save_timeline(self, event):
        """定期保存时间维度数组"""
        if len(self.semantic_timeline) > 0:
            rospy.loginfo(f"定期保存时间维度数组: 当前累积 {len(self.semantic_timeline)} 帧")
            self.save_timeline()
    
    def save_timeline(self):
        """保存时间维度累积数组 (T, 14, H, W) 到raw_npz目录"""
        if not self.timeline_initialized or len(self.semantic_timeline) == 0:
            rospy.logwarn("时间维度数组未初始化或为空，跳过保存")
            return
        
        try:
            # 转换为numpy数组 (T, 14, H, W)
            timeline_array = np.array(self.semantic_timeline, dtype=np.uint8)
            T, num_channels, H, W = timeline_array.shape
            
            rospy.loginfo(f"保存时间维度数组: shape=({T}, {num_channels}, {H}, {W})")
            
            # 保存到raw_npz目录（不是实验子文件夹）
            base_save_dir = "/home/wxy/Downloads/Interactive-Nav-SG-nav/semantic_ws/data/raw_npz"
            os.makedirs(base_save_dir, exist_ok=True)
            
            # 使用实验时间戳作为文件名
            experiment_timestamp = os.path.basename(self.save_dir)
            timeline_filename = f"semantic_timeline_{experiment_timestamp}.npz"
            timeline_filepath = os.path.join(base_save_dir, timeline_filename)
            
            # 保存时间维度数组和元信息
            # 同时保存完整14通道和语义通道（10通道），以便兼容不同格式
            semantic_channels = timeline_array[:, 4:14, :, :]  # (T, 10, H, W) - 提取语义通道
            
            np.savez_compressed(
                timeline_filepath,
                maps=timeline_array,  # (T, 14, H, W) uint8 - 完整14通道
                semantic_timeline=semantic_channels,  # (T, 10, H, W) uint8 - 仅语义通道（向后兼容）
                T=T,
                num_channels=num_channels,
                num_classes=10,  # 语义类别数
                height=H,
                width=W,
                resolution=self.occupancy_grid.info.resolution if self.occupancy_grid else 0.1,
                origin_x=self.occupancy_grid.info.origin.position.x if self.occupancy_grid else 0.0,
                origin_y=self.occupancy_grid.info.origin.position.y if self.occupancy_grid else 0.0,
                map_frame=self.map_frame
            )
            
            rospy.loginfo(f"已保存时间维度数组到: {timeline_filepath}, shape=({T}, {num_channels}, {H}, {W})")
        except Exception as e:
            rospy.logerr(f"保存时间维度数组失败: {e}")


def main():
    try:
        converter = SceneTo14ChannelConverter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()

