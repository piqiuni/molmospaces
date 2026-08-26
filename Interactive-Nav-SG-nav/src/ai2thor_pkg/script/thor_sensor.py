#!/home/lsl/miniconda3/envs/smartllm/bin/python
"""
AI2-THOR传感器数据处理模块
处理RGB-D图像，生成点云和物体检测
"""

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


class ThorSensor:
    def __init__(self, controller, bridge):
        self.controller = controller
        self.bridge = bridge
        
    def depth_to_pointcloud(self, depth_image, rgb_image, position, rotation, initial_position, fov, prev_pose=None, 
                           camera_offset_x=0.0, camera_offset_y=0.0, camera_offset_z=0.0):
        """
        将深度图和RGB图转换为彩色点云（使用相对坐标系+旋转补偿+相机偏移）
        
        Args:
            depth_image: 深度图像 (H x W)
            rgb_image: RGB图像 (H x W x 3)
            position: 机器人位置 {'x', 'y', 'z'}
            rotation: 机器人旋转 {'x', 'y', 'z'}
            initial_position: 初始位置（坐标原点） {'x', 'y', 'z'}
            fov: 视场角（度）- 从AI2-THOR获取的实际FOV
            prev_pose: 上一帧的位姿 {'position', 'rotation'}，用于旋转补偿
            camera_offset_x: 相机相对机器人的X偏移（左右）
            camera_offset_y: 相机相对机器人的Y偏移（高度）
            camera_offset_z: 相机相对机器人的Z偏移（前后）
            
        Returns:
            local_pointcloud: 局部坐标系彩色点云
            global_pointcloud: 全局坐标系彩色点云（相对于初始位置）
        """
        height, width = depth_image.shape
        
        # 使用从AI2-THOR传入的实际FOV计算相机内参
        fx =  width / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
        fy =height / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
        cx = width / 2.0
        cy = height / 2.0
        
        # 生成像素坐标网格
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # ========== 步骤1: 像素坐标转相机坐标（标准OpenCV方法） ==========
        # 创建像素坐标（flatten）
        u_flat = u.flatten()
        v_flat = v.flatten()
        z_flat = depth_image.flatten()
        
        # 过滤无效深度
        valid = (z_flat > 0.1) & (z_flat < 10.0)
        u_valid = u_flat[valid]
        v_valid = v_flat[valid]
        z_valid = z_flat[valid]
        
        # 相机坐标系（标准OpenCV）：X-右，Y-下，Z-前（深度）
        x_cam = (u_valid - cx) * z_valid / fx
        y_cam = (v_valid - cy) * z_valid / fy
        z_cam = z_valid
        
        # 提取对应的RGB颜色
        colors = rgb_image[v_valid.astype(int), u_valid.astype(int)]  # (N, 3) RGB [0-255]
        
        # ========== 步骤2: 生成纯相机坐标点云（不进行旋转） ==========
        # 相机坐标系 → 机器人坐标系（不旋转，纯相对坐标）
        # AI2-THOR相机坐标系: X(右), Y(下), Z(前) → 机器人坐标系: X(前), Y(左), Z(上)
        rel_x = z_cam   # 相机Z（前）→ 机器人X（前进）
        rel_y = -x_cam  # 相机-X（左）→ 机器人Y（左侧）
        abs_z = -y_cam+1.6  # 相机-Y（上）→ 机器人Z（向上）
        
        # ========== 旋转补偿（减少跳变） ==========
        rot_compensation = 0.0
        # if prev_pose is not None:
        #     # 计算旋转变化量（yaw方向）
        #     rot_delta = rotation['y'] - prev_pose['rotation']['y']
        #     # 归一化到[-180, 180]
        #     if rot_delta > 180:
        #         rot_delta -= 360
        #     elif rot_delta < -180:
        #         rot_delta += 360
        #     # 简单的旋转补偿（基于旋转差异，限制在±0.05）
        #     rot_compensation = np.clip(rot_delta * 0.1, -0.05, 0.05)
        
        # 组装相对坐标点云（不进行旋转，纯相对坐标）
        points_global = np.column_stack([
            rel_x,  # 相对X坐标
            rel_y,  # 相对Y坐标
            abs_z   # 绝对Z坐标（高度）
        ])
        
        # 组装相机坐标点云（用于调试）
        points_cam = np.column_stack([x_cam, y_cam, z_cam])
        
        # 将RGB转换为float32格式 (归一化到0-1)
        colors_float = colors.astype(np.float32) / 255.0
        
        # 创建带颜色的点云数据 [x, y, z, r, g, b]
        points_cam_rgb = np.hstack([points_cam, colors_float])
        points_global_rgb = np.hstack([points_global, colors_float])
        
        # 创建点云消息
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        
        # 定义包含RGB的PointField
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('r', 12, PointField.FLOAT32, 1),
            PointField('g', 16, PointField.FLOAT32, 1),
            PointField('b', 20, PointField.FLOAT32, 1),
        ]
        
        # 局部彩色点云（相机坐标系）
        header.frame_id = "tf_frame_camera"
        local_pc = pc2.create_cloud(header, fields, points_cam_rgb)
        
        # 全局彩色点云（机器人坐标系）
        header.frame_id = "tf_frame_base_link"
        global_pc = pc2.create_cloud(header, fields, points_global_rgb)
        
        return local_pc, global_pc
        
    def detect_objects(self, event, initial_position):
        """
        从AI2-THOR事件中提取物体检测结果（使用相对坐标系）
        
        Args:
            event: AI2-THOR事件对象
            initial_position: 初始位置（坐标原点） {'x', 'y', 'z'}
        
        Returns:
            List[dict]: 检测结果列表
        """
        detection_results = []
        
        # 获取可见物体
        visible_objects = event.metadata.get('objects', [])
        
        for obj in visible_objects:
            if not obj['visible']:
                continue
                
            # 提取物体信息
            obj_type = obj['objectType']
            obj_id = obj['objectId']
            position = obj['position']
            
            # 计算物体在图像中的位置（用于置信度）
            # AI2-THOR提供axisAlignedBoundingBox
            bbox = obj.get('axisAlignedBoundingBox')
            if bbox:
                # 计算边界框中心
                center = bbox.get('center', position)
            else:
                center = position
            
            # 计算相对于初始位置的坐标
            rel_x = position['x']# - initial_position['x']
            rel_y = position['y']# - initial_position['y']
            rel_z = position['z']# - initial_position['z']
            
            # print("obj_type: %s" % obj_type);
            if obj_type == "Box":
                print("painting id: %s, position: %f, %f, %f" % (obj_id, rel_x, rel_y, rel_z));


            detection_result = {
                #'semantic_id': hash(obj_type) % 1000,  # 简单的ID映射
                'semantic_class': obj_type,
                'confidence': 0.95,  # AI2-THOR提供的是ground truth，置信度高
                'position': {
                    'x': float(rel_z),    # 相对Z → ROS X（前进）
                    'y': float(-rel_x),   # 相对-X → ROS Y（左侧）
                    'z': float(rel_y)     # 相对Y → ROS Z（高度）
                },
                
                'size': {
                    'length': 0.5,  # 默认尺寸，可以从bounding box计算
                    'width': 0.5,
                    'depth': 0.5,
                },
            }
            
            detection_results.append(detection_result)
        
        return detection_results
        
    def get_navigable_positions(self):
        """获取可导航位置"""
        event = self.controller.step(action="GetReachablePositions")
        return event.metadata["actionReturn"]


