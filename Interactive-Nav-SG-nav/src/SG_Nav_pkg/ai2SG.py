#!/usr/bin/env python3
import rospy
import json
import base64
import numpy as np
from std_msgs.msg import String
import argparse
import copy
import math
import os
from matplotlib import colors
import cv2
import pandas
import skimage
import torch
import tf2_ros
import geometry_msgs.msg
from nav_msgs.msg import OccupancyGrid
import tf                     # ROS 1 tf
from tf import transformations as tft   # 方便后面写 tft.quaternion_matrix
import sys
# 获取项目根目录（SG_Nav_pkg 在 src/ 下，所以需要向上两级）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
# 添加项目根目录到路径（用于导入外部依赖）
sys.path.insert(0, project_root)
try:
    from .scenegraph import SceneGraph
    from visualization_msgs.msg import Marker, MarkerArray
    from .utils.utils_fmm import control_helper as CH
    from .utils.utils_fmm import pose_utils as pu
    from .utils.utils_fmm.fmm_planner import FMMPlanner    
    from .utils.utils_fmm.mapping import Semantic_Mapping
    from .utils.image_process import (
        add_resized_image,
        add_rectangle,
        add_text,
        add_text_list,
        crop_around_point,
        draw_agent,
        draw_goal,
        line_list
    )
except ImportError:
    # 直接运行脚本时的导入方式
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from SG_Nav_pkg.scenegraph import SceneGraph
    from visualization_msgs.msg import Marker, MarkerArray
    from SG_Nav_pkg.utils.utils_fmm import control_helper as CH
    from SG_Nav_pkg.utils.utils_fmm import pose_utils as pu
    from SG_Nav_pkg.utils.utils_fmm.fmm_planner import FMMPlanner    
    from SG_Nav_pkg.utils.utils_fmm.mapping import Semantic_Mapping
    from SG_Nav_pkg.utils.image_process import (
        add_resized_image,
        add_rectangle,
        add_text,
        add_text_list,
        crop_around_point,
        draw_agent,
        draw_goal,
        line_list
    )

class Visualization:
    def __init__(self):
        self.resolution = 0.05        # 5cm grid
        self.width = 800              # 40m x 40m
        self.height = 800
        self.origin_x = -20           # 左下角世界坐标
        self.origin_y = -20
        self.text_pub = rospy.Publisher("/object_labels", MarkerArray, queue_size=1)
        self.marker_id = 0
        self.markarray = np.zeros((self.height, self.width), dtype=np.int8)
        self.map_pub = rospy.Publisher("/object_map", OccupancyGrid, queue_size=1)

    def world_to_map(self, Pw):
        mx = int((Pw[0] - self.origin_x) / self.resolution)
        my = int((Pw[1] - self.origin_y) / self.resolution)

        if mx < 0 or mx >= self.width:  return None
        if my < 0 or my >= self.height: return None
        return mx, my

    def mark_object(self, Pw, radius=0.2, value=100):
        result = self.world_to_map(Pw)
        if result is None:
            return
        mx, my = result

        radius_cells = int(radius / self.resolution)

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx*dx + dy*dy <= radius_cells*radius_cells:
                    mx2 = mx + dx
                    my2 = my + dy
                    if 0 <= mx2 < self.width and 0 <= my2 < self.height:
                        self.markarray[my2][mx2] = value

    def create_text_marker(self, caption, Pw, color=(1.0,1.0,1.0)):
        """在物体世界坐标 Pw 上创建文本 Marker"""

        marker = Marker()
        marker.header.frame_id = "tf_frame_map"
        marker.header.stamp = rospy.Time.now()

        marker.ns = "object_labels"
        marker.id = self.marker_id
        self.marker_id += 1

        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        # 设置位置（略微抬起防止在地面以下）
        marker.pose.position.x = float(Pw[0])
        marker.pose.position.y = float(Pw[1])
        marker.pose.position.z = float(Pw[2]) + 0.4

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # 文本
        marker.text = caption

        # 字体大小（米）
        marker.scale.z = 0.3

        # 颜色（默认白色）
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0

        return marker

    def publish_labels(self, object_list):
        """
        object_list: list of dicts
        每个 dict 格式：
        { "caption": "chair", "world": np.array([x,y,z]) }
        """

        marker_array = MarkerArray()
        self.marker_id = 0  # reset every frame

        for obj in object_list:
            caption = obj["caption"]
            Pw = obj["world"]
            
            marker = self.create_text_marker(caption, Pw)
            marker_array.markers.append(marker)

        self.text_pub.publish(marker_array)

    def publish_object_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "tf_frame_map"

        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0

        msg.data = self.markarray.flatten().tolist()
        self.map_pub.publish(msg)

class TFListener(object):
    def __init__(self):
        self.tfl = tf.TransformListener()   # ROS 1 的监听器

    def get_transform(self, target_frame, source_frame):
        """
        查询 target_frame ← source_frame 的变换
        返回: 成功 -> (trans, rot) , 失败 -> None
        trans 是 [x, y, z], rot 是 [x, y, z, w]
        """
        try:
            # rospy.Time(0) 表示“最新可用”
            (trans, rot) = self.tfl.lookupTransform(
                target_frame, source_frame, rospy.Time(0)
            )
            return trans, rot
        except (tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException) as e:
            rospy.logwarn("TF lookup failed: %s", e)
            return None

    def transform_to_matrix(self, trans_rot_tuple):
        """
        把 get_transform 得到的 (trans, rot) 转成 4×4 numpy 矩阵
        """
        if trans_rot_tuple is None:
            return None
        trans, rot = trans_rot_tuple          # trans=[x,y,z]  rot=[x,y,z,w]
        mat = tft.quaternion_matrix(rot)      # 4×4
        mat[0, 3] = trans[0]
        mat[1, 3] = trans[1]
        mat[2, 3] = trans[2]
        return mat

# 定义相机内参类（封装 fx, fy, cx, cy）
class CameraIntrinsic:
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx  # x轴焦距
        self.fy = fy  # y轴焦距
        self.cx = cx  # 主点x坐标
        self.cy = cy  # 主点y坐标


class SimpleCameraK:
    def __init__(self, fx, cx, cy):
        self.f = fx
        self.xc = cx
        self.zc = cy

class SceneGraphNode:
    def __init__(self):
        self.tf_listener = TFListener()
        self.visual = Visualization() 
        # ==== 参数（SGNav 默认值） ====
        self.map_size_cm = 4000
        self.map_resolution = 5
        self.map_size = self.map_size_cm // self.map_resolution
        fx = fy = 720 / (2 * np.tan(np.deg2rad(90/2)))
        cx = 360
        cy = 360
        # 定义相机内参类（封装 fx, fy, cx, cy）
        self.camera_intrinsic = CameraIntrinsic(fx,fy,cx,cy)
        self.camera_matrix = SimpleCameraK(fx, cx, cy)
        # ==== 初始化 SceneGraph ====
        self.scenegraph = SceneGraph(
            map_resolution=self.map_resolution,
            map_size_cm=self.map_size_cm,
            map_size=self.map_size,
            camera_matrix=self.camera_matrix,
            agent=self      # 允许 SceneGraph 调用 agent.* 字段
        )

        # SceneGraph 需要的字段（最小）
        self.observations = None
        self.image_rgb = None
        self.image_depth = None
        self.pose_matrix = None

        rospy.Subscriber(
            "/ai2thor/habitat_obs",
            String,
            self.obs_callback,
            queue_size=10
        )

        print("SceneGraph ROS Node Ready.")

    # ----------------------------------------------------------------------
    #   这里完全使用 SGNav 内部同款 set_observations 逻辑
    # ----------------------------------------------------------------------
    def set_observations(self, observations):
        self.observations = observations
        self.image_rgb = observations["rgb"].copy()
        self.image_depth = observations["depth"].copy()
        self.pose_matrix = self.get_pose_matrix()

    def bbox_center_to_world(self,bbox_center, depth_map, camera_intrinsic, pose_matrix):
        """
        Convert bbox center (cx, cy) in pixel into world 3D coordinate.
        
        bbox_center: (cx, cy) in pixels
        depth_map: depth image in meters, shape (H, W)
        camera_intrinsic: object with fx, fy, cx, cy attributes
        pose_matrix: 4x4 numpy array (camera-to-world transform)
        
        return:
            Pc: 3D point in camera frame
            Pw: 3D point in world frame
        """
        cx, cy = bbox_center

        # 1. 获取深度 (米)
        depth = depth_map[int(cy), int(cx)]
        print(depth)
        if depth <= 0:
            return None, None  # 无效深度

        # 2. 像素坐标 → 相机坐标系（meters）
        Xc_original = (cx - camera_intrinsic.cx) * depth / camera_intrinsic.fx
        Yc_original = (cy - camera_intrinsic.cy) * depth / camera_intrinsic.fy
        Zc_original = depth
        Xc = Zc_original       # 相机前 → 地图前（X）
        Yc = -Xc_original      # 相机右 → 地图左（Y，反向）
        Zc = -Yc_original      # 相机下 → 地图上（Z，反向）
        Pc = np.array([Xc, Yc, Zc, 1.0])  # 齐次坐标

        # 3. 相机坐标系 → 世界坐标系
        Pw = pose_matrix @ Pc   # matrix multiply

        return Pc[:3], Pw[:3]   # 返回非齐次坐标

    # ----------------------------------------------------------------------
    def get_pose_matrix(self):
        gps = self.observations["gps"]
        compass = self.observations["compass"]

        x = self.map_size_cm / 100.0 / 2.0 + gps[0]
        y = self.map_size_cm / 100.0 / 2.0 - gps[1]
        t = (compass - np.pi / 2)[0]

        pose_matrix = np.array([
            [np.cos(t), -np.sin(t), 0, x],
            [np.sin(t),  np.cos(t), 0, y],
            [0,          0,         1, 0],
            [0,          0,         0, 1],
        ])
        return pose_matrix

    # ----------------------------------------------------------------------
    def obs_callback(self, msg):
        data = json.loads(msg.data)

        # ============== 解码 RGB ==============
        rgb_raw = np.frombuffer(base64.b64decode(data["rgb"]), dtype=np.uint8)
        L = rgb_raw.size

        # auto detect shape
        if L == 480*640*3:
            rgb = rgb_raw.reshape(480, 640, 3)
        elif L == 720*720*3:
            rgb = rgb_raw.reshape(720, 720, 3) 

        # ============== 解码 Depth ==============
        depth = np.frombuffer(base64.b64decode(data["depth"]), dtype=np.float32)
        # L = depth_raw.size

        depth = depth.reshape(720, 720, 1)   # SG 期望 (H,W,1)

        # ============== GPS / Compass ==============
        gps = np.array(data["gps"], dtype=np.float32)
        compass = np.array(data["compass"], dtype=np.float32)

        # ============== 构造 Habitat Obs（原样格式） ==============
        observations = {
            "rgb": rgb,
            "depth": depth,
            "gps": gps,
            "compass": compass
        }

        # ============== 使用 SceneGraph 自己的 set_observations ==============
        self.set_observations(observations)

        # ============== 调用 SceneGraph ==============
        self.scenegraph.set_agent(self)                 # SG 内部需要 agent.*
        self.scenegraph.set_observations(observations)  # 直接用 SG 原生函数
        self.scenegraph.update_scenegraph()
        tf_msg = self.tf_listener.get_transform("tf_frame_map", "tf_frame_camera")
        if tf_msg is not None:
            pose_matrix_to_camera = self.tf_listener.transform_to_matrix(tf_msg)
        else:
            return
        objects = self.scenegraph.get_seg_info_with_caption()
        if not objects:
            print("No objects detected.")
            return
        object_list = []
        for obj in objects:
            caption = obj["caption"]
            cx, cy = obj["center"]
            
            Pc, Pw = self.bbox_center_to_world(
                bbox_center=(cx, cy),
                depth_map=self.image_depth[:, :, 0],
                camera_intrinsic=self.camera_intrinsic,
                pose_matrix=pose_matrix_to_camera  # 来自 TF，而不是 odom topic 或 SceneGraph
            )
            print(f"{obj['caption']:10s} | image={str(obj['bbox']):10s} | world=({Pw})")
            # 标记在栅格地图
            self.visual.mark_object(Pw, radius=0.3, value=100)

            # 收集文本标签
            object_list.append({
                "caption": caption,
                "world": Pw
            })

        # 发布文字标签
        self.visual.publish_labels(object_list)

        # 发布栅格地图
        self.visual.publish_object_map()

        # ============== 打印输出 ==============
        text = self.scenegraph.graph_to_text(
            self.scenegraph.get_nodes(),
            self.scenegraph.get_edges()
        )
        print(text)



if __name__ == "__main__":
    rospy.init_node("scenegraph_builder")
    SceneGraphNode()
    rospy.spin()
