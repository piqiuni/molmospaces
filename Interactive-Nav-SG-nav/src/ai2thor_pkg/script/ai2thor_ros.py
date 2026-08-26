#!/home/lsl/miniconda3/envs/smartllm/bin/python
"""
AI2-THOR ROS接口主节点（支持ProcTHOR-10K数据集）
负责控制AI2-THOR仿真器并发布传感器数据
"""
import base64
import rospy
import numpy as np
import cv2
import json
from pathlib import Path
import threading
import time
# 自定义模块
import sys
import os
import gzip

# 在导入prior之前设置默认缓存目录（可以从ROS参数覆盖）
DEFAULT_DATASET_CACHE_DIR = "/home"
LOCAL_DATASET_DIR = "/home"
os.makedirs(DEFAULT_DATASET_CACHE_DIR, exist_ok=True)

# 设置环境变量（在导入prior之前）
os.environ['HF_HOME'] = DEFAULT_DATASET_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = DEFAULT_DATASET_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = DEFAULT_DATASET_CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = DEFAULT_DATASET_CACHE_DIR

# AI2-THOR和ProcTHOR（现在导入，环境变量已设置）
import prior
try:
    from prior import LazyJsonDataset
except ImportError:
    LazyJsonDataset = None

from tqdm import tqdm
import ai2thor.controller

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
        
        # 使用rospy.loginfo或print（取决于是否在ROS环境中）
        try:
            rospy.loginfo(f"正在从本地加载 {split} split...")
            log_func = rospy.loginfo
        except:
            print(f"正在从本地加载 {split} split...")
            log_func = print
        
        with gzip.open(jsonl_file, "rt") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        
        data[split] = LazyJsonDataset(
            data=houses, dataset="procthor-dataset", split=split
        )
        log_func(f"{split} split加载完成: {len(houses)} 个场景")
    
    return prior.DatasetDict(**data)

# ROS消息
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import tf

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from thor_sensor import ThorSensor
from thor_controller import ThorController


class AI2THORROSBridge:
    def __init__(self):
        """初始化AI2-THOR ROS桥接器"""
        rospy.init_node('ai2thor_ros_bridge', anonymous=True)
        
        # 初始化参数
        self.bridge = CvBridge()
        self.load_config()
        
        # 记录初始位置作为坐标原点（用于相对坐标系）
        self.initial_position = None
        # 记录上一帧的位姿（用于旋转补偿）
        self.prev_pose = None
        # 相机相对机器人的偏移参数（从ROS参数读取）
        self.camera_offset_x = rospy.get_param('~camera_offset_x', 0.0)  # 相机左右偏移
        self.camera_offset_y = rospy.get_param('~camera_offset_y', 0.9)  # 相机高度偏移（AI2-THOR眼睛高度）
        self.camera_offset_z = rospy.get_param('~camera_offset_z', 0.0)  # 相机前后偏移
        
        rospy.loginfo(f"相机偏移参数: X={self.camera_offset_x:.2f}, Y={self.camera_offset_y:.2f}, Z={self.camera_offset_z:.2f}")
        
        # 自定义相机ID
        self.custom_camera_id = "custom_rgbd_camera"
        
        # 初始化AI2-THOR控制器
        rospy.loginfo("正在初始化AI2-THOR仿真器...")
        self.initialize_controller()
        
        # 初始化传感器和控制模块
        self.sensor = ThorSensor(self.controller, self.bridge)
        self.thor_ctrl = ThorController(self.controller)
        
        # 设置ROS发布者
        self.setup_publishers()
        
        # 设置ROS订阅者
        self.setup_subscribers()
        
        # 初始化tf发布器
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        # 坐标系名称配置（与 struct_mapping_pkg 兼容）
        self.map_frame = "tf_frame_map"
        self.odom_frame = "tf_frame_odom"
        self.base_frame = "tf_frame_base_link"
        self.camera_frame = "tf_frame_camera"
        
        rospy.loginfo(f"TF坐标系配置: {self.map_frame} → {self.odom_frame} → {self.base_frame} → {self.camera_frame}")
        
        # 仿真状态
        self.is_moving = False
        self.target_position = None
        
        rospy.loginfo("AI2-THOR ROS桥接器初始化完成!")

        self.cmd_queue = []  # 存储待执行命令 [(action, params), ...]
        self.cmd_queue_lock = threading.Lock()
        self.is_executing = False  # 是否正在执行命令
        self.cmd_thread = threading.Thread(target=self.process_cmd_queue)
        self.cmd_thread.daemon = True
        self.cmd_thread.start()
    
    def add_custom_camera(self):
        """
        添加自定义RGBD相机
        """
        try:
            # 获取当前机器人位置和旋转
            event = self.controller.step(action="Pass")
            agent_pos = event.metadata['agent']['position']
            agent_rot = event.metadata['agent']['rotation']
            
            # 计算相机位置（考虑偏移）
            camera_position = {
                'x': agent_pos['x'] + self.camera_offset_x,
                'y': agent_pos['y'] + self.camera_offset_y,
                'z': agent_pos['z'] + self.camera_offset_z
            }
            
            # 相机旋转（与机器人一致）
            camera_rotation = {
                'x': agent_rot['x'],
                'y': agent_rot['y'],
                'z': agent_rot['z']
            }
            
            # 添加自定义相机
            add_camera_event = self.controller.step(
                action="AddThirdPartyCamera",
                position=camera_position,
                rotation=camera_rotation,
                fieldOfView=self.fov
            )
            
            if add_camera_event.metadata.get('lastActionSuccess', False):
                rospy.loginfo(f"✅ 自定义RGBD相机添加成功: {self.custom_camera_id}")
                rospy.loginfo(f"   位置: {camera_position}")
                rospy.loginfo(f"   旋转: {camera_rotation}")
                rospy.loginfo(f"   分辨率: {self.camera_width}x{self.camera_height}")
                rospy.loginfo(f"   FOV: {self.fov}度")
            else:
                rospy.logwarn("❌ 自定义相机添加失败，使用默认相机")
                self.custom_camera_id = None
                
        except Exception as e:
            rospy.logerr(f"添加自定义相机时出错: {e}")
            self.custom_camera_id = None
    
    def load_config(self):
        """加载配置参数"""
        # 场景类型：procthor 或 floorplan
        self.scene_type = rospy.get_param('~scene_type', 'procthor')
        
        # ProcTHOR配置
        self.procthor_dataset = rospy.get_param('~procthor_dataset', 'procthor-10k')
        self.procthor_split = rospy.get_param('~procthor_split', 'train')
        self.procthor_index = rospy.get_param('~procthor_index', 112)
        
        # 数据集缓存目录配置（用于本地保存数据集）
        # 如果ROS参数提供了不同的路径，更新环境变量
        self.dataset_cache_dir = rospy.get_param('~dataset_cache_dir', DEFAULT_DATASET_CACHE_DIR)
        if self.dataset_cache_dir != DEFAULT_DATASET_CACHE_DIR:
            os.makedirs(self.dataset_cache_dir, exist_ok=True)
            # 更新环境变量（虽然prior已导入，但后续加载数据集时仍会使用）
            os.environ['HF_HOME'] = self.dataset_cache_dir
            os.environ['TRANSFORMERS_CACHE'] = self.dataset_cache_dir
            os.environ['HF_DATASETS_CACHE'] = self.dataset_cache_dir
            os.environ['HUGGINGFACE_HUB_CACHE'] = self.dataset_cache_dir
        else:
            self.dataset_cache_dir = DEFAULT_DATASET_CACHE_DIR
        rospy.loginfo(f"ProcTHOR数据集缓存目录: {self.dataset_cache_dir}")
        
        # FloorPlan配置
        self.floorplan_scene = rospy.get_param('~floorplan_scene', 'FloorPlan1')
        
        # 语义目标
        self.semantic_target = rospy.get_param('semantic_target_type', 'Microwave')
        
        # 相机参数
        self.camera_width = rospy.get_param('~camera_width', 720)
        self.camera_height = rospy.get_param('~camera_height', 720)
        self.fov = rospy.get_param('~fov', 90)
        
        # 深度参数
        self.depth_near = rospy.get_param('~depth_near', 0.1)
        self.depth_far = rospy.get_param('~depth_far', 10.0)
        
        # 机器人参数
        self.robot_height = rospy.get_param('~robot_height', 0.9)
        self.rotation_step = rospy.get_param('~rotation_step', 30)
        self.move_step = rospy.get_param('~move_step', 0.25)
        
        # 初始化选项
        self.random_spawn = rospy.get_param('~random_spawn', True)
        self.enable_top_view = rospy.get_param('~enable_top_view', True)
        
    def initialize_controller(self):
        """初始化AI2-THOR控制器"""
        if self.scene_type == 'procthor':
            # 加载ProcTHOR数据集（优先使用本地文件）
            rospy.loginfo(f"加载ProcTHOR数据集: {self.procthor_dataset}")
            
            # 优先使用本地数据文件
            local_dataset_dir = rospy.get_param('~local_dataset_dir', LOCAL_DATASET_DIR)
            if os.path.exists(local_dataset_dir) and os.path.exists(os.path.join(local_dataset_dir, "train.jsonl.gz")):
                rospy.loginfo(f"从本地目录加载数据集: {local_dataset_dir}")
                try:
                    dataset = load_dataset_from_local(local_dataset_dir)
                    rospy.loginfo("✓ 成功从本地文件加载数据集")
                except Exception as e:
                    rospy.logwarn(f"从本地加载失败: {e}，回退到使用prior库加载...")
                    dataset = prior.load_dataset(self.procthor_dataset)
            else:
                rospy.loginfo(f"本地数据文件不存在，使用prior库加载（会使用缓存）...")
                rospy.loginfo(f"数据集缓存位置: ~/.prior/datasets/")
                dataset = prior.load_dataset(self.procthor_dataset)
            
            house = dataset[self.procthor_split][self.procthor_index]
            rospy.loginfo(f"使用ProcTHOR场景: {self.procthor_split}[{self.procthor_index}]")
            
            # 创建控制器（启用可视化）
            self.controller = ai2thor.controller.Controller(
                scene=house,
                width=self.camera_width,
                height=self.camera_height,
                fieldOfView=self.fov,
                renderDepthImage=True,
                renderInstanceSegmentation=True,
                continuous=True,  # 启用连续动作模式，减少旋转跳变
                headless=False,  # 启用Unity窗口显示
                visibilityDistance=30.0,  # 可视距离
                makeAgentsVisible=True,  # 显示智能体
                gpu_device=0
            )
        else:
            # 使用传统FloorPlan场景（启用可视化）
            rospy.loginfo(f"使用FloorPlan场景: {self.floorplan_scene}")
            self.controller = ai2thor.controller.Controller(
                scene=self.floorplan_scene,
                width=self.camera_width,
                height=self.camera_height,
                fieldOfView=self.fov,
                renderDepthImage=True,
                renderInstanceSegmentation=True,
                continuous=True,  # 启用连续动作模式，减少旋转跳变
                headless=False,  # 启用Unity窗口显示
                visibilityDistance=30.0,
                makeAgentsVisible=True
            )
        
        # 初始化场景
        self.controller.step(action="Pass")
        
        # 添加自定义RGBD相机
        self.add_custom_camera()
        
        # 设置相机水平角度（默认是30度向下，我们设置为0度水平）
        event = self.controller.step(action="RotateLook", horizon=0)
        rospy.loginfo(f"相机俯仰角已设置为: {event.metadata['agent']['cameraHorizon']}度")
        
        # 添加俯视摄像头（如果启用）
        if self.enable_top_view:
            event = self.controller.step(action="GetMapViewCameraProperties")
            self.controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])
            rospy.loginfo("已启用俯视图")
        
        # 获取可达位置
        reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        self.reachable_positions = [(p["x"], p["y"], p["z"]) for p in reachable_positions]
        rospy.loginfo(f"找到 {len(self.reachable_positions)} 个可达位置")
        
        # 随机放置机器人（如果启用）
        if self.random_spawn and self.reachable_positions:
            import random
            init_pos = random.choice(reachable_positions)
            self.controller.step(dict(action="Teleport", position=init_pos, agentId=0))
            rospy.loginfo(f"机器人随机放置到: {init_pos}")
        
        # 记录初始位置作为坐标原点
        event = self.controller.step(action="Pass")
        self.initial_position = event.metadata['agent']['position'].copy()
        rospy.loginfo(f"初始位置（坐标原点）: AI2-THOR({self.initial_position['x']:.2f}, {self.initial_position['y']:.2f}, {self.initial_position['z']:.2f})")
        
    def setup_publishers(self):
        """设置ROS发布者"""
        # 图像发布
        self.rgb_pub = rospy.Publisher('/ai2thor/rgb_image', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('/ai2thor/depth_image', Image, queue_size=10)
        self.seg_pub = rospy.Publisher('/ai2thor/semantic_image', Image, queue_size=10)
        
        # 点云发布
        self.pointcloud_pub = rospy.Publisher('/registered_scan', PointCloud2, queue_size=10)
        # 
        
        
        # 里程计发布
        self.odom_pub = rospy.Publisher('/odometry', Odometry, queue_size=10)
        
        # 检测结果发布
        self.detect_pub = rospy.Publisher('/explore_agent/result_info', String, queue_size=10)
        self.obs_pub = rospy.Publisher('/ai2thor/habitat_obs', String, queue_size=10)
        # 俯视图发布（如果启用）
        if self.enable_top_view:
            self.top_view_pub = rospy.Publisher('/ai2thor/top_view', Image, queue_size=10)
        
    def setup_subscribers(self):
        """设置ROS订阅者"""
        # 速度控制订阅
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel_stamped', TwistStamped, self.cmd_vel_callback)
        
        # 目标点订阅
        self.target_sub = rospy.Subscriber('/explore_agent/explore_target', PointStamped, self.target_callback)
        
    def cmd_vel_callback(self, msg):
        """快速处理速度指令 - 只入队，不阻塞"""
        linear = msg.twist.linear.x
        angular = msg.twist.angular.z
        
        # 计算动作类型和参数
        action = None
        params = None
        dt = 0.5
        if abs(angular) > 0.3:
            if angular > 0:
                action = "rotate_left"
                params = abs(angular) * 20 * dt # 限制最大旋转角度
            else:
                action = "rotate_right"
                params = abs(angular) * 20 * dt
        elif abs(linear) > 0.05:
            if linear > 0:
                action = "move_forward"
                params = min(abs(linear)*dt, self.move_step) # 限制最大移动距离
            else:
                action = "move_back"
                params = min(abs(linear)*dt, self.move_step)
        
        if action:
            with self.cmd_queue_lock:
                # 清空旧命令，只保留最新命令
                self.cmd_queue.clear()
                self.cmd_queue.append((action, params))
                
            rospy.logdebug(f"接收到控制命令: {action}({params})")
    
    def process_cmd_queue(self):
        """在独立线程中处理控制命令队列"""
        while not rospy.is_shutdown():
            try:
                # 获取最新命令
                action = None
                params = None
                
                with self.cmd_queue_lock:
                    if self.cmd_queue and not self.is_executing:
                        action, params = self.cmd_queue.pop(0)
                        self.is_executing = True
                
                if action and params:
                    # 执行控制命令
                    self.execute_action(action, params)
                    
                    # 标记执行完成
                    with self.cmd_queue_lock:
                        self.is_executing = False
                else:
                    # 没有命令，短暂休眠
                    time.sleep(0.01)  # 10ms
                    
            except Exception as e:
                rospy.logerr(f"处理控制命令队列时出错: {e}")
                with self.cmd_queue_lock:
                    self.is_executing = False
                time.sleep(0.01)
    
    def execute_action(self, action, params):
        """执行单个控制动作"""
        try:
            if action == "rotate_left":
                self.thor_ctrl.rotate_left(params)
            elif action == "rotate_right":
                self.thor_ctrl.rotate_right(params)
            elif action == "move_forward":
                self.thor_ctrl.move_forward(params)
            elif action == "move_back":
                self.thor_ctrl.move_back(params)
                
            rospy.logdebug(f"执行完成: {action}({params})")
            
        except Exception as e:
            rospy.logerr(f"执行动作 {action} 时出错: {e}")
            
    def target_callback(self, msg):
        """处理探索目标点"""
        self.target_position = (msg.point.x, msg.point.y, msg.point.z)
        rospy.loginfo(f"收到目标点: {self.target_position}")
        
    def publish_sensor_data(self):
        """发布传感器数据"""
        # 检查初始位置是否已设置
        if self.initial_position is None:
            rospy.logwarn_once("初始位置尚未设置，跳过本次数据发布")
            return
        
        event = self.controller.last_event
        timestamp = rospy.Time.now()
        
        # 获取相机数据（优先使用自定义相机）
        rgb_image = None
        depth_image = None
        
        if self.custom_camera_id and self.custom_camera_id in event.metadata.get('thirdPartyCameras', {}):
            # 使用自定义相机数据
            camera_data = event.metadata['thirdPartyCameras'][self.custom_camera_id]
            rgb_image = camera_data.get('rgb', None)
            depth_image = camera_data.get('depth', None)
            # rospy.logdebug(f"使用自定义相机数据: {self.custom_camera_id}")
        else:
            # 回退到默认相机数据
            rgb_image = event.frame
            if hasattr(event, 'depth_frame') and event.depth_frame is not None:
                depth_image = event.depth_frame
            # rospy.logerr("使用默认相机数据")
        
        # 1. 发布RGB图像
        if rgb_image is not None:
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
            rgb_msg.header.stamp = timestamp
            rgb_msg.header.frame_id = "tf_frame_camera"
            self.rgb_pub.publish(rgb_msg)
        
        # 2. 发布深度图像和生成点云
        if depth_image is not None:
            # 转换为米单位（AI2-THOR深度已经是米）
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image.astype(np.float32), encoding="32FC1")
            depth_msg.header.stamp = timestamp
            depth_msg.header.frame_id = "tf_frame_camera"
            self.depth_pub.publish(depth_msg)
            
            # 3. 生成并发布彩色点云（使用相对坐标+旋转补偿+相机偏移）
            if rgb_image is not None:
                pointcloud_local, pointcloud_global = self.sensor.depth_to_pointcloud(
                    depth_image,
                    rgb_image,  # 传入RGB图像生成彩色点云
                    event.metadata['agent']['position'],
                    event.metadata['agent']['rotation'],
                    self.initial_position,  # 传入初始位置用于相对坐标转换
                    self.fov,  # 传入实际FOV，确保与AI2-THOR渲染一致
                    self.prev_pose,  # 传入上一帧位姿用于旋转补偿
                    self.camera_offset_x,  # 相机X偏移
                    self.camera_offset_y,  # 相机Y偏移（高度）
                    self.camera_offset_z   # 相机Z偏移
                )
                # self.local_pointcloud_pub.publish(pointcloud_local)
                self.pointcloud_pub.publish(pointcloud_global)
            
            # 更新上一帧位姿
            self.prev_pose = {
                'position': event.metadata['agent']['position'].copy(),
                'rotation': event.metadata['agent']['rotation'].copy()
            }
        
        # 4. 发布语义分割图像
        if hasattr(event, 'instance_segmentation_frame') and event.instance_segmentation_frame is not None:
            seg_image = event.instance_segmentation_frame
            seg_msg = self.bridge.cv2_to_imgmsg(seg_image, encoding="rgb8")
            seg_msg.header.stamp = timestamp
            seg_msg.header.frame_id = "tf_frame_camera"
            self.seg_pub.publish(seg_msg)

        # 发布 Habitat 风格 observation
        obs_str = json.dumps(self.build_habitat_obs(event))
        self.obs_pub.publish(obs_str)

        # 5. 发布里程计
        odom_msg = self.create_odom_msg(event.metadata, timestamp)
        self.odom_pub.publish(odom_msg)
        
        # 6. 发布 TF 变换链
        #.publish_map_odom_tf(timestamp)  # map → odom
        self.publish_base_link_tf(event.metadata, timestamp)  # odom → base_link
        self.publish_camera_tf(event.metadata, timestamp)  # base_link → camera

        # 7. 发布物体检测结果（使用相对坐标）
        detection_results = self.sensor.detect_objects(event, self.initial_position)
        if detection_results:
            detect_msg = String()
            detect_msg.data = json.dumps(detection_results)
            self.detect_pub.publish(detect_msg)
        
        # 8. 发布俯视图（如果启用）
        if self.enable_top_view and hasattr(event, 'events') and len(event.events) > 0:
            if hasattr(event.events[0], 'third_party_camera_frames') and len(event.events[0].third_party_camera_frames) > 0:
                top_view_frame = cv2.cvtColor(event.events[0].third_party_camera_frames[-1], cv2.COLOR_BGR2RGB)
                top_view_msg = self.bridge.cv2_to_imgmsg(top_view_frame, encoding="rgb8")
                top_view_msg.header.stamp = timestamp
                top_view_msg.header.frame_id = "tf_frame_map"
                self.top_view_pub.publish(top_view_msg)
        
    def create_odom_msg(self, metadata, timestamp):
        """创建里程计消息"""
        odom = Odometry()
        odom.header.stamp = timestamp
        odom.header.frame_id = self.odom_frame  # "tf_frame_odom"
        odom.child_frame_id = self.base_frame   # "tf_frame_base_link"
        
        # 获取位置（使用与点云相同的世界坐标系统）
        pos = metadata['agent']['position']
        print(pos)
        
        # 使用与点云相同的坐标转换：AI2-THOR → ROS世界坐标
        world_x = pos['x']   # AI2-THOR Z → ROS X（前进）
        world_y = pos['z']  # AI2-THOR -X → ROS Y（左侧）
        world_z = pos['y']   # AI2-THOR Y → ROS Z（高度）
        
        # 转换为相对坐标（与点云一致）
        odom.pose.pose.position.x = world_x
        odom.pose.pose.position.y = world_y
        odom.pose.pose.position.z = world_z

        #print(f"world_x: {world_x}, world_y: {world_y}, world_z: {world_z}")

        # 获取旋转（转换为四元数）
        # AI2-THOR yaw=0朝向+Z，转到ROS后应该朝向+X，需要逆时针旋转90度
        rot = metadata['agent']['rotation']
        cameraP   =  metadata['cameraPosition']
        cameraH    =  metadata['agent']['cameraHorizon']
        fov   =   metadata["fov"]
        #print(f"cameraPosition:{cameraP}")
        #print(f"rot:{rot}")
        #print(f"fov:{fov}")
        yaw = -np.deg2rad(rot['y'] - 90) # 在角度上加90度，与点云转换逻辑一致
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        
        return odom
        # ================== Habitat 格式 observation 构造器 ==================
    def build_habitat_obs(self, event):
        meta = event.metadata
        agent = meta['agent']
        rgb_image = None
        depth_image = None
        
        if self.custom_camera_id and self.custom_camera_id in event.metadata.get('thirdPartyCameras', {}):
            # 使用自定义相机数据
            camera_data = event.metadata['thirdPartyCameras'][self.custom_camera_id]
            rgb_image = camera_data.get('rgb', None)
            depth_image = camera_data.get('depth', None)
            # rospy.logerr(f"使用自定义相机数据: {self.custom_camera_id}")
        else:
            # 回退到默认相机数据
            rgb_image = event.frame
            if hasattr(event, 'depth_frame') and event.depth_frame is not None:
                depth_image = event.depth_frame
            # rospy.logerr("使用默认相机数据")
        # 获取自定义相机数据

        # --- RGB ---
        rgb_bytes = np.array(rgb_image).tobytes()
        rgb_b64 = base64.b64encode(rgb_bytes).decode('utf-8')

        # --- Depth ---
        depth_m = np.array(depth_image, dtype=np.float32)
        depth_bytes = depth_m.tobytes()
        depth_b64 = base64.b64encode(depth_bytes).decode('utf-8')

        if self.initial_position is None:
            gps = np.zeros(2, dtype=np.float32)
        else:
            gps = np.array([
                agent['position']['x'] - self.initial_position['x'],
                agent['position']['z'] - self.initial_position['z']
            ], dtype=np.float32)
        gps_list = gps.tolist()

        # --- compass (1,)  float32  弧度  0=+X ---
        yaw_deg = agent['rotation']['y']
        yaw_rad = np.deg2rad(-(yaw_deg - 90))  # 与 odom 保持一致
        compass = np.array([yaw_rad], dtype=np.float32).tolist()

        # --- objectgoal (int32)  这里固定 0，可扩展 ---
        objectgoal = 0

        # 打包成 dict（与 Habitat 键名完全一致）
        obs_dict = {
            "rgb": rgb_b64,
            "depth": depth_b64,
            "gps": gps_list,
            "compass": compass,
            "objectgoal": objectgoal
        }
        return obs_dict
    def publish_base_link_tf(self, metadata, timestamp):
        """发布 odom → base_link TF变换"""
        try:
            # 获取位置（使用与odom相同的坐标转换）
            pos = metadata['agent']['position']
            
            # 使用与odom相同的坐标转换：AI2-THOR → ROS世界坐标
            world_x = pos['x']   # AI2-THOR X → ROS X
            world_y = pos['z']   # AI2-THOR Z → ROS Y
            world_z = pos['y']   # AI2-THOR Y → ROS Z
            
            # 转换为相对坐标（与odom一致）
            base_x = world_x
            base_y = world_y
            base_z = world_z - 0.9  # 与odom中的z坐标一致
            
            # 获取旋转（使用与odom相同的旋转转换）
            rot = metadata['agent']['rotation']
            yaw = -np.deg2rad(rot['y'] - 90)  # 与odom中的旋转逻辑一致
            q = tf.transformations.quaternion_from_euler(0, 0, yaw)
            
            # 发布tf变换：odom -> base_link
            self.tf_broadcaster.sendTransform(
                (base_x, base_y, base_z),  # 平移
                q,  # 旋转
                timestamp,  # 时间戳
                self.base_frame,  # "tf_frame_base_link"
                self.odom_frame   # "tf_frame_odom"
            )
            
        except Exception as e:
            rospy.logerr(f"发布base_link tf时出错: {e}")
    
    # def publish_map_odom_tf(self, timestamp):
    #     """发布 map → odom 静态TF（假设无漂移）"""
    #     try:
    #         # 静态变换（无平移和旋转）
    #         self.tf_broadcaster.sendTransform(
    #             (0.0, 0.0, 0.0),  # 无平移
    #             (0.0, 0.0, 0.0, 1.0),  # 无旋转
    #             timestamp,
    #             self.odom_frame,  # "tf_frame_odom"
    #             self.map_frame    # "tf_frame_map"
    #         )
    #     except Exception as e:
    #         rospy.logerr(f"发布map→odom tf时出错: {e}")
    
    def publish_camera_tf(self, metadata, timestamp):
        """发布 camera 的TF（base_link -> camera），相机在机器人正上方1.6m，无相对旋转"""
        try:
            # 相机相对于 base_link 的固定外参
            camera_x = 0.0
            camera_y = 0.0
            camera_z = 1.6

            # 无相对旋转
            q = (0.0, 0.0, 0.0, 1.0)

            # 发布TF：base_link -> camera
            self.tf_broadcaster.sendTransform(
                (camera_x, camera_y, camera_z),
                q,
                timestamp,
                self.camera_frame,  # "tf_frame_camera"
                self.base_frame     # "tf_frame_base_link"
            )
        except Exception as e:
            rospy.logerr(f"发布camera tf时出错: {e}")
    
    def run(self):
        """主循环 - 优化版"""
        rate = rospy.Rate(30)  # 提高控制频率到30Hz
        
        # 添加计数器，控制传感器发布频率
        sensor_pub_count = 0
        sensor_pub_interval = 3  # 每3次循环发布1次传感器数据 (≈10Hz)
        
        while not rospy.is_shutdown():
            # 实时处理控制指令（每次循环都处理）
            self.process_pending_commands()
            
            # 降低传感器数据发布频率
            sensor_pub_count += 1
            if sensor_pub_count >= sensor_pub_interval:
                self.publish_sensor_data()
                sensor_pub_count = 0
            
            rate.sleep()

    def process_pending_commands(self):
        """处理积压的速度指令"""
        # 检查是否有未处理的命令（如果有积压）
        rospy.sleep(0.001)  # 微小延迟，让回调函数有机会处理


if __name__ == '__main__':
    try:
        bridge = AI2THORROSBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        pass