#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BLIP ROS节点模拟脚本
用于测试scenemapper功能，不加载BLIP模型，避免显存问题
接口与blip_ros_node_client.py相同（但不调用远程服务，而是返回模拟结果）
"""

import rospy
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import random
import time
import re
import json

class BLIPROSNodeMock:
    def __init__(self):
        """初始化BLIP ROS节点（模拟版本）"""
        # ROS节点初始化
        rospy.init_node('blip_ros_node_mock', anonymous=True)
        self.bridge = CvBridge()
        
        # 获取参数（从全局命名空间读取，因为参数通过rosparam加载）
        self.image_topic = rospy.get_param("blip_image_topic", "/habitat/rgb_image")
        self.output_topic = rospy.get_param("blip_output_topic", "/semantic_mapping/scene_attribute")
        self.question = rospy.get_param("blip_question", "")  # 如果为空，则生成图像描述
        
        # 模拟场景属性列表（可以根据需要修改）
        self.scene_attributes = [
            "kitchen",
            "living room",
            "bedroom",
            "bathroom",
            "dining room",
            "office",
            "corridor",
            "hallway",
            "staircase",
            "entrance"
        ]
        
        # 模拟模式：random（随机）、sequential（顺序）、fixed（固定）
        self.mock_mode = rospy.get_param("blip_mock_mode", "random")
        self.fixed_scene = rospy.get_param("blip_mock_fixed_scene", "kitchen")
        
        # ROS发布者和订阅者
        self.image_sub = rospy.Subscriber(self.image_topic, ROSImage, self.image_callback, queue_size=1)
        self.output_pub = rospy.Publisher(self.output_topic, String, queue_size=10)
        
        rospy.loginfo(f"BLIP ROS节点（模拟版本）已启动")
        rospy.loginfo(f"  订阅图片话题: {self.image_topic}")
        rospy.loginfo(f"  发布结果话题: {self.output_topic}")
        rospy.loginfo(f"  模拟模式: {self.mock_mode}")
        if self.mock_mode == "fixed":
            rospy.loginfo(f"  固定场景属性: {self.fixed_scene}")
        if self.question:
            rospy.loginfo(f"  视觉问答模式（模拟），问题: {self.question}")
        else:
            rospy.loginfo(f"  图像描述模式（模拟）")
        
        # 处理标志（防止并发处理）
        self.processing = False
        
        # 顺序模式计数器
        self.sequential_index = 0
        
        # 统计信息
        self.image_count = 0
    
    def preprocess_scene_attribute(self, raw_output):
        """
        预处理BLIP输出的场景属性
        
        处理内容：
        1. 统一首字母大小写（转为小写）
        2. 统一类似表达（hallway/corridor, dinningroom/diningroom等）
        3. 去除多余空格
        4. 只取第一个单词（如果返回多个词）
        
        参数:
            raw_output: BLIP原始输出字符串
        
        返回:
            处理后的场景属性字符串
        """
        if not raw_output:
            return ""
        
        # 1. 去除首尾空格，转为小写
        processed = raw_output.strip().lower()
        
        # 2. 去除所有空格（处理"dinning room" -> "dinningroom"这种情况）
        processed = processed.replace(" ", "")
        
        # 3. 只取第一个单词（去除标点符号）
        # 使用正则表达式提取第一个连续的字母
        match = re.match(r'^([a-z]+)', processed)
        if match:
            processed = match.group(1)
        else:
            # 如果没有匹配到单词，返回空字符串
            return ""
        
        # 4. 统一类似表达
        # 定义同义词映射表（统一到标准名称）
        synonym_map = {
            # 走廊相关
            "hallway": "corridor",        # hallway统一为corridor
            "hall": "corridor",           # hall统一为corridor
            
            # 餐厅相关（注意：代码中已去除空格，所以"dining room"会变成"diningroom"）
            "diningroom": "dinningroom",  # diningroom统一为dinningroom（保持原有拼写）
            "dining": "dinningroom",      # dining简写统一为dinningroom
            
            # 客厅相关
            "living": "livingroom",       # living简写统一为livingroom
            "lounge": "livingroom",      # lounge统一为livingroom
            "sittingroom": "livingroom",  # sittingroom统一为livingroom
            "familyroom": "livingroom",   # familyroom统一为livingroom
            
            # 停车场相关
            "parking": "parkinglot",      # parking统一为parkinglot
            "carpark": "parkinglot",      # carpark统一为parkinglot
            
            # 浴室相关
            "restroom": "bathroom",       # restroom统一为bathroom
            "washroom": "bathroom",       # washroom统一为bathroom
            "toilet": "bathroom",        # toilet统一为bathroom
            
            # 储藏相关
            "storeroom": "storage",       # storeroom统一为storage
            "closet": "storage",          # closet统一为storage
            
            # 花园相关
            "yard": "garden",             # yard统一为garden
            "backyard": "garden",        # backyard统一为garden
            "frontyard": "garden",       # frontyard统一为garden
        }
        
        # 应用同义词映射
        if processed in synonym_map:
            processed = synonym_map[processed]
        
        return processed
    
    def get_mock_scene_attribute(self):
        """获取模拟的场景属性"""
        if self.mock_mode == "fixed":
            return self.fixed_scene
        elif self.mock_mode == "sequential":
            scene = self.scene_attributes[self.sequential_index % len(self.scene_attributes)]
            self.sequential_index += 1
            return scene
        else:  # random
            return random.choice(self.scene_attributes)
    
    def image_callback(self, msg):
        """图片回调函数"""
        if self.processing:
            rospy.logwarn_throttle(1.0, "正在处理上一张图片，跳过本次回调")
            return
        
        self.processing = True
        
        try:
            # 记录图片的时间戳，如果时间戳无效（为0或1970年），使用当前时间
            image_timestamp = msg.header.stamp
            original_timestamp = image_timestamp.to_sec()
            
            # 调试输出：检查接收到的图片时间戳
            rospy.logdebug_throttle(10.0, f"[模拟BLIP] Received image with timestamp: {original_timestamp:.3f}")
            
            if image_timestamp.to_sec() == 0 or image_timestamp.to_sec() < 1000000000:  # 1970年或更早，使用当前时间
                image_timestamp = rospy.Time.now()
                rospy.logwarn_throttle(5.0, f"[模拟BLIP] 图片时间戳无效（原始值={original_timestamp:.3f}，year 1970），使用当前时间: {image_timestamp.to_sec():.3f}")
            else:
                rospy.logdebug_throttle(10.0, f"[模拟BLIP] 图片时间戳有效: {image_timestamp.to_sec():.3f}")
            
            # 转换ROS Image到OpenCV格式（仅用于验证，不实际处理）
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            height, width = cv_image.shape[:2]
            
            self.image_count += 1
            
            # 模拟BLIP处理延迟（可选）
            # time.sleep(0.1)  # 模拟100ms处理时间
            
            # 生成模拟的场景属性
            if self.question:
                # 视觉问答模式：返回简短的答案
                raw_scene_attr = self.get_mock_scene_attribute()
            else:
                # 图像描述模式：返回场景属性
                raw_scene_attr = self.get_mock_scene_attribute()
            
            # 预处理结果
            processed_result = self.preprocess_scene_attribute(raw_scene_attr)
            
            if processed_result:
                # 发布处理后的结果，包含图片时间戳（JSON格式）
                # 使用 sec 和 nsec 字段（整数），而不是 to_sec() 和 to_nsec()，避免精度丢失
                output_data = {
                    "scene_attribute": processed_result,
                    "image_timestamp_sec": image_timestamp.secs,  # 秒部分（整数）
                    "image_timestamp_nsec": image_timestamp.nsecs  # 纳秒部分（整数）
                }
                output_msg = String()
                output_json = json.dumps(output_data)
                output_msg.data = output_json
                
                # 调试输出：检查发布的时间戳
                rospy.loginfo_throttle(5.0, f"[模拟BLIP] 图片#{self.image_count} ({width}x{height}) -> 场景属性: '{processed_result}', 时间戳: {image_timestamp.to_sec():.3f} (sec={image_timestamp.secs}, nsec={image_timestamp.nsecs})")
                rospy.logdebug_throttle(10.0, f"[模拟BLIP] JSON输出: {output_json}")
                
                self.output_pub.publish(output_msg)
            else:
                rospy.logwarn(f"[模拟BLIP] 场景属性预处理后为空: {raw_scene_attr}")
            
        except Exception as e:
            rospy.logerr(f"处理图片时出错: {e}")
        finally:
            self.processing = False

def main():
    try:
        node = BLIPROSNodeMock()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"节点运行出错: {e}")

if __name__ == '__main__':
    main()

