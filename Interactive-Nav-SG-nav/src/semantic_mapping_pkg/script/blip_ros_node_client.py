#!/usr/bin/env python
"""
BLIP ROS节点 - 客户端版本
通过HTTP客户端调用远程BLIP服务，避免本地库冲突
"""

import rospy
import tempfile
import os
import re
import json
import time
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2

# 导入BLIP客户端
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from blip_client import BlipClient

class BLIPROSNodeClient:
    def __init__(self):
        """初始化BLIP ROS节点（客户端版本）"""
        # ROS节点初始化
        rospy.init_node('blip_ros_node_client', anonymous=True)
        self.bridge = CvBridge()
        
        # 获取参数
        self.server_url = rospy.get_param("blip_server_url", "http://127.0.0.1:5000")
        self.image_topic = rospy.get_param("blip_image_topic", "/ai2thor/rgb_image")
        self.output_topic = rospy.get_param("blip_output_topic", "/semantic_mapping/scene_attribute")
        self.question = rospy.get_param("blip_question", "")  # 如果为空，则生成图像描述
        self.max_new_tokens = rospy.get_param("blip_max_new_tokens", 50)
        self.timeout = rospy.get_param("blip_timeout", 60)
        self.connect_timeout = rospy.get_param("blip_connect_timeout", 5)
        
        # 初始化BLIP客户端
        self.client = BlipClient(
            server_url=self.server_url,
            timeout=self.timeout,
            connect_timeout=self.connect_timeout
        )
        
        # 检查服务器健康状态
        rospy.loginfo(f"检查BLIP服务器状态: {self.server_url}")
        health = self.client.check_health()
        if health.get("status") != "healthy":
            rospy.logwarn(f"BLIP服务器不可用: {health.get('error', '未知错误')}")
            rospy.logwarn("节点将继续运行，但请求可能失败")
            rospy.logwarn("请确保BLIP服务器已启动（运行 start_blip_server.sh）")
        else:
            rospy.loginfo("BLIP服务器状态正常")
        
        # ROS发布者和订阅者
        self.image_sub = rospy.Subscriber(self.image_topic, ROSImage, self.image_callback, queue_size=1)
        self.output_pub = rospy.Publisher(self.output_topic, String, queue_size=10)
        
        rospy.loginfo(f"BLIP ROS节点（客户端）已启动")
        rospy.loginfo(f"  服务器地址: {self.server_url}")
        rospy.loginfo(f"  订阅图片话题: {self.image_topic}")
        rospy.loginfo(f"  发布结果话题: {self.output_topic}")
        if self.question:
            rospy.loginfo(f"  视觉问答模式，问题: {self.question}")
        else:
            rospy.loginfo(f"  图像描述模式")
        
        # 处理标志（防止并发处理）
        self.processing = False
        
        # 创建临时目录用于存储图片
        self.temp_dir = tempfile.mkdtemp(prefix="blip_ros_client_")
        rospy.loginfo(f"临时文件目录: {self.temp_dir}")
    
    def preprocess_scene_attribute(self, raw_output):
        """
        预处理BLIP输出的场景属性（简化版本）
        
        只做基本清理，场景类型标准化由SceneSemanticMapper统一处理
        
        处理内容：
        1. 去除首尾空格
        2. 转小写
        3. 提取第一个单词（去除标点符号）
        
        参数:
            raw_output: BLIP原始输出字符串
        
        返回:
            处理后的场景属性字符串（后续由SceneSemanticMapper标准化）
        """
        if not raw_output:
            return ""
        
        # 1. 去除首尾空格，转为小写
        processed = raw_output.strip().lower()
        
        # 2. 提取第一个单词（去除标点符号和空格）
        # 使用正则表达式提取第一个连续的字母
        match = re.match(r'^([a-z]+)', processed)
        if match:
            processed = match.group(1)
        else:
            # 如果没有匹配到单词，返回空字符串
            return ""
        
        # 注意：同义词映射和标准化现在由SceneSemanticMapper统一处理
        # 这里只做基本清理，保持原始输出
        return processed
    
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
            rospy.loginfo_throttle(10.0, f"[BLIP] Received image with timestamp: {original_timestamp:.3f}")
            
            if image_timestamp.to_sec() == 0 or image_timestamp.to_sec() < 1000000000:  # 1970年或更早，使用当前时间
                image_timestamp = rospy.Time.now()
                rospy.logwarn_throttle(5.0, f"[BLIP] 图片时间戳无效（原始值={original_timestamp:.3f}，year 1970），使用当前时间: {image_timestamp.to_sec():.3f}")
            else:
                rospy.loginfo_throttle(10.0, f"[BLIP] 图片时间戳有效: {image_timestamp.to_sec():.3f}")
            
            # 转换ROS Image到OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            
            # 保存为临时文件
            temp_image_path = os.path.join(self.temp_dir, f"image_{image_timestamp.to_nsec()}.jpg")
            cv2.imwrite(temp_image_path, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
            
            # 调用BLIP服务
            raw_result = self.call_blip_service(temp_image_path)
            
            # 删除临时文件
            try:
                os.remove(temp_image_path)
            except:
                pass
            
            if raw_result:
                # 预处理结果
                processed_result = self.preprocess_scene_attribute(raw_result)
                
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
                    
                    # 调试输出：检查发布的时间戳（年月日格式）
                    time_str = time.strftime("%Y年%m月%d日", time.localtime(image_timestamp.to_sec()))
                    rospy.loginfo_throttle(5.0, f"[BLIP] Publishing scene attribute: '{processed_result}', timestamp: {time_str}")
                    rospy.logdebug_throttle(10.0, f"[BLIP] JSON output: {output_json}")
                    
                    self.output_pub.publish(output_msg)
                else:
                    rospy.logwarn(f"BLIP输出预处理后为空: {raw_result}")
            else:
                rospy.logwarn("BLIP推理返回空结果")
            
        except Exception as e:
            rospy.logerr(f"处理图片时出错: {e}")
        finally:
            self.processing = False
    
    def call_blip_service(self, image_path):
        """调用BLIP服务（通过HTTP客户端）"""
        try:
            if self.question:
                # 视觉问答模式
                result = self.client.visual_qa(
                    image_path,
                    self.question,
                    max_new_tokens=self.max_new_tokens
                )
                
                if result.get("success"):
                    return result.get("answer", "")
                else:
                    error_msg = result.get('error', '未知错误')
                    # 如果是连接错误，使用throttle减少日志输出
                    if "无法连接到服务器" in error_msg or "ConnectionError" in str(error_msg):
                        rospy.logerr_throttle(5.0, f"BLIP视觉问答失败: {error_msg}")
                    else:
                        rospy.logerr(f"BLIP视觉问答失败: {error_msg}")
                    return None
            else:
                # 图像描述模式
                result = self.client.image_caption(
                    image_path,
                    max_new_tokens=self.max_new_tokens
                )
                
                if result.get("success"):
                    return result.get("caption", "")
                else:
                    error_msg = result.get('error', '未知错误')
                    # 如果是连接错误，使用throttle减少日志输出
                    if "无法连接到服务器" in error_msg or "ConnectionError" in str(error_msg):
                        rospy.logerr_throttle(5.0, f"BLIP图像描述失败: {error_msg}")
                    else:
                        rospy.logerr(f"BLIP图像描述失败: {error_msg}")
                    return None
                    
        except Exception as e:
            rospy.logerr(f"调用BLIP服务时出错: {e}")
            return None
    
    def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except:
            pass

def main():
    try:
        node = BLIPROSNodeClient()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"节点运行出错: {e}")
    finally:
        # 清理临时文件
        if 'node' in locals():
            node.cleanup()

if __name__ == '__main__':
    main()

