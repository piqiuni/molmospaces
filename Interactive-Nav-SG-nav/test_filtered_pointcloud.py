#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试过滤后点云发布的脚本
用于验证 GridMapper 中的过滤后点云发布功能
"""

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def pointcloud_callback(msg):
    """接收过滤后的点云数据"""
    print("收到过滤后的点云:")
    print(f"  帧ID: {msg.header.frame_id}")
    print(f"  时间戳: {msg.header.stamp}")
    print(f"  点数: {msg.width}")
    print(f"  高度: {msg.height}")
    print(f"  点步长: {msg.point_step}")
    print(f"  行步长: {msg.row_step}")
    
    # 统计点云中的点数量
    point_count = 0
    for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        point_count += 1
    
    print(f"  实际点数: {point_count}")
    print("-" * 50)

def main():
    """主函数"""
    rospy.init_node('test_filtered_pointcloud', anonymous=True)
    
    print("开始监听过滤后的点云话题: /filtered_pointcloud")
    print("等待数据...")
    
    # 订阅过滤后的点云话题
    rospy.Subscriber('/filtered_pointcloud', PointCloud2, pointcloud_callback)
    
    # 保持节点运行
    rospy.spin()

if __name__ == '__main__':
    main()
