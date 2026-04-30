#!/home/lsl/miniconda3/envs/smartllm/bin/python

"""
AI2-THOR控制器模块
处理机器人移动和旋转指令
"""

import rospy
import base64

class ThorController:
    def __init__(self, controller):
        self.controller = controller
        
    def move_forward(self, distance=0.25):
        """前进指定距离"""
        event = self.controller.step(
            action="MoveAhead",
            moveMagnitude=distance
        )
        return event.metadata['lastActionSuccess']
        
    def move_back(self, distance=0.25):
        """后退指定距离"""
        event = self.controller.step(
            action="MoveBack",
            moveMagnitude=distance
        )
        return event.metadata['lastActionSuccess']
        
    def rotate_left(self, degrees=30):
        """左转指定角度"""
        event = self.controller.step(
            action="RotateLeft",
            degrees=degrees
        )
        return event.metadata['lastActionSuccess']
        
    def rotate_right(self, degrees=30):
        """右转指定角度"""
        event = self.controller.step(
            action="RotateRight",
            degrees=degrees
        )
        return event.metadata['lastActionSuccess']
        
    def teleport(self, x, y, z, rotation=0):
        """传送到指定位置"""
        event = self.controller.step(
            action="Teleport",
            position=dict(x=x, y=y, z=z),
            rotation=dict(x=0, y=rotation, z=0)
        )
        return event.metadata['lastActionSuccess']
        
    def look_up(self, degrees=30):
        """抬头"""
        event = self.controller.step(
            action="LookUp",
            degrees=degrees
        )
        return event.metadata['lastActionSuccess']
        
    def look_down(self, degrees=30):
        """低头"""
        event = self.controller.step(
            action="LookDown",
            degrees=degrees
        )
        return event.metadata['lastActionSuccess']


