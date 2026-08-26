#!/home/lsl/miniconda3/envs/smartllm/bin/python
"""
键盘控制机器人移动
用于实时更新栅格占据地图
"""

import rospy
from geometry_msgs.msg import TwistStamped
import sys
import termios
import tty

class KeyboardController:
    def __init__(self):
        rospy.init_node('keyboard_controller', anonymous=True)
        
        # 发布速度指令
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel_stamped', TwistStamped, queue_size=1)
        
        # 运动参数
        self.linear_speed = 1.0   # 线速度
        self.angular_speed = 0.8  # 角速度
        
        print("\n" + "="*50)
        print("  键盘控制 - AI2-THOR机器人")
        print("="*50)
        print("\n控制键:")
        print("  w/i - 前进")
        print("  s/k - 后退")  
        print("  a/j - 左转")
        print("  d/l - 右转")
        print("  q/u - 停止")
        print("  x/m - 退出")
        print("\n速度调整:")
        print("  r - 增加速度")
        print("  f - 降低速度")
        print("\n当前速度: 线速度={:.1f}, 角速度={:.1f}".format(
            self.linear_speed, self.angular_speed))
        print("="*50 + "\n")
        
    def get_key(self):
        """获取键盘输入"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    
    def publish_velocity(self, linear_x, angular_z):
        """发布速度指令"""
        cmd = TwistStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.twist.linear.x = linear_x
        cmd.twist.angular.z = angular_z
        self.cmd_vel_pub.publish(cmd)
        
    def run(self):
        """主循环"""
        rate = rospy.Rate(10)  # 10Hz
        
        try:
            while not rospy.is_shutdown():
                key = self.get_key()
                
                linear_x = 0.0
                angular_z = 0.0
                
                # 运动控制
                if key in ['w', 'i']:  # 前进
                    linear_x = self.linear_speed
                    print("→ 前进")
                elif key in ['s', 'k']:  # 后退
                    linear_x = -self.linear_speed
                    print("← 后退")
                elif key in ['a', 'j']:  # 左转
                    angular_z = self.angular_speed
                    print("↶ 左转")
                elif key in ['d', 'l']:  # 右转
                    angular_z = -self.angular_speed
                    print("↷ 右转")
                elif key in ['q', 'u']:  # 停止
                    linear_x = 0.0
                    angular_z = 0.0
                    print("■ 停止")
                    
                # 速度调整
                elif key == 'r':  # 增加速度
                    self.linear_speed += 0.1
                    self.angular_speed += 0.1
                    print("⬆ 速度增加: 线速度={:.1f}, 角速度={:.1f}".format(
                        self.linear_speed, self.angular_speed))
                elif key == 'f':  # 降低速度
                    self.linear_speed = max(0.1, self.linear_speed - 0.1)
                    self.angular_speed = max(0.1, self.angular_speed - 0.1)
                    print("⬇ 速度降低: 线速度={:.1f}, 角速度={:.1f}".format(
                        self.linear_speed, self.angular_speed))
                    
                # 退出
                elif key in ['x', 'm', '\x03']:  # x, m 或 Ctrl+C
                    print("\n退出键盘控制")
                    break
                
                # 发布速度
                self.publish_velocity(linear_x, angular_z)
                
                rate.sleep()
                
        except Exception as e:
            print(f"\n错误: {e}")
        finally:
            # 退出时发送停止指令
            self.publish_velocity(0.0, 0.0)
            print("已停止机器人")

if __name__ == '__main__':
    try:
        controller = KeyboardController()
        controller.run()
    except rospy.ROSInterruptException:
        pass

