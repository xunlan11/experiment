#!/usr/bin/env python3
import rospy
import time  # 添加time模块导入
from std_msgs.msg import Float64

class MultiJointControl:
    def __init__(self):
        rospy.on_shutdown(self.cleanup)
        self.max_runtime = 10  # 设置最大运行时间为10秒
        self.start_time = time.time()  # 记录启动时间
        
        # 初始化5个关节的发布者
        self.joint_publishers = [
            rospy.Publisher(f"/motor{i}_controller/command", Float64, queue_size=10)
            for i in range(1, 6)  # 生成 motor1~motor5 的发布者
        ]
        
        # 等待发布者注册
        rospy.sleep(0.5)
        
        # 添加时间监控定时器
        self.timer = rospy.Timer(rospy.Duration(0.1), self.check_runtime)
        
        # 执行运动序列
        self.execute_motion_sequence()
    
    def check_runtime(self, event):
        """检查是否超过最大运行时间"""
        elapsed = time.time() - self.start_time
        if elapsed > self.max_runtime:
            rospy.loginfo(f"已达到最大运行时间 {self.max_runtime} 秒，停止程序")
            self.timer.shutdown()  # 停止定时器
            rospy.signal_shutdown("最大运行时间到达")  # 关闭ROS节点
    
    def execute_motion_sequence(self):
        """执行完整的运动序列"""
        self.move_to_first_positions()
        self.move_to_initial_positions()
        self.move_joint5_additional()
        self.move_to_target_positions()
    
    def move_to_first_positions(self):
        """移动到第一位置"""
        target_positions = [-1.5, -1, 1.1, 0.5, 2]
        self._publish_each_joint(target_positions)
        rospy.loginfo("Moving joints to target positions:")
        for i, pos in enumerate(target_positions, 1):
            rospy.loginfo(f"  Joint {i}: {pos} radians")
        rospy.sleep(1.0)
    
    def move_to_initial_positions(self):
        """移动到初始位置"""
        initial_positions = [-1.5, -1.6, 1.1, 0.5, 2]
        self._publish_each_joint(initial_positions)
        rospy.loginfo("Moving joints to initial positions:")
        for i, pos in enumerate(initial_positions, 1):
            rospy.loginfo(f"  Joint {i}: {pos} radians")
        rospy.sleep(1.0)
    
    def move_to_target_positions(self):
        """移动到目标位置"""
        target_positions = [-1.5, -1, 1.1, 0.5, 0]
        self._publish_each_joint(target_positions)
        rospy.loginfo("Moving joints to target positions:")
        for i, pos in enumerate(target_positions, 1):
            rospy.loginfo(f"  Joint {i}: {pos} radians")
        rospy.sleep(1.0)
    
    def move_joint5_additional(self):
        """第五关节额外运动"""
        rospy.loginfo("Preparing additional movement for Joint 5...")
        rospy.sleep(1.0)
        
        # 第五关节新目标位置
        joint5_new_position = 0
        msg = Float64()
        msg.data = joint5_new_position
        self.joint_publishers[4].publish(msg)
        
        rospy.loginfo(f"Joint 5 moving to additional position: {joint5_new_position} radians")
        rospy.sleep(2.0)
    
    def _publish_each_joint(self, positions):
        """分别发布5个关节的不同位置指令"""
        for pub, pos in zip(self.joint_publishers, positions):
            msg = Float64()
            msg.data = pos
            pub.publish(msg)
    
    def cleanup(self):
        """关闭节点时的清理工作"""
        rospy.loginfo("Shutting down...")

if __name__ == "__main__":
    rospy.init_node("multi_joint_control")
    try:
        controller = MultiJointControl()
        rospy.spin()  # 保持节点活跃
    except rospy.ROSInterruptException:
        pass