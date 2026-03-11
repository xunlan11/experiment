#!/usr/bin/env python
import rospy
import time
from std_msgs.msg import Float64

class MultiJointControl:
    def __init__(self):
        self.max_runtime = 10  # 最大运行时间
        self.start_time = time.time()  # 启动时间
        
        # 初始化5个关节的发布者
        self.joint_publishers = [
            rospy.Publisher(f"/motor{i}_controller/command", Float64, queue_size=10)
            for i in range(1, 6)
        ]
        rospy.sleep(0.5)
        
        # 添加时间监控定时器
        self.timer = rospy.Timer(rospy.Duration(0.1), self.check_runtime)
        
        # 执行运动序列
        self.execute_motion_sequence()
    
    # 检查是否超过最大运行时间
    def check_runtime(self, event):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_runtime:
            rospy.loginfo(f"已达到最大运行时间 {self.max_runtime} 秒，停止程序")
            self.timer.shutdown()  # 停止定时器
            rospy.signal_shutdown("最大运行时间到达")  # 关闭ROS节点
    
    # 执行完整的运动序列
    def execute_motion_sequence(self):
        self.move_to_first_positions()
        self.move_to_initial_positions()
        self.move_joint5_additional()
        self.move_to_target_positions()
    
    # 第一位置
    def move_to_first_positions(self):
        target_positions = [-1.5, -1, 1.1, 0.5, 2]
        self._publish_each_joint(target_positions)
        rospy.loginfo("Moving joints to target positions:")
        for i, pos in enumerate(target_positions, 1):
            rospy.loginfo(f"  Joint {i}: {pos} radians")
        rospy.sleep(1.0)
    
    # 初始位置
    def move_to_initial_positions(self):
        initial_positions = [-1.5, -1.6, 1.1, 0.5, 2]
        self._publish_each_joint(initial_positions)
        rospy.loginfo("Moving joints to initial positions:")
        for i, pos in enumerate(initial_positions, 1):
            rospy.loginfo(f"  Joint {i}: {pos} radians")
        rospy.sleep(1.0)
    
    # 目标位置
    def move_to_target_positions(self):
        target_positions = [-1.5, -1, 1.1, 0.5, 0]
        self._publish_each_joint(target_positions)
        rospy.loginfo("Moving joints to target positions:")
        for i, pos in enumerate(target_positions, 1):
            rospy.loginfo(f"  Joint {i}: {pos} radians")
        rospy.sleep(1.0)
    
    # 第五关节额外运动
    def move_joint5_additional(self):
        rospy.loginfo("Preparing additional movement for Joint 5...")
        rospy.sleep(1.0)
        joint5_new_position = 0
        msg = Float64()
        msg.data = joint5_new_position
        self.joint_publishers[4].publish(msg)
        rospy.loginfo(f"Joint 5 moving to additional position: {joint5_new_position} radians")
        rospy.sleep(2.0)
    
    # 发布关节位置指令
    def _publish_each_joint(self, positions):
        for pub, pos in zip(self.joint_publishers, positions):
            msg = Float64()
            msg.data = pos
            pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node("multi_joint_control")
    try:
        controller = MultiJointControl()
        rospy.spin()  # 保持节点活跃
    except rospy.ROSInterruptException:
        pass