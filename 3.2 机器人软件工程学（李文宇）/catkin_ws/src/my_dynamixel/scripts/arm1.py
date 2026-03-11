#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64
import time

class MultiJointControl:
    def __init__(self, max_runtime):
        self.max_runtime = max_runtime   # 最大运行时间（s）
        self.start_time = time.time()    # 启动时间
        
        # 初始化5个关节的发布者
        self.joint_publishers = [
            rospy.Publisher(f"/motor{i}_controller/command", Float64, queue_size=10)
            for i in range(1, 6)
        ]
        rospy.sleep(0.5)
        
        # 执行运动序列
        self.execute_motion_sequence()
        
        # 设置定时器检查是否超时（每0.1秒检查一次）
        self.timer = rospy.Timer(rospy.Duration(0.1), self.check_runtime)
    
    # 检查是否超过最大运行时间
    def check_runtime(self, event):
        elapsed = time.time() - self.start_time
        if elapsed > self.max_runtime:
            rospy.loginfo(f"已达到最大运行时间 {self.max_runtime} 秒，停止程序")
            self.timer.shutdown()  # 停止定时器
            rospy.signal_shutdown("最大运行时间到达")  # 关闭ROS节点
    
    # 执行运动序列
    def execute_motion_sequence(self):
        rospy.loginfo("开始执行运动序列")
        self.move_to_first_positions()
        # self.move_to_initial_positions()
        # self.move_to_target_positions()
        # self.move_joint5_additional()
    
    # 初始位置
    def move_to_first_positions(self):
        target_positions = [-1.5, -1.6, 1.1, 0.3, 0] 
        self._publish_each_joint(target_positions)
        rospy.loginfo("移动关节到初始位置:")
        for i, pos in enumerate(target_positions, 1):
            rospy.loginfo(f"  关节 {i}: {pos} 弧度")
        rospy.sleep(1.0)
    
    # 拾取位置
    def move_to_initial_positions(self):
        initial_positions = [-1.5, -2.3, 1.2, -0.5, 2]  
        self._publish_each_joint(initial_positions)
        rospy.loginfo("移动关节到拾取位置:")
        for i, pos in enumerate(initial_positions, 1):
            rospy.loginfo(f"  关节 {i}: {pos} 弧度")
        rospy.sleep(1.0)
    
    # 目标位置
    def move_to_target_positions(self):
        target_positions = [-1.5, -1.8, 0.5, 0.5, 0]
        self._publish_each_joint(target_positions)
        rospy.loginfo("移动关节到目标位置:")
        for i, pos in enumerate(target_positions, 1):
            rospy.loginfo(f"  关节 {i}: {pos} 弧度")
        rospy.sleep(1.0)
    
    # 第五关节额外运动
    def move_joint5_additional(self):
        rospy.loginfo("准备第五关节的额外运动...")
        rospy.sleep(1.0) 
        joint5_new_position = 0
        msg = Float64()
        msg.data = joint5_new_position
        self.joint_publishers[4].publish(msg)
        rospy.loginfo(f"第五关节移动到额外位置: {joint5_new_position} 弧度")
        rospy.sleep(2.0)
    
    # 发布关节位置指令
    def _publish_each_joint(self, positions):
        for pub, pos in zip(self.joint_publishers, positions):
            msg = Float64()
            msg.data = pos
            pub.publish(msg)

if __name__ == "__main__":
    try:
        rospy.init_node("multi_joint_control")
        max_runtime = 3  # 最大运行时间（s）
        rospy.loginfo(f"启动节点，将在 {max_runtime} 秒后自动停止")
        controller = MultiJointControl(max_runtime)
        rospy.spin()  # 保持节点活跃
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")