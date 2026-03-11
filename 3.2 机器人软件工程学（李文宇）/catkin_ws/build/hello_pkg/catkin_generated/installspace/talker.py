#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import String

def talker():
    # 初始化节点，命名为 'talker'（匿名名称避免冲突）
    rospy.init_node('talker', anonymous=True)
    
    # 创建Publisher，发布到 'chatter' 话题，消息类型为String
    pub = rospy.Publisher('chatter', String, queue_size=10)
    
    # 设置发布频率（Hz）
    rate = rospy.Rate(10)  # 10Hz
    
    # 循环直到节点被关闭
    while not rospy.is_shutdown():
        hello_str = "Hello ROS at time %s" % rospy.get_time()
        
        # 发布消息
        pub.publish(hello_str)
        rospy.loginfo("Published: %s", hello_str)  # 打印日志
        
        # 按频率休眠
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
