#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

def callback(msg):
    # 当收到消息时执行
    rospy.loginfo("Received: %s", msg.data)

def listener():
    # 初始化节点，命名为 'listener'（匿名名称避免冲突）
    rospy.init_node('listener', anonymous=True)
    
    # 创建Subscriber，订阅 'chatter' 话题，消息类型为String
    rospy.Subscriber('chatter', String, callback)
    
    # 保持节点运行，直到关闭
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
