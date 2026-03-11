#!/usr/bin/env python3
import rospy
from hello_pkg.msg import homework_msg  # 使用当前包名

def callback(data):
    rospy.loginfo("[Python] 订阅器收到: %.2f, %s", 
                 data.number, data.text)

def listener():
    rospy.init_node('py_subscriber', anonymous=True)
    rospy.Subscriber("mixed_data",homework_msg, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
