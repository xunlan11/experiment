#!/usr/bin/env python
import rospy
from homework3.msg import messages

def callback(message):
    rospy.loginfo("收到: number=%f, text=%s", message.number, message.text)

def listener():
    rospy.init_node("listener", anonymous=True)
    rospy.Subscriber("chatter", messages, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()