#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import time

# 控制向前移动的距离和速度
def move_forward(distance, speed=0.1):
    rospy.init_node('move_forward', anonymous=True)
    pub = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)
    rate = rospy.Rate(10)  # Hz
    
    move_time = distance / speed
    start_time = time.time()
    
    
    while (time.time() - start_time) < move_time:
        twist = Twist()
        twist.linear.x = speed
        pub.publish(twist)
        rate.sleep()
    
    # 停止小车
    twist = Twist()
    pub.publish(twist)

if __name__ == '__main__':
    try:
        move_forward(distance=0.51, speed=0.1)
    except rospy.ROSInterruptException:
        pass