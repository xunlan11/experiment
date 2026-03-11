import rospy
from std_msgs.msg import Float64, String

def callback(msg):
    if "One" in msg.data.lower():
        print(1)

rospy.init_node('voice_command_listener')
rospy.Subscriber('/voiceWords', String, callback)
rospy.spin()
