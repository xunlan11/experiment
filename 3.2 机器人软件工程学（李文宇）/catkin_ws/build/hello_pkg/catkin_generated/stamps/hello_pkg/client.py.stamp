from hello_pkg.srv import AddTwoInts
import rospy
import sys

def add_two_ints_client(x, y):
    rospy.wait_for_service('add_two_ints')
    try:
        add_two_ints = rospy.ServiceProxy('add_two_ints', AddTwoInts)
        resp = add_two_ints(x, y)
        return resp.sum
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: client.py X Y")
        sys.exit(1)

    x = int(sys.argv[1])
    y = int(sys.argv[2])

    rospy.init_node('add_two_ints_client')
    result = add_two_ints_client(x, y)
    if result is not None:
        rospy.loginfo(f"Sum of {x} and {y} is {result}")
