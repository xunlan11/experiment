#include <ros/ros.h>
#include <homework3/messages.h> 
#include <sstream>

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "talker");
    ros::NodeHandle n;
    
    ros::Publisher publisher = n.advertise<homework3::messages>("chatter", 1000);
    ros::Rate loop_rate(10);

    int count = 0;
    while (ros::ok()) {
        homework3::messages msg; // 自定义消息类型
        msg.number = count * 1.0f;           
        msg.text = "Nankai university";

        ROS_INFO("Publishing: number=%f, text=%s", msg.number, msg.text.c_str());
        publisher.publish(msg);

        ros::spinOnce();
        loop_rate.sleep();
        ++count;
    }
    return 0;
}