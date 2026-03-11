#!/usr/bin/env python
# coding: utf-8
import cv2 as cv
import rospy
import numpy as np
from ultralytics import YOLO
from std_msgs.msg import Float64
from PID import PositionalPID
import time

class PersonFollowROS:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node("person_follow_control", anonymous=True)
        rospy.on_shutdown(self.cleanup)
        
        # 初始化关节发布者（五个）
        self.joint_publishers = [
            rospy.Publisher(f"/motor{i}_controller/command", Float64, queue_size=10)
            for i in range(1, 6)
        ]
        rospy.sleep(0.5)
        
        # 初始化PID控制器
        self.xservo_pid = PositionalPID(0.05, 0.06, 0.05)
        self.yservo_pid = PositionalPID(0.05, 0.06, 0.05)
        
        self.a = 0  # x轴控制标志
        self.b = 0  # y轴控制标志
        self.target_servox = -1.5 # 初始位置
        self.target_servoy = 1.1 # 初始位置
        
        # 加载YOLOv8模型
        try:
            self.model = YOLO("/home/zhaopengyu/catkin_ws/src/for_version/scripts/yolov8s-pose.pt")
            rospy.loginfo("YOLOv8 模型加载成功")
        except Exception as e:
            rospy.logerr(f"YOLOv8 模型加载失败: {e}")
            raise
        
        self.conf_threshold = 0.5 # 置信度阈值
        
        # 初始化摄像头
        try:
            self.cap = cv.VideoCapture(2) 
            if not self.cap.isOpened():
                rospy.logerr("无法打开摄像头")
                raise ValueError("Camera initialization failed")
            rospy.loginfo("摄像头初始化成功")
        except Exception as e:
            rospy.logerr(f"摄像头初始化失败: {e}")
            raise
        
        # 初始化机械臂位置
        self.move_to_initial_position()
    
    # 移动机械臂到初始跟踪位置
    def move_to_initial_position(self):
        initial_positions = [-1.5, -1.6, 1.1, 0.3, 0]
        self._publish_each_joint(initial_positions)
        rospy.loginfo("初始化机械臂位置...")
        rospy.sleep(1.0)
    
    # 发布各关节位置指令
    def _publish_each_joint(self, positions):
        for pub, pos in zip(self.joint_publishers, positions):
            msg = Float64()
            msg.data = pos
            pub.publish(msg)
    
    # 人体跟踪
    def follow_function(self, img):
        self.a = 0
        self.b = 0
        img = cv.resize(img, (640, 480))

        # 使用YOLOv8进行姿态估计
        results = self.model(img, conf=self.conf_threshold)

        if len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            # 获取第一个检测到的人的关键点
            keypoints = results[0].keypoints.xy[0]
            
            if len(keypoints) > 0:
                # 使用鼻子的位置作为跟踪点 (关键点索引0为鼻子)
                nose = keypoints[0]
                x, y = int(nose[0]), int(nose[1])

                # 绘制关键点和文字
                annotated_frame = results[0].plot()
                cv.putText(
                    annotated_frame,
                    "Person",
                    (280, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (105, 105, 105),
                    2,
                )
                
                # 绘制目标区域
                cv.rectangle(annotated_frame, (260, 180), (380, 300), (0, 255, 0), 2)

                # 检查是否在目标区域内
                if (260 < x < 380) and (180 < y < 300):
                    self.b = 1
                    self.a = 1
                    cv.putText(
                        annotated_frame,
                        "Locked",
                        (x+10, y),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                # X轴控制逻辑
                if not (self.a == 1 and ((self.target_servox <= -2.5 and x <= 320) or (self.target_servox >= 0 and x >= 320))):
                    if self.a == 0:
                        # 设置PID目标值（图像中心）
                        self.xservo_pid.SetStepSignal(320)
                        # 更新PID系统反馈（当前检测位置）
                        self.xservo_pid.SystemOutput = x
                        # 获取PID控制量
                        pid_output = self.xservo_pid.PidOutput
                        # 将PID输出转换为角度变化
                        angle_delta = pid_output * (1.25 / 320)
                        # 更新目标位置（向下调整，因为右侧对应更大的负值）
                        self.target_servox = -1.5 +angle_delta
                        # 限制关节1的范围
                        self.target_servox = max(-2.5, min(0, self.target_servox))

                # Y轴控制逻辑
                if not (self.b == 1 and ((self.target_servoy >= 1.8 and y <= 240) or (self.target_servoy <= 0.5 and y >= 240))):
                    if self.b == 0:
                        self.yservo_pid.SetStepSignal(240)
                        self.yservo_pid.SystemOutput = y
                        pid_output = self.yservo_pid.PidOutput
                        angle_delta = pid_output * (0.65 / 240)
                        # 更新目标位置（向上调整，因为下方对应更大的正值）
                        self.target_servoy = 1.1 - angle_delta
                        # 限制关节3的范围
                        self.target_servoy = max(0.5, min(1.8, self.target_servoy))

                # 构建关节位置消息
                positions = [
                    self.target_servox,  # 关节1
                    -1.6,                # 关节2
                    self.target_servoy,  # 关节3
                    0.3,                 # 关节4
                    0                    # 关节5
                ]
                # 发布关节位置命令
                self._publish_each_joint(positions)
                rospy.loginfo(f"关节位置: {positions}")
                
                cv.putText(
                    annotated_frame,
                    f"X:{self.target_servox:.2f}, Y:{self.target_servoy:.2f}",
                    (20, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

                return annotated_frame

        return img
    
    # 主循环
    def run(self):
        rate = rospy.Rate(10)  # Hz
        
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("无法读取摄像头帧")
                break
                
            annotated_frame = self.follow_function(frame)
            
            cv.imshow("Person Follow ROS", annotated_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
                
            rate.sleep()
    
    # 清理资源
    def cleanup(self):
        self.cap.release()
        cv.destroyAllWindows()
        rospy.loginfo("正在关闭人体跟踪节点...")


if __name__ == "__main__":
    try:
        follower = PersonFollowROS()
        follower.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"发生错误: {e}")