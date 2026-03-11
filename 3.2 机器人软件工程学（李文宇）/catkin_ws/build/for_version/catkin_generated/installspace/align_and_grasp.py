#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import datetime
import time
import subprocess
import threading
from ultralytics import YOLO
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class BottlePositionAdjuster:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('bottle_position_adjuster', anonymous=True)
        
        # 加载YOLOv8模型
        self.model = YOLO("/home/zhaopengyu/catkin_ws/src/for_version/scripts/best.pt")
        
        # 类别名称映射
        self.class_names = ['cola', 'pepsi', 'sprite', 'fanta', 'spring', 'ice', 'scream', 'milk', 'red', 'king']
        
        # 预设目标框坐标 (在实际位置校准后的值)
        self.target_points = np.array([
            [95, 166],  # 左上
            [209, 166],  # 右上
            [209, 448],  # 右下
            [95, 448]   # 左下
        ], dtype=np.float32)
        
        # 计算理想方框中心点和高度
        self.target_center_x = (95 + 209) / 2.0
        self.target_center_y = (166 + 448) / 2.0
        self.target_height = 448 - 166  # 目标水瓶的像素高度
        
        # 控制参数 (优化后的值)
        self.kp_distance = 0.003      # 距离调整比例系数 (更安全的值)
        self.kp_rotation = 0.005       # 旋转调整比例系数 (提高响应性)
        self.kp_size = 0.001           # 大小调整比例系数
        
        self.position_threshold = 30.0  # 位置误差阈值(像素)
        self.angle_threshold = 0.15     # 角度误差阈值(弧度)
        self.size_threshold = 20.0      # 大小误差阈值(像素)
        self.max_adjust_time = 45.0     # 最大调整时间(秒) - 更长
        self.adjustment_count = 0       # 调整计数器
        
        # 通信接口
        self.cmd_pub = rospy.Publisher('/cmd_vel_mux/input/navi', Twist, queue_size=10)
        self.image_pub = rospy.Publisher('/visualization/image_raw', Image, queue_size=10)
        self.bridge = CvBridge()
        
        # 状态变量
        self.start_time = 0
        self.adjustment_complete = False
        self.subscriber_warning_shown = False
        self.last_detection_time = rospy.get_time()
        self.current_linear = 0.0  # 当前线性速度
        self.current_angular = 0.0  # 当前角速度
        self.stable_count = 0       # 稳定计数器
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            rospy.logerr("无法打开摄像头")
            exit()
            
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 创建OpenCV可视化窗口
        cv2.namedWindow('Position Adjuster', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Position Adjuster', 800, 600)
        
        # 检查底盘控制状态
        self.check_subscriber_status()
        
        rospy.loginfo(f"目标水瓶像素高度: {self.target_height}px, 目标位置: ({self.target_center_x:.1f}, {self.target_center_y:.1f})")
    
    def check_subscriber_status(self):
        """检查/cmd_vel是否有订阅者"""
        try:
            # 获取话题信息
            topic_info = rospy.get_published_topics()
            cmd_vel_info = [info for info in topic_info if info[0] == '/cmd_vel']
            
            if cmd_vel_info:
                subscribers = cmd_vel_info[0][1]
                if subscribers:
                    rospy.loginfo(f"找到/cmd_vel订阅者: {subscribers}")
                    return True
        except Exception as e:
            rospy.logwarn(f"订阅者检查失败: {str(e)}")
        
        if not self.subscriber_warning_shown:
            rospy.logwarn("/cmd_vel没有订阅者! 请手动启动底盘控制器节点")
            rospy.logwarn("例如: roslaunch turtlebot3_bringup turtlebot3_robot.launch")
            rospy.logwarn("或在仿真环境中: roslaunch turtlebot3_gazebo turtlebot3_world.launch")
            self.subscriber_warning_shown = True
        return False
    
    def get_center_error(self, current_points):
        """计算当前检测框中心和目标框中心的偏差"""
        if current_points is None or len(current_points) < 4:
            return None, None
        
        # 计算当前框的中心点
        x_center = (current_points[0][0] + current_points[1][0] + 
                    current_points[2][0] + current_points[3][0]) / 4.0
        y_center = (current_points[0][1] + current_points[1][1] + 
                    current_points[2][1] + current_points[3][1]) / 4.0
        
        # 计算水平和垂直偏差
        dx = x_center - self.target_center_x  # 正偏差表示物体在右边，小车需要左转
        dy = y_center - self.target_center_y  # 正偏差表示物体在上方，小车需要后退
        
        return dx, dy
    
    def get_size_error(self, box):
        """计算当前检测框大小与目标大小的偏差"""
        if box is None:
            return None
            
        # 提取检测框坐标
        x1, y1, x2, y2 = box
        height = abs(y2 - y1)
        
        # 大小偏差（正偏差表示瓶子太大，小车需要后退）
        size_error = height - self.target_height
        
        return size_error
    
    def get_rotation_error(self, box):
        """根据检测框的宽高比估算旋转偏差"""
        if box is None:
            return None
            
        # 提取检测框坐标
        x1, y1, x2, y2 = box
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # 计算宽高比 (理想应为方形)
        aspect_ratio = width / float(height) if height > 0 else 1.0
        
        # 估计旋转角度 (偏差)
        # 理想情况下，当水瓶正面朝前时，宽高比应该接近1
        # 偏离1的程度表示旋转角度
        rotation_error = (aspect_ratio - 1.0) * 0.5
        
        return rotation_error
    
    def detect_bottle(self, frame):
        """使用YOLO检测水瓶并返回最佳检测框坐标"""
        # 缩小检测区域尺寸以提高性能
        small_frame = cv2.resize(frame, (320, 240))
        
        # 直接使用帧进行检测，避免文件IO
        results = self.model(small_frame, verbose=False, imgsz=320)  # 禁用冗余输出
        
        # 寻找最佳检测结果
        best_box = None
        max_conf = 0.0
        best_label = ""
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                conf = box.conf.item()
                if conf > max_conf and conf > 0.5:  # 添加置信度阈值
                    max_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # 将坐标映射回原图尺寸
                    x1 = int(x1 * frame.shape[1] / 320)
                    y1 = int(y1 * frame.shape[0] / 240)
                    x2 = int(x2 * frame.shape[1] / 320)
                    y2 = int(y2 * frame.shape[0] / 240)
                    
                    best_box = (x1, y1, x2, y2)
                    cls_idx = int(box.cls.item())
                    best_label = self.class_names[cls_idx]
        
        return best_box, max_conf, best_label
    
    def calculate_adjustment(self, dx, size_error, rotation_error):
        """根据偏差计算调整命令"""
        cmd = Twist()
        
        # 设置默认值
        linear_x = 0.0
        angular_z = 0.0
        
        # 如果所有误差都可用
        if dx is not None and size_error is not None and rotation_error is not None:
            # 水平位置调整 (左右移动)：使用比例控制
            angular_z = -self.kp_rotation * dx  # 取负值，使小车朝物体转向
            
            # 大小调整：根据高度差异控制距离
            linear_x = -self.kp_size * size_error
            
            # 旋转调整：使用比例控制（减少影响）
            angular_z += self.kp_rotation * rotation_error * 0.3
            
            # 当位置误差较小时，优先调整大小（距离）
            if abs(dx) < 50:
                angular_z *= 0.5
                linear_x = -self.kp_size * size_error * 1.5  # 加强距离控制
            else:
                linear_x *= 0.3  # 弱化距离控制，优先转向
                
            # 添加安全限制
            linear_x = np.clip(linear_x, -0.2, 0.2)
            angular_z = np.clip(angular_z, -0.4, 0.4)
        
        # 更新当前速度状态
        self.current_linear = linear_x
        self.current_angular = angular_z
        
        # 应用控制指令
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        
        # 记录控制指令
        self.adjustment_count += 1
        if self.adjustment_count % 5 == 0:  # 每5次调整记录一次日志
            rospy.loginfo(f"控制指令: 线性x={cmd.linear.x:.3f}, 角速度z={cmd.angular.z:.3f}")
            if dx is not None and size_error is not None and rotation_error is not None:
                rospy.loginfo(f"误差值: dx={dx:.1f}px, 大小误差={size_error:.1f}px, 旋转误差={rotation_error:.2f}rad")
        
        return cmd
    
    def visualize_detection(self, frame, box, label, conf, dx=None, size_error=None, rotation_error=None):
        """可视化检测结果和调整信息"""
        # 绘制整个理想方框
        target_pts = self.target_points.astype(int).reshape((-1, 1, 2))
        cv2.polylines(frame, [target_pts], True, (0, 0, 255), 2)
        
        # 绘制理想方框中心点
        cv2.circle(frame, (int(self.target_center_x), int(self.target_center_y)), 8, (0, 0, 255), -1)
        
        # 添加目标大小信息
        cv2.putText(frame, f"目标高度: {self.target_height}px", 
                    (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 绘制当前检测框
        if box:
            x1, y1, x2, y2 = box
            height = abs(y2 - y1)
            width = abs(x2 - x1)
            
            # 显示检测框高度
            cv2.putText(frame, f"当前高度: {height}px", 
                        (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制检测框中心点
            center_x_box = (x1 + x2) // 2
            center_y_box = (y1 + y2) // 2
            cv2.circle(frame, (center_x_box, center_y_box), 5, (255, 0, 0), -1)
            
            # 绘制从理想中心到当前中心的连线
            cv2.line(frame, (int(self.target_center_x), int(self.target_center_y)), 
                     (center_x_box, center_y_box), (0, 255, 255), 2)
            
            # 在连线上方添加偏差标签
            if dx is not None:
                cv2.putText(frame, f"水平偏差:{dx:.1f}px", 
                           ((int(self.target_center_x) + center_x_box)//2, 
                            (int(self.target_center_y) + center_y_box)//2 - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 绘制大小误差
            if size_error is not None:
                cv2.putText(frame, f"大小误差:{size_error:.1f}px", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
                
                # 绘制指示箭头
                if size_error > 0:  # 瓶子太小，需要前进
                    cv2.arrowedLine(frame, 
                                   (center_x_box, y2), 
                                   (center_x_box, y2 - 30), 
                                   (0, 255, 0), 2, tipLength=0.3)
                else:  # 瓶子太大，需要后退
                    cv2.arrowedLine(frame, 
                                   (center_x_box, y2), 
                                   (center_x_box, y2 + 30), 
                                   (0, 0, 255), 2, tipLength=0.3)
            
            # 绘制旋转指示
            if rotation_error is not None:
                angle = rotation_error * 45  # 放大显示
                cv2.putText(frame, f"旋转:{rotation_error:.2f}", (x2 + 10, (y1+y2)//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1)
        
        # 显示调整信息
        if dx is not None and size_error is not None:
            status_text = f"调整中: 水平偏差:{dx:.1f}px, 大小误差:{size_error:.1f}px"
            cv2.putText(frame, status_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 绘制方向箭头
            if dx < -5:  # 物体偏右，需要向左转
                cv2.arrowedLine(frame, 
                              (int(self.target_center_x), int(self.target_center_y)), 
                              (int(self.target_center_x) - 40, int(self.target_center_y)), 
                              (255, 0, 0), 2, tipLength=0.3)
            elif dx > 5:  # 物体偏左，需要向右转
                cv2.arrowedLine(frame, 
                              (int(self.target_center_x), int(self.target_center_y)), 
                              (int(self.target_center_x) + 40, int(self.target_center_y)), 
                              (255, 0, 0), 2, tipLength=0.3)
            
            # 显示控制状态
            if self.adjustment_complete:
                comp_text = "调整完成!"
                cv2.putText(frame, comp_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "寻找目标...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 添加警告信息（如果没有订阅者）
        if self.subscriber_warning_shown:
            warn_text = "警告: /cmd_vel无订阅者!"
            cv2.putText(frame, warn_text, (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            warn_text2 = "请手动启动底盘节点!"
            cv2.putText(frame, warn_text2, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 添加时间戳
        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, time_str, (10, frame.shape[0]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 添加目标位置信息
        target_text = f"目标位置: ({self.target_center_x:.0f}, {self.target_center_y:.0f})"
        cv2.putText(frame, target_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        return frame
    
    def adjustment_complete_check(self, dx, size_error, rotation_error):
        """检查是否达到调整目标"""
        if dx is None or size_error is None or rotation_error is None:
            self.stable_count = 0
            return False
            
        # 计算总位置误差
        pos_error = abs(dx)
        size_error_val = abs(size_error)
        rot_error = abs(rotation_error)
        
        # 检查是否在阈值范围内
        in_position = pos_error < self.position_threshold
        in_size = size_error_val < self.size_threshold
        in_angle = rot_error < self.angle_threshold
        timeout = (rospy.get_time() - self.start_time) > self.max_adjust_time
        
        # 稳定性检查：需要连续几次在阈值内
        if in_position and in_size and in_angle:
            self.stable_count += 1
        else:
            self.stable_count = max(0, self.stable_count - 1)  # 如果不满足条件，减少计数
        
        # 调整完成条件：稳定在目标位置或超时
        if (self.stable_count > 5) or timeout:
            self.adjustment_complete = True
            self.send_zero_velocity()
            rospy.loginfo(f"位置调整完成! 水平误差={pos_error:.1f}px, 大小误差={size_error_val:.1f}px, 旋转误差={rot_error:.3f}rad")
            return True
        
        return False
    
    def send_zero_velocity(self):
        """发送零速度指令"""
        twist = Twist()
        for _ in range(5):
            self.cmd_pub.publish(twist)
            rospy.sleep(0.1)
        self.current_linear = 0.0
        self.current_angular = 0.0
    
    def search_bottle(self):
        """执行水瓶搜索模式"""
        rospy.loginfo("未检测到水瓶，执行搜索模式...")
        
        # 创建搜索指令 (缓慢旋转)
        search_cmd = Twist()
        search_cmd.angular.z = 0.3 if rospy.get_time() % 4 < 2 else -0.3
        self.cmd_pub.publish(search_cmd)
        self.current_linear = 0.0
        self.current_angular = search_cmd.angular.z
    
    def main_loop(self):
        """主控制循环"""
        rate = rospy.Rate(15)  # 降低到15Hz以获得更稳定的检测
        self.start_time = rospy.get_time()
        self.stable_count = 0
        self.adjustment_count = 0
        
        rospy.loginfo("位置调整器已启动. 开始位置调整...")
        rospy.loginfo(f"目标水瓶像素高度: {self.target_height}px, 目标位置: ({self.target_center_x:.0f}, {self.target_center_y:.0f})")
        
        while not rospy.is_shutdown() and not self.adjustment_complete:
            # 定期检查订阅者状态
            if rospy.get_time() - self.last_detection_time > 3.0:
                self.check_subscriber_status()
                self.last_detection_time = rospy.get_time()
            
            # 获取当前帧
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("摄像头帧捕获失败")
                rospy.sleep(0.1)
                continue
            
            # 检测水瓶
            bottle_box, conf, label = self.detect_bottle(frame)
            
            dx = None
            size_error = None
            rotation_error = None
            
            # 如果有检测结果
            if bottle_box:
                # 提取框坐标
                x1, y1, x2, y2 = bottle_box
                current_points = np.array([
                    [x1, y1], [x2, y1],
                    [x2, y2], [x1, y2]
                ], dtype=np.float32)
                
                # 计算位置偏差
                dx, _ = self.get_center_error(current_points)
                
                # 计算大小偏差（关键改进）
                size_error = self.get_size_error(bottle_box)
                
                # 计算旋转偏差
                rotation_error = self.get_rotation_error(bottle_box)
                
                # 检查是否完成调整
                if self.adjustment_complete_check(dx, size_error, rotation_error):
                    rospy.loginfo("位置调整完成!")
                    break
                
                # 计算并发送调整命令
                cmd = self.calculate_adjustment(dx, size_error, rotation_error)
                self.cmd_pub.publish(cmd)
            else:
                # 未检测到水瓶时执行搜索模式
                self.search_bottle()
                
            # 可视化显示
            visual_frame = self.visualize_detection(
                frame, bottle_box, label, conf, dx, size_error, rotation_error
            )
            
            # 显示图像
            cv2.imshow('Position Adjuster', visual_frame)
            
            # 检查是否按下退出键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                rospy.loginfo("退出系统...")
                break
            
            rate.sleep()
        
        # 调整完成后停留3秒再退出
        if self.adjustment_complete:
            rospy.loginfo("位置调整完成. 停止小车并暂停...")
            rospy.sleep(3.0)
        else:
            rospy.loginfo("位置调整超时或中断")
    
    def cleanup(self):
        """资源清理"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.send_zero_velocity()
        rospy.loginfo("系统资源已清理")

if __name__ == "__main__":
    adjuster = BottlePositionAdjuster()
    try:
        adjuster.main_loop()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"发生错误: {str(e)}")
        import traceback
        rospy.logerr(traceback.format_exc())
    finally:
        adjuster.cleanup()