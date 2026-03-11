#!/usr/bin/env python3

from cv_bridge import CvBridge, CvBridgeError
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image

class TemplateMatcher:
    def __init__(self):
        self.bridge = CvBridge()
        # 加载模板（初始化时加载一次即可）
        self.template = cv2.imread('/home/zhaopengyu/图片/template.bmp')
        if self.template is None:
            rospy.logerr("无法加载模板图像！请检查路径：/home/zhaopengyu/图片/template.bmp")
            rospy.signal_shutdown("模板加载失败")

        # 多尺度参数配置
        self.scales = np.linspace(0.5, 1.5, 20)  # 缩放范围：50%~150%，共20个尺度
        self.sqd_threshold = 50000                # TM_SQDIFF阈值（需根据实际调整）

    def match_template(self, cv_image):
        best_match = None
        best_val = float('inf')  # TM_SQDIFF需要最小值

        # 多尺度匹配循环
        for scale in self.scales:
            # 缩放模板
            scaled_template = cv2.resize(self.template, None, 
                                       fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
            
            # 跳过比图像大的模板
            if (scaled_template.shape[0] > cv_image.shape[0] or 
                scaled_template.shape[1] > cv_image.shape[1]):
                continue
            
            # 执行模板匹配
            res = cv2.matchTemplate(cv_image, scaled_template, cv2.TM_SQDIFF)
            min_val, _, min_loc, _ = cv2.minMaxLoc(res)
            
            # 记录最佳匹配
            if min_val < best_val:
                best_val = min_val
                best_match = {
                    "loc": min_loc,
                    "size": scaled_template.shape[:2],
                    "scale": scale
                }

        # 判断是否找到有效匹配
        if best_val < self.sqd_threshold:
            return best_match
        else:
            return None

def call_back(data):
    try:
        # 初始化匹配器（避免重复加载模板）
        if not hasattr(call_back, "matcher"):
            call_back.matcher = TemplateMatcher()
        
        # 转换ROS图像
        cv_image = call_back.matcher.bridge.imgmsg_to_cv2(data, "bgr8")
        
        # 显示原始图像
        cv2.imshow("src", cv_image)
        cv2.waitKey(1)
        
        # 显示模板图像
        cv2.imshow("template", call_back.matcher.template)
        cv2.waitKey(1)
        
        # 执行多尺度匹配
        match_result = call_back.matcher.match_template(cv_image)
        
        if match_result:
            # 提取匹配信息
            top_left = match_result["loc"]
            h, w = match_result["size"]
            
            # 绘制红色矩形框
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(cv_image, top_left, bottom_right, (0, 0, 255), 2)
            
            # 在左上角显示匹配信息
            text = f"Scale: {match_result['scale']:.2f}, Score: {match_result['score']}"
            cv2.putText(cv_image, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0,0,255), 2)
        else:
            rospy.logwarn("未找到有效匹配目标")
        
        # 显示结果
        cv2.imshow("dst", cv_image)
        cv2.waitKey(1)
    
    except Exception as e:
        rospy.logerr(f"处理异常: {str(e)}")

if __name__ == '__main__':
    rospy.init_node('template_matching', anonymous=False)
    img_topic = '/usb_cam/image_raw'
    rospy.Subscriber(img_topic, Image, call_back)
    rospy.loginfo("多尺度模板匹配节点已启动...")
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("关闭节点...")
    
    cv2.destroyAllWindows()