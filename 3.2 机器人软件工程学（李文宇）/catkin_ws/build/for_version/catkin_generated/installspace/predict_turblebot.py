#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import datetime
import os

# 加载YOLOv8模型
model = YOLO(r"/home/zhaopengyu/catkin_ws/src/for_version/scripts/best.pt")

# 定义类别名称映射
class_names = ['cola', 'pepsi', 'sprite', 'fanta', 'spring', 'ice', 'scream', 'milk', 'red', 'king']

# 打开默认摄像头
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧（可能是摄像头断开）")
        break

    # 将图像帧保存到临时文件
    temp_image_path = 'temp_image.jpg'
    cv2.imwrite(temp_image_path, frame)

    # 使用YOLOv8进行检测
    results = model(temp_image_path)

    # 清理临时文件
    os.remove(temp_image_path)

    # 解析结果并绘制检测框
    for result in results:
        # 获取所有检测框信息
        boxes = result.boxes
        for box in boxes:
            # 正确提取坐标、置信度和类别
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf.item()
            cls_idx = int(box.cls.item())
            label = class_names[cls_idx]

            # 绘制检测框和标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(x1, y1, x2, y2, conf, label)

    # 添加时间戳
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示当前帧
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
