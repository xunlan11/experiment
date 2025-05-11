clc;clear all;close all;

% 读入图像并转换为二值图像
img = imread('testimg.jpg');
bw = im2bw(img);

% 距离变换
DisTrans_result = MyDisTrans(bw);

% 展示结果
figure;
imshow(DisTrans_result,[]);
title('DisTrans-result');