clc;clear all;close all;

% 读入图像并转为灰度图
img=imread('testimg1.jpg');
img=rgb2gray(img);

% 调用函数
[OptimalThreshold_result,t_changing] = MyOptimalThreshold(img);

% 处理后的图像
figure;
imshow(OptimalThreshold_result);
title('OptimalThreshold_result');

% t的变化曲线
figure;
plot(1:length(t_changing), t_changing); 
xlabel('times');  
ylabel('t');
title('t_changing');  