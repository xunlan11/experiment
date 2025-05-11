clc;clear all;close all;

% 读入图像并转换为灰度图像
img = imread('testimg.jpg');
img = im2gray(img);

% 积分图像
InteImg_result = MyInteImg(img);

% 输出图像
figure;
imshow(InteImg_result,[]);
title('InteImg-result');