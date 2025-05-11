clc;clear all;close all;

% 输入图像并转换为灰度图
img=imread('testimg.jpg');
img=rgb2gray(img);

% OTSU阈值检测
OTSU_result = MyOTSU(img);

figure;
imshow(OTSU_result);
title('OTSU_result');