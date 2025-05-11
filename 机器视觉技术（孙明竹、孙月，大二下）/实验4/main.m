clc;clear all;close all;

% 读入图像并转换为灰度图
img=imread('testimg.jpg');
img=rgb2gray(img);

% 加噪音
in = imnoise(img, 'gaussian', 0.1);

% 中值滤波
Medfilt2_result = MyMedfilt2(in);

% 输出图像
imshow(img);
title('img');
figure;
imshow(in);
title('gaussian');
figure;
imshow(Medfilt2_result,[]);
title('Medfilt2_result');