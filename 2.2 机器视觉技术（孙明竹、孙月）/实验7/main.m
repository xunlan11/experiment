clc;clear all;close all;

% ����ͼ��ת��Ϊ�Ҷ�ͼ
img=imread('testimg.jpg');
img=rgb2gray(img);

% OTSU��ֵ���
OTSU_result = MyOTSU(img);

figure;
imshow(OTSU_result);
title('OTSU_result');