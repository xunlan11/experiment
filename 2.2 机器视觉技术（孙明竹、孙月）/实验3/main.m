clc;clear all;close all;

% ����ͼ��ת��Ϊ�Ҷ�ͼ��
img = imread('testimg.jpg');
img = im2gray(img);

% ����ͼ��
InteImg_result = MyInteImg(img);

% ���ͼ��
figure;
imshow(InteImg_result,[]);
title('InteImg-result');