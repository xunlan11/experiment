clc;clear all;close all;

% ����ͼ��ת��Ϊ��ֵͼ��
img = imread('testimg.jpg');
bw = im2bw(img);

% ����任
DisTrans_result = MyDisTrans(bw);

% չʾ���
figure;
imshow(DisTrans_result,[]);
title('DisTrans-result');