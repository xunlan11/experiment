clc;clear all;close all;

% ����ͼ��ת��Ϊ�Ҷ�ͼ
img=imread('testimg.jpg');
img=rgb2gray(img);

% ������
in = imnoise(img, 'gaussian', 0.1);

% ��ֵ�˲�
Medfilt2_result = MyMedfilt2(in);

% ���ͼ��
imshow(img);
title('img');
figure;
imshow(in);
title('gaussian');
figure;
imshow(Medfilt2_result,[]);
title('Medfilt2_result');