clc;clear all;close all;

% ����ͼ��תΪ�Ҷ�ͼ
img=imread('testimg1.jpg');
img=rgb2gray(img);

% ���ú���
[OptimalThreshold_result,t_changing] = MyOptimalThreshold(img);

% ������ͼ��
figure;
imshow(OptimalThreshold_result);
title('OptimalThreshold_result');

% t�ı仯����
figure;
plot(1:length(t_changing), t_changing); 
xlabel('times');  
ylabel('t');
title('t_changing');  