% ���庯��MyOptimalThreshold������Ϊ��ֵͼ��img
function [OptimalThreshold_result,t_changing]=MyOptimalThreshold(img)

img = double(img);

% ��ʼ����洢���� 
t0 = 0;
t1 = (min(img(:)) + max(img(:))) / 2; 
t = zeros(1, 100);
t_index = 1; % ��ǰ��ֵ�������е�����

% ��������  
while abs(t0 - t1) > 1  
    t0 = t1;
    img1 = img >= t0;  
    img2 = img < t0;  
    u1 = mean(img(img1));  
    u2 = mean(img(img2));  
    t1 = (u1 + u2) / 2;  
    t(t_index) = t1;  
    t_index = t_index + 1; 
end  

% ���ͼ�� 
new_img = zeros(size(img));  
new_img(img >= t1) = 0; 
new_img(img < t1) = 1; 

t = t(1:t_index-1); 

% ����ֵ
OptimalThreshold_result=new_img;
t_changing=t;

end