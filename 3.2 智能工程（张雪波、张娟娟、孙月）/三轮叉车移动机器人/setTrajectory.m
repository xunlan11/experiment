clear; clc; close all;
%% 参数           
DELTA = 0.001; % 更新步长
r = 1;         % 轮胎半径
dot_phi = pi;  % 轮胎角速度
l = 2;         % 轮胎到质心的距离
k_1 = 5;
k_2 = 1;
n = 10000;     % 循环次数
% 起点
x_s = 0; 
y_s = 0; 
theta_s = 0;
%% 生成期望轨迹（惯性系）
X = [];     dot_X = []; 
Y = [];     dot_Y = []; 
Theta = []; dot_Theta = [];
x = x_s;
y = y_s;
theta = theta_s;
for i = 1:n
    % 轨迹生成
    t = i * DELTA;
    if t < 2 % 初始直线段 
        v = 1.0;
        omega = 0;
    elseif t < 6 % 左转圆弧
        v = 1.0;
        omega = 0.5;
    elseif t < 10 % 右转圆弧
        v = 1.0;
        omega = -0.5;
    elseif t < 15 % 正弦波轨迹
        v = 1.2;
        omega = 0.4 * cos(0.8 * (t-10));
    else % 最终直线段
        v = 1.0;
        omega = 0;
    end
    % 轨迹保存
    dot_x = v * cos(theta);
    dot_y = v * sin(theta);
    x = x + dot_x * DELTA;
    y = y + dot_y * DELTA;
    theta = theta + omega * DELTA;
    X = [X, x];             dot_X = [dot_X, dot_x];
    Y = [Y, y];             dot_Y = [dot_Y, dot_y];
    Theta = [Theta, theta]; dot_Theta = [dot_Theta, omega];
end
%% 轨迹跟踪
XX = [];
YY = [];
TTheta = [];
pos_error = []; 
angle_error = [];         
x = x_s;
y = y_s;
theta = theta_s;
for i = 1:n
    % 旋转阵（机器人到惯性）
    R = [cos(theta), -sin(theta), 0;
         sin(theta), cos(theta),  0;
         0,           0,          1];
    % 误差信号：惯性系->机器人系
    q_d = [x - X(i); y - Y(i); theta - Theta(i)];
    e = R' * q_d;
    % 控制器
    if e(3) >= 0.0001       
        v = [-k_1*e(1) + dot_X(i)*cos(e(3)); 
             -dot_X(i)*sin(e(3))/e(3)*e(2) - k_2*e(3) + dot_Theta(i)];
        dot_e = [-k_1*e(1) + v(2)*e(2); 
                 -v(2)*e(1) + dot_X(i)*sin(e(3)); 
                 -k_2*e(3) - dot_X(i)*sin(e(3))/e(3)*e(2)];
    else % 角小时      
        v = [-k_1*e(1) + dot_X(i); 
             -dot_X(i)*e(2) + dot_Theta(i)];
        dot_e = [-k_1*e(1) + v(2)*e(2); 
                 -v(2)*e(1);  
                 -k_2*e(3) - dot_X(i)*e(2)];
    end
    e = e + dot_e .* DELTA;
    % 更新
    dot_x = v(1) * cos(theta);
    dot_y = v(1) * sin(theta);
    x = x + dot_x * DELTA;
    y = y + dot_y * DELTA;
    theta = theta + v(2) * DELTA;
    % 记录
    XX = [XX, x];
    YY = [YY, y];
    TTheta = [TTheta, theta];
    pos_error = [pos_error, sqrt(e(1)^2 + e(2)^2)]; 
    angle_error = [angle_error, e(3)];
end
%% 绘图
% 轨迹
figure;
hold on; grid on;
plot(X, Y, 'b-', 'LineWidth', 1); 
plot(XX, YY, 'y-', 'LineWidth', 1);
quiver(x_s, y_s, cos(theta_s), sin(theta_s), 1, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
x_e = XX(end);
y_e = YY(end);
theta_e = TTheta(end);
quiver(x_e, y_e, cos(theta_e), sin(theta_e), 1, 'g', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
xlabel('X坐标');
ylabel('Y坐标');
title('轨迹控制器效果图');
legend('轨迹', '跟踪轨迹', 'Location', 'southeast');
hold off; 
% 误差
figure;
hold on; grid on;
t = (0:n-1) * DELTA;
plot(t, pos_error, 'b-', 'LineWidth', 1.2); 
plot(t, angle_error, 'r-', 'LineWidth', 1.2);
xlabel('时间(s)');
ylabel('误差');
title('跟踪误差');
legend('位置误差', '姿态误差');