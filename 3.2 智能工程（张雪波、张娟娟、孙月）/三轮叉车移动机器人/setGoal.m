clear; clc; close all;
%% 参数
DELTA = 0.001; % 更新步长
k_rho = 3;     % 距离误差更新幅度
k_alpha = 8;   % 方位角误差更新幅度
k_beta = -1.5; % 俯仰角误差更新幅度
n = 10000;     % 循环次数
% 目标（惯性系）
x_e = 0;       
y_e = 0;
theta_e = pi/2;
% 起点（惯性系）
x_s = 1;
y_s = 1;
theta_s = pi/2;
%% 误差转化
% 旋转阵（目标到惯性）
R = [cos(theta_e), -sin(theta_e), 0; 
     sin(theta_e),  cos(theta_e), 0;
     0,             0,            1];
% 相对误差：惯性系->目标系->极坐标系
q = [x_s - x_e; y_s - y_e; theta_s - theta_e];
e = R' * q;
rho = sqrt(e(1)^2 + e(2)^2);
beta = -atan2(-e(2), -e(1));
alpha = -beta - e(3);
%% 控制器
X = [];
Y = [];
Theta = [];
for i = 1:n
    if alpha >= 0.1  
        % 线性控制器
        rho = rho + DELTA * (-k_rho*rho*cos(alpha));
        beta = beta + DELTA * (-k_rho*sin(alpha));
        alpha = alpha + DELTA * (k_rho*sin(alpha) - k_alpha*alpha - k_beta*beta);
        % 非线性控制器
        rho = rho + DELTA * (-k_rho*rho*cos(alpha)*cos(alpha));
        beta = beta + DELTA * (-k_rho*cos(alpha)*sin(alpha));
        alpha = alpha + DELTA * (k_rho*cos(alpha)*sin(alpha) - k_alpha*alpha - k_rho*sin(alpha)*cos(alpha)/alpha*(alpha-k_beta*beta));
    else % 角小时
        % 线性控制器
        rho = rho + DELTA * (-k_rho*rho);
        beta = beta + DELTA * (-k_rho*alpha);
        alpha = alpha + DELTA * (k_rho*alpha - k_alpha*alpha - k_beta*beta);
        % 非线性控制器
        rho = rho + DELTA * (-k_rho*rho);
        beta = beta + DELTA * (-k_rho*alpha);
        alpha = alpha + DELTA * (-k_alpha*alpha + k_rho*k_beta*beta);
    end
    % 增量转化：极坐标系->目标系->惯性系（beta逆时针）
    x_d = rho * -cos(-beta);
    y_d = rho * -sin(-beta);
    theta_d = -alpha - beta;
    p_i = R * [x_d; y_d; theta_d];
    % 更新位姿
    X = [X, p_i(1)];
    Y = [Y, p_i(2)];
    Theta = [Theta, p_i(3)];
end
%% 绘图
figure;
hold on; grid on;
plot(X, Y, 'b-', 'LineWidth', 1);
quiver(x_s, y_s, cos(theta_s), sin(theta_s), 0.2, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
quiver(x_e, y_e, cos(theta_e), sin(theta_e), 0.2, 'g', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
xlabel('X坐标');
ylabel('Y坐标');
title('定点控制器效果图');
legend('轨迹', '起点', '目标', 'Location', 'southeast');
hold off; 