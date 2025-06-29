clear; clc; close all;
%% 参数
r = 1;          % 轮半径
l = 2;          % 轮到质心距离
dot_phi = pi;   % 轮角速度
% 曲线
beta = pi/4;    % 舵机角度
v = r * dot_phi * sin(beta);         % 线速度
omega = r * dot_phi * cos(beta) / l; % 角速度
% 直线
%beta = pi/2;     % 舵机角度
%v = r * dot_phi; % 线速度
%omega = 0;       % 角速度
k_phi = 0.001;  % 车轮转速误差系数
k_beta = 0.001; % 舵机角度误差系数
dt = 0.001;     % 时间步长
t = 0:dt:1;
% 初始条件
x = zeros(size(t));
y = zeros(size(t));
theta = zeros(size(t));
x(1) = 0;
y(1) = 0;
theta(1) = 0;
%% 解析积分法
x_exact = x;
y_exact = y;
theta_exact = theta;
for k = 1:(length(t)-1)
    if abs(omega) == 0 % 角速度为0
        x_exact(k+1) = x_exact(k) + v * dt;
        y_exact(k+1) = y_exact(k);
    else
        theta_exact(k+1) = theta_exact(k) + omega * dt;
        x_exact(k+1) = x_exact(k) + v / omega * (sin(theta_exact(k+1)) - sin(theta_exact(k)));
        y_exact(k+1) = y_exact(k) - v / omega * (cos(theta_exact(k+1)) - cos(theta_exact(k)));
    end
end
%% 欧拉法
x_euler = x;
y_euler = y;
theta_euler = theta;
for k = 1:(length(t)-1)
    x_euler(k+1) = x_euler(k) + v * dt * cos(theta_euler(k));
    y_euler(k+1) = y_euler(k) + v * dt * sin(theta_euler(k));
    theta_euler(k+1) = theta_euler(k) + omega * dt;
end
error_euler = sqrt((x_euler - x_exact).^2 + (y_euler - y_exact).^2);
%% 二阶Runge-Kutta法
x_rk2 = x;
y_rk2 = y;
theta_rk2 = theta;
F_p = zeros(3, 3, length(t)); 
F_rl = zeros(3, 2, length(t));
for k = 1:(length(t)-1)
    s = v * dt;
    theta_rk = theta_rk2(k) + dt / 2 * omega;
    x_rk2(k+1) = x_rk2(k) + s * cos(theta_rk);
    y_rk2(k+1) = y_rk2(k) + s * sin(theta_rk);
    theta_rk2(k+1) = theta_rk2(k) + dt * omega;
    % 误差传导矩阵
    F_p(:, :, k) = [
        1, 0, -s * sin(theta_rk);
        0, 1, s * cos(theta_rk);
        0, 0, 1];
    F_rl(:, :, k) = [
        r * sin(beta) * cos(theta_rk) - r^2 * dot_phi * cos(beta) * sin(beta) / (2 * l) * sin(theta_rk), ...
        r * dot_phi * cos(beta) * cos(theta_rk) + r^2 * dot_phi^2 * sin(beta)^2 / (2 * l) * sin(theta_rk);
        r * sin(beta) * sin(theta_rk) + r^2 * dot_phi * cos(beta) * sin(beta) / (2 * l) * cos(theta_rk), ...
        r * dot_phi * cos(beta) * sin(theta_rk) - r^2 * dot_phi^2 * sin(beta)^2 / (2 * l) * cos(theta_rk);
        r * cos(beta) / l, -r * dot_phi * sin(beta) / l];
end
error_rk2 = sqrt((x_rk2 - x_exact).^2 + (y_rk2 - y_exact).^2);
%% 协方差矩阵
P = zeros(3, 3, length(t));
delta = [k_phi * dt * dot_phi, 0;
    0, k_beta];
for k = 1:(length(t)-1)
    F_p_k = squeeze(F_p(:, :, k));
    F_rl_k = squeeze(F_rl(:, :, k));
    P(:, :, k+1) = F_p_k * P(:, :, k) * F_p_k' + F_rl_k * delta * F_rl_k';
end
%% 绘制结果
figure;
% 结果
subplot(2,1,1);
grid on;
plot(x_exact, y_exact, 'b', 'LineWidth', 2); hold on;
plot(x_euler, y_euler, 'r--', 'LineWidth', 2);
plot(x_rk2, y_rk2, 'g-.', 'LineWidth', 2);
xlabel('X');
ylabel('Y');
legend('解析积分法', '欧拉法', '二阶Runge-Kutta法');
% 误差
subplot(2,1,2);
grid on;
semilogy(t, error_euler, 'r', 'LineWidth', 2); hold on;
semilogy(t, error_rk2, 'g', 'LineWidth', 2);
xlabel('时间(s)');
ylabel('误差');
legend('欧拉法', '二阶Runge-Kutta法');
% 误差传导
figure;
hold on; grid on; axis equal;
plot(x_rk2, y_rk2, 'b', 'LineWidth', 2); 
xlabel('X');
ylabel('Y');
title('误差传导（二阶Runge-Kutta法）');
for k = 1:100:length(t) 
    P_k = squeeze(P(1:2, 1:2, k));
    [V, D] = eig(P_k); % 矩阵特征
    a = sqrt(0.005 * D(1,1)); % 长轴
    b = sqrt(0.005 * D(2,2)); % 短轴
    angle = atan2(V(2, 1), V(1, 1)); % 方向角
    R = [cos(angle), -sin(angle); sin(angle), cos(angle)]; % 旋转矩阵
    t_ellipse = linspace(0, 2*pi, 100); 
    x_ellipse = a * cos(t_ellipse);
    y_ellipse = b * sin(t_ellipse);
    xy_ellipse = R * [x_ellipse(:), y_ellipse(:)]';
    plot(x_rk2(k) + xy_ellipse(1, :), y_rk2(k) + xy_ellipse(2, :), 'k');
end
legend('轨迹', '误差');