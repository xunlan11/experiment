clear; clc; close all;
%% 参数
dt = 0.1;         % 时间步长
T = 20;           % 总仿真时间
steps = T/dt;     % 总步数
sigma_a = 0.2;    % 加速度噪声标准差
sigma_meas = 0.5; % 测量噪声标准差
%% 卡尔曼滤波参数矩阵
% 状态向量[x; y; vx; vy]
% 状态转移矩阵（匀速模型）
F = [1 0 dt 0; 
     0 1 0 dt; 
     0 0 1 0; 
     0 0 0 1];
% 控制输入（加速度）矩阵 
B = [0.5*dt^2 0; 
     0 0.5*dt^2; 
     dt 0; 
     0 dt];
% 过程噪声（加速度噪声）协方差
G = [0.5*dt^2 0; 
     0 0.5*dt^2; 
     dt 0; 
     0 dt];
Q = G * diag([sigma_a^2, sigma_a^2]) * G';
% 测量矩阵（只观测位置）
H = [1 0 0 0; 
     0 1 0 0];
% 测量噪声协方差
R = diag([sigma_meas^2, sigma_meas^2]);
%% 初始化
x0 = [0; 0; 1; 0.5];
P = diag([0.5, 0.5, 0.2, 0.2]); 
true_state = zeros(4, steps);   
true_state(:,1) = x0;
meas = zeros(2, steps);      
est_state = zeros(4, steps);   
est_state(:,1) = x0;
acceleration = 0.5*[sin(0.1*(1:steps)*dt); cos(0.2*(1:steps)*dt)]; % 加速度随时间变化
%% 主循环
for k = 2:steps
    % 随机加速度扰动
    acc_noise = sigma_a * randn(2,1);
    u = acceleration(:,k-1) + acc_noise;
    % 更新真实状态
    true_state(:,k) = F * true_state(:,k-1) + B * u;
    % 生成带噪声的测量
    meas_noise = sigma_meas * randn(2,1);
    meas(:,k) = H * true_state(:,k) + meas_noise;
    % 预测
    pred_state = F * est_state(:,k-1) + B * u;
    P_pred = F * P * F' + Q;
    % 更新
    K = P_pred * H' / (H * P_pred * H' + R);
    est_state(:,k) = pred_state + K * (meas(:,k) - H * pred_state);
    P = (eye(4) - K * H) * P_pred;
end
%% 可视化
t = dt:dt:T;
figure('Position', [100, 100, 1200, 800]);
subplot(2,2,[1,3]);
hold on;
plot(true_state(1,:), true_state(2,:), 'b-', 'LineWidth', 1.5);
plot(meas(1,:), meas(2,:), 'ro', 'MarkerSize', 4, 'MarkerFaceColor', 'r');
plot(est_state(1,:), est_state(2,:), 'g--', 'LineWidth', 1.5);
legend('真实轨迹', '观测值', '卡尔曼估计', 'Location', 'best');
title('机器人运动轨迹');
xlabel('X位置'); ylabel('Y位置');
grid on; axis equal;
subplot(2,2,2);
hold on;
plot(t, true_state(1,:), 'b-', 'LineWidth', 1.5);
plot(t, meas(1,:), 'ro', 'MarkerSize', 4);
plot(t, est_state(1,:), 'g--', 'LineWidth', 1.5);
title('X方向位置估计');
xlabel('时间'); ylabel('X位置');
legend('真实值', '观测值', '估计值');
grid on;
subplot(2,2,4);
hold on;
plot(t, true_state(2,:), 'b-', 'LineWidth', 1.5);
plot(t, meas(2,:), 'ro', 'MarkerSize', 4);
plot(t, est_state(2,:), 'g--', 'LineWidth', 1.5);
title('Y方向位置估计');
xlabel('时间 (s)'); ylabel('Y位置');
legend('真实值', '观测值', '估计值');
grid on;
%% 误差分析
pos_error = sqrt((true_state(1,:) - est_state(1,:)).^2 + (true_state(2,:) - est_state(2,:)).^2);
figure;
plot(t, pos_error, 'LineWidth', 1.5);
title('位置估计误差');
xlabel('时间 (s)'); ylabel('误差');
grid on;
fprintf('平均位置误差: %.4f \n', mean(pos_error));