clear; clc; close all;
%% 基于SVD的机器人定位算法
function [R, t] = svd_localization(P, Q, w)
    % 默认等权重
    if nargin < 3
        w = ones(1, size(P, 2)); 
    end
    % 计算加权中心点
    sum_w = sum(w);
    hat_p = (P * w') / sum_w;
    hat_q = (Q * w') / sum_w;
    % 去中心化
    X = P - hat_p;
    Y = Q - hat_q;
    % 构造S矩阵并进行SVD分解
    W = diag(w); % 权重对角阵
    S = X * W * Y';
    [U, ~, V] = svd(S);
    % 旋转矩阵
    R = V * U';
    if det(R) < 0 % 保证单位阵
        V(:, end) = -V(:, end);
        R = V * U';
    end
    % 平移向量
    t = hat_q - R * hat_p;
end
%% 结果对比
function display_results(R_true, t_true, R_est, t_est)
    theta_true = atan2(R_true(2,1), R_true(1,1)) * (180 / pi);
    theta_est = atan2(R_est(2,1), R_est(1,1)) * (180 / pi);
    fprintf('真实旋转角度: %.2f°\n', theta_true);
    fprintf('估计旋转角度: %.2f°\n', theta_est);
    fprintf('真实平移量: [%.2f, %.2f]\n', t_true);
    fprintf('估计平移量: [%.2f, %.2f]\n', t_est);
    % 旋转矩阵误差（Frobenius范数）
    rotation_error = norm(R_true - R_est, 'fro');
    fprintf('旋转矩阵误差: %.4f\n', rotation_error);
    % 平移量误差（欧几里得范数）
    translation_error = norm(t_true - t_est);
    fprintf('平移向量误差: %.4f\n', translation_error);
end
%% 点集匹配可视化
function visualize_point_sets(P, Q, R, t, title_str)
    figure;
    hold on; grid on; axis equal;
    % 原始点集P（蓝色），目标点集Q（红色）
    scatter(P(1,:), P(2,:), 100, 'b', 'filled', 'DisplayName', '原始点集P');
    scatter(Q(1,:), Q(2,:), 100, 'r', 'filled', 'DisplayName', '目标点集Q');
    % 变换后点集P（绿色）
    P_transformed = R * P + t;
    scatter(P_transformed(1,:), P_transformed(2,:), 80, 'g', 'o', 'DisplayName', '变换后的P');
    % 对齐线
    for i = 1:size(P,2)
        plot([P_transformed(1,i), Q(1,i)], [P_transformed(2,i), Q(2,i)], 'k--', 'LineWidth', 0.5,  'HandleVisibility', 'off');
    end
    legend('Location', 'best');
    title(title_str);
end
%% 理想数据测试
fprintf('=== 理想数据测试 ===\n');
n = 100; % 点数
P = rand(2, n); % 世界坐标系
theta = pi/6; % 旋转角度
R_true = [cos(theta), -sin(theta); % 真实旋转矩阵
          sin(theta),  cos(theta)]; 
t_true = [1.5; -0.8]; % 真实平移量
Q = R_true * P + t_true; % 用户坐标系
[R_est, t_est] = svd_localization(P, Q);
display_results(R_true, t_true, R_est, t_est);
visualize_point_sets(P, Q, R_est, t_est, '理想数据匹配结果');
%% 带噪声数据测试
fprintf('\n=== 带噪声数据测试 ===\n');
noise_level = 0.05; % 噪声水平
Q_noisy = Q + noise_level * randn(size(Q));
w = 1./var(Q_noisy, 0, 1); % 根据方差（噪音距离平方）分配权重，方差小的权重高
[R_est_noisy, t_est_noisy] = svd_localization(P, Q_noisy, w);
display_results(R_true, t_true, R_est_noisy, t_est_noisy);
visualize_point_sets(P, Q_noisy, R_est_noisy, t_est_noisy, '带噪声数据匹配结果');