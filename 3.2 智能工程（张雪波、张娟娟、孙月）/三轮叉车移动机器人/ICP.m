clear; clc; close all;
%% 数据生成
numPoints = 100; % 点云数量
theta = pi/6; % 旋转角度
R_true = [cos(theta) -sin(theta); % 真实旋转矩阵
         sin(theta)  cos(theta)];
t_true = [0.5; 0.3]; % 真实平移量
% 源点云（基础矩形随机扰动）
src = [-1 1 1 -1; -1 -1 1 1];
src = repmat(src,1,numPoints/4) + 0.5*randn(2,numPoints);
% 目标点云（变换、噪声）
dst = R_true * src + t_true + 0.02*randn(2,numPoints);
%% ICP
% 参数
maxIter = 50;     % 最大迭代次数
tolerance = 1e-6; % 收敛阈值
prevError = inf;  % 初始误差
T = eye(3);       % 初始变换矩阵
aligned = src;    % 初始对齐点云
for iter = 1:maxIter
    [idx, matched_dst] = Neighbor(aligned, dst); % 最近邻匹配
    [R, t] = SVD(aligned, matched_dst); % SVD变换
    % 应用变换
    T = [R t; 0 0 1] * T;      % 累积变换
    aligned = R * aligned + t; % 更新对齐点云
    % 误差
    dist = sqrt(sum((aligned - matched_dst).^2, 1));
    meanError = mean(dist);
    fprintf('Iter %d: Error = %.4f\n', iter, meanError);
    % 判敛
    if abs(prevError - meanError) < tolerance
        break;
    end
    prevError = meanError;
end
%% 结果显示
R_est = T(1:2, 1:2);
t_est = T(1:2, 3);
theta_est = atan2(R_est(2,1), R_est(1,1)) * (180 / pi);
fprintf('真实旋转角度: %.2f°\n', theta * (180 / pi));
fprintf('估计旋转角度: %.2f°\n', theta_est);
fprintf('真实平移量: [%.2f, %.2f]\n', t_true);
fprintf('估计平移量: [%.2f, %.2f]\n', t_est);
% 旋转矩阵误差（Frobenius范数）
rotation_error = norm(R_true - R_est, 'fro');
fprintf('旋转矩阵误差: %.4f\n', rotation_error);
% 平移量误差（欧几里得范数）
translation_error = norm(t_true - t_est);
fprintf('平移向量误差: %.4f\n', translation_error);
% 可视化
figure;
hold on; axis equal; grid on;
% 源点云（红点），目标点云（蓝点），对齐点云（绿星）
plot(src(1,:), src(2,:), 'r.', 'MarkerSize', 10); 
plot(dst(1,:), dst(2,:), 'b.', 'MarkerSize', 10);
plot(aligned(1,:), aligned(2,:), 'g*', 'MarkerSize', 8);
legend('源点云', '目标点云', '对齐点云');
title('ICP点云配准结果');
% 对齐线
for i = 1:size(aligned, 2)
    plot([aligned(1,i), dst(1,idx(i))], [aligned(2,i), dst(2,idx(i))], ...
          'k--', 'LineWidth', 0.5, 'HandleVisibility', 'off');
end
%% 最近邻搜索函数
function [idx, matched_dst] = Neighbor(src, dst)
    distances = sqrt((src(1,:)' - dst(1,:)).^2 + (src(2,:)' - dst(2,:)).^2);
    [~, idx] = min(distances, [], 2);
    matched_dst = dst(:, idx(:));
end
%% SVD点集配准函数
function [R, t] = SVD(src, dst)
    % 计算质心（权重相同）
    center_src = mean(src, 2);
    center_dst = mean(dst, 2);
    % 去中心化
    src_centered = src - center_src;
    dst_centered = dst - center_dst;
    % SVD分解
    H = src_centered * dst_centered';
    [U,~,V] = svd(H);
    % 旋转矩阵
    R = V * U';
    if det(R) < 0 % 保证单位阵（正）
        V(:,2) = -V(:,2);
        R = V * U';
    end
    % 平移向量
    t = center_dst - R * center_src;
end