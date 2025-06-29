clear; clc; close all;
%% 数据生成
num = 100; % 点数（可变）
% 目标直线
a = 3; % 参数（可变）
b = 2;
x_ori = zeros(2, num);
x_ori(1,:) = 1:num;
x_ori(2,:) = a + b:a:num * a + b;
x_obs = x_ori + randn(2, num); % 加噪音
% 外点
rate = 0.3; % 外点率（可变）
num_off = round(num * rate); 
scale = [max(x_ori(1,:))-min(x_ori(1,:)); max(x_ori(2,:))-min(x_ori(2,:))];
x_off = rand(2, num_off) .* scale;
x_obs = [x_obs, x_off];
all_points = size(x_obs, 2);
% 按x排序
[~, idx] = sort(x_obs(1,:));
x_sorted = x_obs(:, idx);
%% 法1：split-and-merge
tic;
% 参数
dist_threshold1 = 0.5; % 距离阈值（可变）
min_points = 5;        % 分段最少点数（可变）
segments = {x_sorted};
% Split
split_done = false;
while ~split_done
    split_done = true;
    new_segments = {};
    for i = 1:length(segments)
        current_segment = segments{i}; 
        % 防止过度分割
        if size(current_segment, 2) < min_points
            new_segments{end + 1} = current_segment;
            continue;
        end
        % 端点拟合
        p1 = current_segment(:, 1);
        p2 = current_segment(:, end);
        dx = p2(1) - p1(1);
        dy = p2(2) - p1(2);
        % 点集到直线距离最远点
        distances = abs(dy * current_segment(1, :) - dx * current_segment(2, :) + (p2(1) * p1(2) - p2(2) * p1(1))) / sqrt(dx^2 + dy^2);
        [max_dist, max_idx] = max(distances);
        % 大于阈值分裂
        if max_dist > dist_threshold1
            split_done = false;
            new_segments{end + 1} = current_segment(:, 1:max_idx);
            new_segments{end + 1} = current_segment(:, max_idx:end);
        else
            new_segments{end + 1} = current_segment;
        end
    end
    segments = new_segments;
end
% Merge
merge_done = false;
while ~merge_done
    merge_done = true;
    new_segments = {};
    i = 1;
    while i <= length(segments)
        % 处理最后一个段
        if i == length(segments)
            new_segments{end + 1} = segments{i};
            break;
        end
        % 尝试合并当前段和下一段
        seg1 = segments{i};
        seg2 = segments{i + 1};
        combined_seg = [seg1, seg2];
        % 端点拟合
        p1 = combined_seg(:, 1);
        p2 = combined_seg(:, end);
        dx = p2(1) - p1(1);
        dy = p2(2) - p1(2);
        % 点集到直线距离最远点
        distances = abs(dy * combined_seg(1, :) - dx * combined_seg(2, :) + (p2(1) * p1(2) - p2(2) * p1(1))) / sqrt(dx^2 + dy^2);
        [max_dist,~] = max(distances);
        % 判断是否执行合并
        if max_dist < dist_threshold1
            merge_done = false;
            new_segments{end + 1} = combined_seg;
            i = i + 2;
        else
            new_segments{end + 1} = seg1;
            i = i + 1;
        end
    end
    segments = new_segments;
end
time1 = toc;
fprintf('Split-and-Merge耗时%.4f秒。\n', time1);
% 可视化
figure(1);
hold on; grid on;
scatter(x_obs(1,:), x_obs(2,:)); 
title('Split-and-Merge');
colors = lines(length(segments)); % 分配颜色
for i = 1:length(segments)
    seg = segments{i};
    if size(seg, 2) > 2 % 去除部分外点线段
        scatter(seg(1,:), seg(2,:), 'filled', 'MarkerFaceColor', colors(i,:));
        p1 = seg(:,1);
        p2 = seg(:,end);
        % 计算斜率时防止除零
        if abs(p2(1) - p1(1)) < 1e-6
            a_seg = Inf; 
        else
            a_seg = (p2(2) - p1(2)) / (p2(1) - p1(1));
        end
        b_seg = p1(2) - a_seg * p1(1);
        x_vals = [min(seg(1,:)), max(seg(1,:))];
        y_vals = a_seg * x_vals + b_seg;
        plot(x_vals, y_vals, 'Color', colors(i,:), 'LineWidth', 2);
    end
end
%% 法2：Line-Regression
tic;
% 参数
window_size = 20; % 滑动窗口大小（可变）
step = 5; % 滑动步长（可变）
merge_angle_threshold = 30; % 合并角度阈值(°)（可变）
merge_dist_threshold = 10; % 合并距离阈值（可变）
segments = struct('a',[],'b',[],'x_range',[],'points',[]); % 存储结构
% 滑动窗口拟合
count = 0;
for i = 1:step:(length(x_sorted) - window_size + 1)
    % 提取当前窗口数据
    window_idx = i : (i + window_size - 1);
    x_window = x_sorted(:, window_idx)';
    % 最小二乘法拟合
    A = [x_window(:,1), ones(window_size,1)];
    beta = (A'*A) \ (A'*x_window(:,2));
    % 保存线段
    count = count + 1;
    segments(count).a = beta(1);
    segments(count).b = beta(2);
    segments(count).x_range = [x_window(1,1), x_window(end,1)];
    segments(count).points = x_window;
end
% 合并
changed = true; 
while changed % 外层循环：反复从头尝试合并
    changed = false;
    i = 1;
    n = length(segments);
    while i < n - 1 % 内层循环：相邻尝试合并
        current = segments(i);
        next = segments(i + 1);
        % 计算角度差
        angle_diff = rad2deg(abs(atan(current.a) - atan(next.a)));
        % 计算中点到对方直线的距离
        mid_x = mean(current.x_range);
        mid_y = current.a * mid_x + current.b;
        dist = abs(next.a * mid_x - mid_y + next.b) / sqrt(next.a^2 + 1);
        % 合并条件判断
        if angle_diff < merge_angle_threshold && dist < merge_dist_threshold
            % 合并两个线段
            merged_points = [current.points; next.points];
            A = [merged_points(:,1), ones(size(merged_points,1),1)];
            beta = (A'*A) \ (A'*merged_points(:,2));
            segments(i).a = beta(1);
            segments(i).b = beta(2);
            segments(i).x_range = [current.x_range(1), next.x_range(2)];
            segments(i).points = merged_points;
            segments(i + 1) = [];
            n = n - 1; 
            changed = true;
        else % 不合并移动到下一个
            i = i + 1; 
        end
    end
end
time2 = toc;
fprintf('Line-Regression耗时%.4f秒。\n', time2);
% 可视化
figure(2); 
hold on; grid on;
scatter(x_obs(1,:), x_obs(2,:));
title('Line-Regression');
for k = 1:length(segments)
    seg = segments(k);
    x_plot = linspace(seg.x_range(1), seg.x_range(2), 100);
    y_plot = seg.a * x_plot + seg.b;
    plot(x_plot, y_plot, 'r-', 'LineWidth', 2);
end
%% 法3：RANSAC
tic;
% 参数
dist_threshold2 = 0.1; % 距离阈值（可变）
p = 0.99; % 找到一个完全由内点组成的样本的期望概率（可变）
iter_num = log(1 - p) / log(1 - (1 - rate)^2); % 迭代次数
for i = 1:iter_num
    count = 0;
    count_old = 0;
    % 随机选点（未防止重复选择）
    x1 = x_obs(:, randi(length(x_obs)));
    x2 = x_obs(:, randi(length(x_obs)));
    while norm(x1 - x2) < 1 % 防止选点过近（确保斜率）
        x2 = x_obs(:, randi(length(x_obs)));
    end
    % 统计内点数
    for j = 1:length(x_obs)
        distances = abs((x2(2) - x1(2)) * x_obs(1,:) - (x2(1) - x1(1)) * x_obs(2,:) + x2(1) * x1(2) - x2(2) * x1(1)) ...
            ./ sqrt((x2(2) - x1(2))^2 + (x2(1) - x1(1))^2);
        count = sum(distances < dist_threshold2);
    end
    % 更新最佳点组合
    if count > count_old
        count_old = count;
        x1_best = x1;
        x2_best = x2;
    end
end
time3 = toc; 
fprintf('RANSAC耗时%.4f秒。\n', time3);
% 可视化
figure(3);
hold on; grid on;
scatter(x_obs(1,:), x_obs(2,:));
title('RANSAC');
a_tilde = (x1(2) - x2(2)) / (x1(1) - x2(1));
b_tilde = x1(2) - a_tilde * x1(1);
plot([1, max(x_obs(1,:))], [a_tilde + b_tilde, a_tilde * max(x_obs(1,:)) + b_tilde], 'k-', 'LineWidth', 2);
plot(x1_best(1), x1_best(2), 'rx');
plot(x2_best(1), x2_best(2), 'rx');
legend('点云', '拟合直线', '最优点组合');
%% 法4：Hough-Transform
tic;
% 参数
theta_res = 1; % 角度分辨率（可变）
rho_res = 2; % 距离分辨率（可变）
peak_thresh = 0.8; % 累加器峰值检测阈值（检测多直线）（可变）
% 数据范围
x_range = max(x_obs(1,:)) - min(x_obs(1,:));
y_range = max(x_obs(2,:)) - min(x_obs(2,:));
max_dist = ceil(sqrt(x_range^2 + y_range^2));
theta = deg2rad(-90:theta_res:90-theta_res); % 角度范围
rho = -max_dist:rho_res:max_dist; % 距离范围
accumulator = zeros(length(rho), length(theta)); % 累加器
% 投票
[THETA, RHO] = meshgrid(theta, rho);
for i = 1:all_points
    rho_vals = x_obs(1,i)*cos(THETA) + x_obs(2,i)*sin(THETA);
    rho_diff = abs(rho_vals - RHO);
    [~, idx] = min(rho_diff, [], 1);
    for t = 1:length(theta)
        accumulator(idx(t), t) = accumulator(idx(t), t) + 1;
    end
end
% 寻找峰值
threshold = peak_thresh * max(accumulator(:));
[rho_idxs, theta_idxs] = find(accumulator >= threshold);
time4 = toc; 
fprintf('Hough Transform耗时%.4f秒。\n', time4);
% 可视化
figure(4);
hold on; grid on;
scatter(x_obs(1,:), x_obs(2,:)); 
title('Hough Transform');
for i = 1:length(rho_idxs)
    rho_val = rho(rho_idxs(i));
    theta_val = theta(theta_idxs(i));
    if abs(theta_val - pi/2) < deg2rad(10) % 垂直
        x1 = rho_val / cos(theta_val);
        x2 = x1;
        y1 = min(x_obs(2,:));
        y2 = max(x_obs(2,:));
    else
        x1 = min(x_obs(1,:));
        y1 = (rho_val - x1*cos(theta_val)) / sin(theta_val);
        x2 = max(x_obs(1,:));
        y2 = (rho_val - x2*cos(theta_val)) / sin(theta_val);
    end
    plot([x1 x2], [y1 y2], 'g-', 'LineWidth', 2);
end
legend('数据点', '直线');
% 参数空间
figure(5);
imagesc(rad2deg(theta), rho, accumulator);
colormap(hot); colorbar; hold on;
title('Hough参数空间');
xlabel('角度θ (度)'); ylabel('距离ρ');
for i = 1:length(rho_idxs)
    plot(rad2deg(theta(theta_idxs(i))), rho(rho_idxs(i)), ...
        'o', 'Color', colors(i,:), 'MarkerSize', 10, 'LineWidth', 2);
    text(rad2deg(theta(theta_idxs(i))), rho(rho_idxs(i)), ...
        sprintf('%d', i), 'Color', 'white', 'FontWeight', 'bold');
end