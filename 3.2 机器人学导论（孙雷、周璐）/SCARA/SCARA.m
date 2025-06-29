clc; clear;
%% 机器人模型（改进DH参数）
L1 = Revolute('d', 0,      'a', 0,   'alpha', 0,   'modified');
L2 = Revolute('d', 41,     'a', 225, 'alpha', 0,   'modified'); 
L3 = Revolute('d', 17,     'a', 275, 'alpha', 0,   'modified'); 
L4 = Prismatic('theta', 0, 'a', 0,   'alpha', pi,  'modified', 'qlim', [102, 302]); 
robot = SerialLink([L1, L2, L3, L4]); % 串联
robot.name = '3RP SCARA';
robot.display();
%% 正运动学
% 随机生成关节变量
thetaConfig = randn(1, 4);
q1 = thetaConfig(1);
q2 = thetaConfig(2);
q3 = thetaConfig(3);
d4 = thetaConfig(4)*200 + 102;
thetaConfig = [q1, q2, q3, d4];
% 理论推导
T_theoretical = [cos(q1+q2+q3), sin(q1+q2+q3),  0,  275*cos(q1+q2) + 225*cos(q1);
                 sin(q1+q2+q3), -cos(q1+q2+q3), 0,  275*sin(q1+q2) + 225*sin(q1);
                 0,             0,              -1, -d4 + 58;
                 0,             0,              0,  1];
disp('===== 理论推导求解正运动学 =====');
disp(T_theoretical);
% 工具箱求解
T_toolbox = robot.fkine(thetaConfig);
disp('===== 工具箱求解正运动学 =====');
disp(T_toolbox);
disp('===== 工具箱求解正运动学（保留小数点后四位） =====');
Tmat = T_toolbox.T; % 提取矩阵，保留小数点后四位
Tmat_4 = round(Tmat, 4);
disp(Tmat_4);
%% 逆运动学
% 提取计算信息
Px = T_toolbox.t(1);
Py = T_toolbox.t(2);
Pz = T_toolbox.t(3);
r11 = T_toolbox.n(1);
r21 = T_toolbox.n(2);
denominator = Px^2 + Py^2;
if abs(denominator - 126250) > 123750
    error('超出工作空间，无解');
end
% q2（双解）
radicand = (Px^2 + Py^2 - 126250) / 123750;
q2_sol1 = acos(radicand);  
q2_sol2 = -acos(radicand);  
% q1
[sin_q2_1, cos_q2_1] = deal(sin(q2_sol1), cos(q2_sol1));
s1_sol1 = ((275*cos_q2_1 + 225)*Py - 275*Px*sin_q2_1) / denominator;
c1_sol1 = ((275*cos_q2_1 + 225)*Px + 275*Py*sin_q2_1) / denominator;
q1_sol1 = atan2(s1_sol1, c1_sol1);
[sin_q2_2, cos_q2_2] = deal(sin(q2_sol2), cos(q2_sol2));
s1_sol2 = ((275*cos_q2_2 + 225)*Py - 275*Px*sin_q2_2) / denominator;
c1_sol2 = ((275*cos_q2_2 + 225)*Px + 275*Py*sin_q2_2) / denominator;
q1_sol2 = atan2(s1_sol2, c1_sol2);
% q3
q3_sol1 = atan2(r21, r11) - q1_sol1 - q2_sol1;
q3_sol2 = atan2(r21, r11) - q1_sol2 - q2_sol2;
% d4
d4_sol = 58 - Pz;
disp('===== 原始关节输入 =====');
fprintf('[θ1=%.4f, θ2=%.4f, θ3=%.4f, d4=%.4f]\n', thetaConfig);
% 代数法求解
inverseSolution1 = [q1_sol1, q2_sol1, q3_sol1, d4_sol];
inverseSolution2 = [q1_sol2, q2_sol2, q3_sol2, d4_sol];
disp('===== 代数法求解逆运动学 =====');
fprintf('解1: [θ1=%.4f, θ2=%.4f, θ3=%.4f, d4=%.4f]\n', inverseSolution1);
fprintf('解2: [θ1=%.4f, θ2=%.4f, θ3=%.4f, d4=%.4f]\n', inverseSolution2);
% 工具箱求解
q = [0 0 0 0]; % 初始猜测
q_toolbox = robot.ikunc(T_toolbox, q); % 接近初始猜测的一个解
disp('===== 工具箱求解逆运动学 =====');
disp(q_toolbox);
% 可视化（防止覆盖，立即保存）
xyzlim = [-500 500 -500 500 -500 500]; % 坐标轴范围
figure;
robot.plot(thetaConfig);
title('构型可视化');
saveas(gcf, '构型可视化.png');
figure;
robot.plot(inverseSolution1);
title('代数法1');
saveas(gcf, '代数法1.png');
figure;
robot.plot(inverseSolution2);
title('代数法2');
saveas(gcf, '代数法2.png'); 
figure;
robot.plot(q_toolbox);
title('工具箱求解');
saveas(gcf, '工具箱求解.png');