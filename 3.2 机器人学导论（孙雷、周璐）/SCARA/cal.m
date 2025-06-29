clc; clear;
%% 信息
% 定义符号变量
syms q1 q2 q3 q4 real   
syms m1 m2 m3 m4 real     
syms I1xx I1yy I1zz real  
syms I2xx I2yy I2zz real   
syms I3xx I3yy I3zz real   
syms I4xx I4yy I4zz real  
syms g real
c1 = cos(q1);   
s1 = sin(q1);
c12 = cos(q1+q2); 
s12 = sin(q1+q2);
% 实际参数
n = 4; % 自由度
m1_val = 50;    m2_val = 10;    m3_val = 1;    m4_val = 5;
% 转动惯量矩阵
I1xx_val = 50;  I1yy_val = 50;  I1zz_val = 100;
I2xx_val = 10;  I2yy_val = 10;  I2zz_val = 20;
I3xx_val = 1;   I3yy_val = 1;   I3zz_val = 1;
I4xx_val = 5;   I4yy_val = 5;   I4zz_val = 10;
Ic1 = diag([I1xx, I1yy, I1zz]);
Ic2 = diag([I2xx, I2yy, I2zz]);
Ic3 = diag([I3xx, I3yy, I3zz]);
Ic4 = diag([I4xx, I4yy, I4zz]);
%% 雅可比（质心）
% 连杆1
Jv1 = [-112.5*s1, 0, 0, 0;
       112.5*c1, 0, 0, 0;
       0, 0, 0, 0];
Jw1 = [0, 0, 0, 0;
       0, 0, 0, 0;
       1, 0, 0, 0];
% 连杆2
Jv2 = [-137.5*s12 - 225*s1, -137.5*s12 - 225*s1, 0, 0;
       137.5*c12 + 225*c1, 137.5*c12 + 225*c1, 0, 0;
       0, 0, 0, 0];
Jw2 = [0, 0, 0, 0;
       0, 0, 0, 0;
       1, 1, 0, 0];
% 连杆3
Jv3 = [-275*s12 - 225*s1, -275*s12 - 225*s1, 0, 0;
       275*c12 + 225*c1, 275*c12 + 225*c1, 0, 0;
       0, 0, 0, 0];
Jw3 = [0, 0, 0, 0;
       0, 0, 0, 0;
       1, 1, 1, 0];
% 连杆4
Jv4 = [-275*s12 - 225*s1, -275*s12 - 225*s1, 0, 0;
       275*c12 + 225*c1, 275*c12 + 225*c1, 0, 0;
       0, 0, 0, -0.5];
Jw4 = [0, 0, 0, 0;
       0, 0, 0, 0;
       1, 1, 1, 0];
%% M
M = sym(zeros(4,4));
M = M + simplify(Jv1' * m1 * Jv1 + Jw1' * Ic1 * Jw1 + Jv2' * m2 * Jv2 + Jw2' * Ic2 * Jw2 + Jv3' * m3 * Jv3 + Jw3' * Ic3 * Jw3 + Jv4' * m4 * Jv4 + Jw4' * Ic4 * Jw4);
M = subs(M, [m1, m2, m3, m4, ...
             I1xx, I1yy, I1zz, ...
             I2xx, I2yy, I2zz, ...
             I3xx, I3yy, I3zz, ...
             I4xx, I4yy, I4zz], ...
            [m1_val, m2_val, m3_val, m4_val, ...
             I1xx_val, I1yy_val, I1zz_val, ...
             I2xx_val, I2yy_val, I2zz_val, ...
             I3xx_val, I3yy_val, I3zz_val, ...
             I4xx_val, I4yy_val, I4zz_val]);
disp('M:');
pretty(M)
%% V
% b_{i,j,k} = ∂m_{ij}/∂q_k
b = sym(zeros(n, n, n));
for i = 1:n
    for j = 1:n
        for k = 1:n
            b(i, j, k) = diff(M(i, j), ['q' num2str(k)]);
        end
    end
end
% c_{ijk} = 1/2 * (b_{i,j,k} + b_{i,k,j} - b_{j,k,i})
c = sym(zeros(n, n, n));
for i = 1:n
    for j = 1:n
        for k = 1:n
            c(i, j, k) = 0.5 * (b(i, j, k) + b(i, k, j) - b(j, k, i));
        end
    end
end
% 系数矩阵格式
C_squared = sym(zeros(n,n));
for i = 1:n
    for j = 1:n
        C_squared(i,j) = c(i,j,j);
    end
end
C_cross = sym(zeros(n, n*(n-1)/2));
col = 1;
for j = 1:n-1
    for k = j+1:n
        for i = 1:n
            C_cross(i,col) = c(i,j,k);
        end
        col = col + 1;
    end
end
disp('V:');
pretty(C_squared)
pretty(C_cross)
%% G
gravity = [0; 0; g];
G = (-m1_val * Jv1' - m2_val * Jv2'- m3_val * Jv3' - m4_val * Jv4') * gravity;
disp('G:');
pretty(G)