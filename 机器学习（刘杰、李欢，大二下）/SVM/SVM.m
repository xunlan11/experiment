% 数据样本生成
n = 100;                % 样本量
center1 = [1,1];        % 数据中心（第二类：可分[6,6]，不可分[3,3]）
center2 = [3,3];        
X = zeros(2*n,2);       % 数据点（2维）：高斯噪声
X(1:n,:) = ones(n,1)*center1 + randn(n,2);
X(n+1:2*n,:) = ones(n,1)*center2 + randn(n,2);    
Y = zeros(2*n,1);       % 类别标签
Y(1:n) = 1; 
Y(n+1:2*n) = -1;  
X_hat = X .* Y;

%{
% 图一：数据点
figure(1)
set(gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'go','LineWidth',1,'MarkerSize',10);         
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);  
hold on;
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2');
%}

% SVM模型 
tic()
alpha = zeros(2*n,1);       % 对偶问题变量α
lambda = 0.1;               % 拉格朗日参数λ
beta = 0.1;                 % 惩罚参数β
C = 0.3;                    % 惩罚参数C
eta = 0.00005;              % 步长η
m = 100000;                 % 迭代次数m             
for i=1:m                   % 求最优解
    alpha = alpha - eta*(X_hat*X_hat'*alpha-1+lambda*Y+beta*Y'*alpha*Y);
    alpha(alpha>C) = C;
    alpha(alpha<0) = 0;
    lambda = lambda + beta*(Y'*alpha);
end
idx = find(alpha<C&alpha>0);    % 分类界面参数
len = length(idx);
j = idx(randi(len)); 
w = X_hat' * alpha;
b = Y(j)-sum(Y.*alpha.*X*X(j,:)');
b_all = zeros(len,1);           % 观测不同b是否收敛
for i=1:len
    b_all(i) = Y(i)-sum(Y.*alpha.*X*X(i,:)');
end
L1 = 1/2*alpha'*(X_hat*X_hat')*alpha-sum(alpha)+lambda*Y'*alpha+beta/2*(Y'*alpha)^2; % 线性增广拉格朗日函数
z = 1-Y.*(X*w+b);
z(z<0) = 0;
L2 = sum(z)+1/2/C*(w'*w);                                                            % 合页损失函数
disp(L1)
disp(L2)
toc()

% 图二：分类器可视图（x1为横轴，y为纵轴，1为分类界面，2、3为间隔边界）
x1 = -2 : 0.00001 : 7;
y1 = ( -b * ones(1,length(x1)) - w(1) * x1 )/w(2);         
y2 = ( ones(1,length(x1)) - b * ones(1,length(x1)) - w(1) * x1 )/w(2);
y3 = ( -ones(1,length(x1)) - b * ones(1,length(x1)) - w(1) * x1 )/w(2);
figure(2)
set(gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'go','LineWidth',1,'MarkerSize',10);            
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);    
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                      
hold on;
plot( x1,y2,'k-.','LineWidth',1,'MarkerSize',10);                     
hold on;
plot( x1,y3,'k-.','LineWidth',1,'MarkerSize',10);                     
hold on;
plot(X(alpha>0,1),X(alpha>0,2),'rs','LineWidth',1,'MarkerSize',10);    % 支持向量
hold on;
plot(X(alpha<C&alpha>0,1),X(alpha<C&alpha>0,2),'rs','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);    % 间隔边界上的支持向量
hold on;
xlabel('x axis');
ylabel('y axis');
set(gca,'Fontsize',10)
legend('class 1','class 2','classification surface','boundary','boundary','support vectors','support vectors on boundary');