% 数据生成
n = 2000;                % 样本量
X = rand(n,2)*10;        % 数据点（2维）：0-10随机
Y = zeros(n,1);          % 类别标签
for i=1:n
   if 0<X(i,1) && X(i,1)<6 && 0<X(i,2) && X(i,2)<6               
       Y(i) = 1;
   end
   if 7<X(i,1) && X(i,1)<10 && 0<X(i,2) && X(i,2)<3
       Y(i) = 1;
   end
   if 7<X(i,1) && X(i,1)<10 && 3<X(i,2) && X(i,2)<6
       Y(i) = 1;
   end
   if 0<X(i,1) && X(i,1)<3 && 7<X(i,2) && X(i,2)<10
       Y(i) = 1;
   end
   if 3<X(i,1) && X(i,1)<6 && 7<X(i,2) && X(i,2)<10
       Y(i) = 1;
   end
   if 7<X(i,1) && X(i,1)<10 && 7<X(i,2) && X(i,2)<10
       Y(i) = 1;
   end
end
X = X(Y>0,:);              % 去掉类别间隔中的点，其Y=0
Y = Y(Y>0,:);                                                  
n = length(Y);                                                  

%{
% 图一：数据点
figure(1)
set(gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(Y==1,1),X(Y==1,2),'ro','LineWidth',1,'MarkerSize',10);            
hold on;
plot(X(Y==2,1),X(Y==2,2),'ko','LineWidth',1,'MarkerSize',10);           
hold on;
plot(X(Y==3,1),X(Y==3,2),'bo','LineWidth',1,'MarkerSize',10);           
hold on;
plot(X(Y==4,1),X(Y==4,2),'g*','LineWidth',1,'MarkerSize',10);            
hold on;
plot(X(Y==5,1),X(Y==5,2),'m*','LineWidth',1,'MarkerSize',10);           
hold on;
plot(X(Y==6,1),X(Y==6,2),'c*','LineWidth',1,'MarkerSize',10);            
hold on;
xlabel('x axis');
ylabel('y axis');
clear Y;     
%}

% K-means
tic()
K = 6;                                 % 中心点
Ym = zeros(n,1);                       % 预测的数据标签 
meanpoint = rand(K,2)*10;              % 随机初始化中心点  
meanpoint_change = Inf;                % 中心点变化
threshold = 0.01;                      % 阈值
while(meanpoint_change>threshold)      % 迭代直到中心点变化小于阈值
    [Ym, meanpoint, meanpoint_change] = iteration(X, meanpoint);
end
disp(meanpoint_change)
toc()

% 图二：聚类结果及中心点
figure(2)
set(gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(Ym==1,1),X(Ym==1,2),'ro','LineWidth',1,'MarkerSize',10);          
hold on;
plot(X(Ym==2,1),X(Ym==2,2),'ko','LineWidth',1,'MarkerSize',10);        
hold on;
plot(X(Ym==3,1),X(Ym==3,2),'bo','LineWidth',1,'MarkerSize',10);         
hold on;
plot(X(Ym==4,1),X(Ym==4,2),'g*','LineWidth',1,'MarkerSize',10);        
hold on;
plot(X(Ym==5,1),X(Ym==5,2),'m*','LineWidth',1,'MarkerSize',10);           
hold on;
plot(X(Ym==6,1),X(Ym==6,2),'c*','LineWidth',1,'MarkerSize',10);          
hold on;
plot(meanpoint(:,1),meanpoint(:,2),'ms','MarkerFaceColor','m','LineWidth',1,'MarkerSize',10);   
hold on;
xlabel('x axis');
ylabel('y axis');

% 迭代函数，输入数据集和初始中心点，输出数据标签预测结果、新中心点和中心点变化距离
function [y_result, meanpoint_result, meanpoint_change] = iteration(x, meanpoint)  
[n, ~] = size(x);             % 数据点个数
y = zeros(n,1);               % 数据标签 
[row, ~] = size(meanpoint);   % 中心点个数
new_meanpoint = zeros(row,2); % 新中心点
for i = 1:n                   % 离数据点最近的中心点的数据标签是数据点的数据标签
    distances = zeros(row,1);
    for j = 1:row
        distances(j) = norm(x(i,:) - meanpoint(j,:));  
    end
    [~, idx] = min(distances);  
    y(i) = idx;
end
change = zeros(row,1);        % 中心点变化距离
for label = 1:row  
    idx = (y == label);  
    x_label = x(idx, :);  
    new_meanpoint(label,:) = mean(x_label,1);  
    change(label) = norm(new_meanpoint(label,:)-meanpoint(label,:));
end 
y_result = y;
meanpoint_result = new_meanpoint;
meanpoint_change = sum(change);
end