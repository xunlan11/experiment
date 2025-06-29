% ��������
n = 2000;                % ������
X = rand(n,2)*10;        % ���ݵ㣨2ά����0-10���
Y = zeros(n,1);          % ����ǩ
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
X = X(Y>0,:);              % ȥ��������еĵ㣬��Y=0
Y = Y(Y>0,:);                                                  
n = length(Y);                                                  

%{
% ͼһ�����ݵ�
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
K = 6;                                 % ���ĵ�
Ym = zeros(n,1);                       % Ԥ������ݱ�ǩ 
meanpoint = rand(K,2)*10;              % �����ʼ�����ĵ�  
meanpoint_change = Inf;                % ���ĵ�仯
threshold = 0.01;                      % ��ֵ
while(meanpoint_change>threshold)      % ����ֱ�����ĵ�仯С����ֵ
    [Ym, meanpoint, meanpoint_change] = iteration(X, meanpoint);
end
disp(meanpoint_change)
toc()

% ͼ���������������ĵ�
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

% �����������������ݼ��ͳ�ʼ���ĵ㣬������ݱ�ǩԤ�����������ĵ�����ĵ�仯����
function [y_result, meanpoint_result, meanpoint_change] = iteration(x, meanpoint)  
[n, ~] = size(x);             % ���ݵ����
y = zeros(n,1);               % ���ݱ�ǩ 
[row, ~] = size(meanpoint);   % ���ĵ����
new_meanpoint = zeros(row,2); % �����ĵ�
for i = 1:n                   % �����ݵ���������ĵ�����ݱ�ǩ�����ݵ�����ݱ�ǩ
    distances = zeros(row,1);
    for j = 1:row
        distances(j) = norm(x(i,:) - meanpoint(j,:));  
    end
    [~, idx] = min(distances);  
    y(i) = idx;
end
change = zeros(row,1);        % ���ĵ�仯����
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