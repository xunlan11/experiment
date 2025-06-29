% ������������
n = 100;                % ������
center1 = [1,1];        % �������ģ��ڶ��ࣺ�ɷ�[6,6]�����ɷ�[3,3]��
center2 = [3,3];        
X = zeros(2*n,2);       % ���ݵ㣨2ά������˹����
X(1:n,:) = ones(n,1)*center1 + randn(n,2);
X(n+1:2*n,:) = ones(n,1)*center2 + randn(n,2);    
Y = zeros(2*n,1);       % ����ǩ
Y(1:n) = 1; 
Y(n+1:2*n) = -1;  
X_hat = X .* Y;

%{
% ͼһ�����ݵ�
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

% SVMģ�� 
tic()
alpha = zeros(2*n,1);       % ��ż���������
lambda = 0.1;               % �������ղ�����
beta = 0.1;                 % �ͷ�������
C = 0.3;                    % �ͷ�����C
eta = 0.00005;              % ������
m = 100000;                 % ��������m             
for i=1:m                   % �����Ž�
    alpha = alpha - eta*(X_hat*X_hat'*alpha-1+lambda*Y+beta*Y'*alpha*Y);
    alpha(alpha>C) = C;
    alpha(alpha<0) = 0;
    lambda = lambda + beta*(Y'*alpha);
end
idx = find(alpha<C&alpha>0);    % ����������
len = length(idx);
j = idx(randi(len)); 
w = X_hat' * alpha;
b = Y(j)-sum(Y.*alpha.*X*X(j,:)');
b_all = zeros(len,1);           % �۲ⲻͬb�Ƿ�����
for i=1:len
    b_all(i) = Y(i)-sum(Y.*alpha.*X*X(i,:)');
end
L1 = 1/2*alpha'*(X_hat*X_hat')*alpha-sum(alpha)+lambda*Y'*alpha+beta/2*(Y'*alpha)^2; % ���������������պ���
z = 1-Y.*(X*w+b);
z(z<0) = 0;
L2 = sum(z)+1/2/C*(w'*w);                                                            % ��ҳ��ʧ����
disp(L1)
disp(L2)
toc()

% ͼ��������������ͼ��x1Ϊ���ᣬyΪ���ᣬ1Ϊ������棬2��3Ϊ����߽磩
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
plot(X(alpha>0,1),X(alpha>0,2),'rs','LineWidth',1,'MarkerSize',10);    % ֧������
hold on;
plot(X(alpha<C&alpha>0,1),X(alpha<C&alpha>0,2),'rs','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);    % ����߽��ϵ�֧������
hold on;
xlabel('x axis');
ylabel('y axis');
set(gca,'Fontsize',10)
legend('class 1','class 2','classification surface','boundary','boundary','support vectors','support vectors on boundary');