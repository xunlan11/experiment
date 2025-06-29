% ��������
n = 100;                                                               % ������
center1 = [1,1];                                                       % ��������
center2 = [3,4];       
X = zeros(2*n,2);                                                      % ���ݵ㣨2ά�������ĵ�+��˹����
X(1:n,:) = ones(n,1)*center1 + randn(n,2);           
X(n+1:2*n,:) = ones(n,1)*center2 + randn(n,2);
Y = zeros(2*n,1);                                                      % ����ǩ����һ��Ϊ1���ڶ���Ϊ-1�� 
Y(1:n) = 1; 
Y(n+1:2*n) = -1;        

%{
% ͼһ���������ݵ�
figure(1)
set(gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);          
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);  
hold on;
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2');
%}

% ��֪��ģ�ͣ�y=x*w+b
tic()
w = zeros(2,1);
b = zeros(1);             
x = 1000; % ��������
c = 0.1; % ����

for i = 1:x
    a = randi([1,2*n]);
    random = X(a,:); 
    judge = (random*w+b)*Y(a);
    if judge <= 0
        w = w+(X(a,:)').*c.*Y(a);
        b = b+c*Y(a);
    end
end
toc()

% ͼ��������������ͼ��x1Ϊ���ᣬy1Ϊ���ᣩ
x1 = -2:0.00001:7;
y1 = (-b*ones(1,length(x1))-w(1)*x1)/w(2);       
%{                                                        
figure(2)
set(gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);          
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);   
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                     
xlabel('x axis');
ylabel('y axis');
legend('class 1','class 2','classification surface');
%}

% ����
m = 10;              
Xt = zeros(2*m,2);  
Xt(1:m,:) = ones(m,1)*center1 + randn(m,2);
Xt(m+1:2*m,:) = ones(m,1)*center2 + randn(m,2);  
Yt = zeros(2*m,1);        
Yt(1:m) = 1; 
Yt(m+1:2*m) = -1;    

% ͼ�������Խ��
figure(3)
set(gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',18)
plot(X(1:n,1),X(1:n,2),'ro','LineWidth',1,'MarkerSize',10);           
hold on;
plot(X(n+1:2*n,1),X(n+1:2*n,2),'b*','LineWidth',1,'MarkerSize',10);  
hold on;
plot(Xt(1:m,1),Xt(1:m,2),'go','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(m+1:2*m,1),Xt(m+1:2*m,2),'g*','LineWidth',1,'MarkerSize',10);  
hold on;
plot( x1,y1,'k','LineWidth',1,'MarkerSize',10);                     
xlabel('x axis');
xlabel('x axis');
ylabel('y axis');
legend('class 1: train','class 2: train','class 1: test','class 2: test','classification surface');

% ����������
disp(w)
disp(b)

wrong = 0;
for i = 1:2*n
     if (X(i,:)*w+b)*Y(i) < 0
         wrong = wrong + 1;
     end
end
wrong = wrong / (2*n);
test_wrong = 0;
for i = 1:2*m
     if (Xt(i,:)*w+b)*Yt(i) < 0
         test_wrong = test_wrong + 1;
     end
end
test_wrong = test_wrong / (2*m);
disp(wrong)
disp(test_wrong)