% 数据样本、噪声点生成
n = 2000;                % 样本量
X = rand(n,2)*10;        % 数据点（2维）：0-10随机
Y = zeros(n,1);          % 类别标签
for i=1:n
   if 0<X(i,1) && X(i,1)<3 && 0<X(i,2) && X(i,2)<3      
       Y(i) = 1;
   end
   if 0<X(i,1) && X(i,1)<3 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 2;
   end
   if 0<X(i,1) && X(i,1)<3 && 7<X(i,2) && X(i,2)<10
       Y(i) = 3;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 0<X(i,2) && X(i,2)<3
       Y(i) = 4;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 5;
   end
   if 3.5<X(i,1) && X(i,1)<6.5 && 7<X(i,2) && X(i,2)<10
       Y(i) = 6;
   end
   if 7<X(i,1) && X(i,1)<10 && 0<X(i,2) && X(i,2)<3
       Y(i) = 7;
   end
   if 7<X(i,1) && X(i,1)<10 && 3.5<X(i,2) && X(i,2)<6.5
       Y(i) = 8;
   end
   if 7<X(i,1) && X(i,1)<10 && 7<X(i,2) && X(i,2)<10
       Y(i) = 9;
   end
end
X = X(Y>0,:);                                  % 去掉类别间隔中的点，其Y=0                                    
Y = Y(Y>0,:);                                                  
nn = length(Y);
X(nn+1:n,:) = rand(n-nn,2)*10;                 % （n-nn）个噪声点，标签随机选取
Y(nn+1:n) = ceil( rand(n-nn,1)*9 );   

%{
% 图一：数据点与噪声点
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
plot(X(Y==7,1),X(Y==7,2),'b+','LineWidth',1,'MarkerSize',10);          
hold on;
plot(X(Y==8,1),X(Y==8,2),'r+','LineWidth',1,'MarkerSize',10);           
hold on;
plot(X(Y==9,1),X(Y==9,2),'k+','LineWidth',1,'MarkerSize',10);           
hold on;
xlabel('x axis');
ylabel('y axis');
%}

% 测试样本生成
m = 100;                % 测试样本量
Xt = rand(m,2)*10;       
Yt = zeros(m,1);
for i=1:m
   if 0<Xt(i,1) && Xt(i,1)<3 && 0<Xt(i,2) && Xt(i,2)<3
       Yt(i) = 1;
   end
   if 0<Xt(i,1) && Xt(i,1)<3 && 3.5<Xt(i,2) && Xt(i,2)<6.5
       Yt(i) = 2;
   end
   if 0<Xt(i,1) && Xt(i,1)<3 && 7<Xt(i,2) && Xt(i,2)<10
       Yt(i) = 3;
   end
   if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 0<Xt(i,2) && Xt(i,2)<3
       Yt(i) = 4;
   end
   if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 3.5<Xt(i,2) && Xt(i,2)<6.5
       Yt(i) = 5;
   end
   if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 7<Xt(i,2) && Xt(i,2)<10
       Yt(i) = 6;
   end
   if 7<Xt(i,1) && Xt(i,1)<10 && 0<Xt(i,2) && Xt(i,2)<3
       Yt(i) = 7;
   end
   if 7<Xt(i,1) && Xt(i,1)<10 && 3.5<Xt(i,2) && Xt(i,2)<6.5
       Yt(i) = 8;
   end
   if 7<Xt(i,1) && Xt(i,1)<10 && 7<Xt(i,2) && Xt(i,2)<10
       Yt(i) = 9;
   end
end
Xt = Xt(Yt>0,:);
Yt = Yt(Yt>0,:);
m = length(Yt);

%{
% 图二：数据点与测试点
figure(2)
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
plot(X(Y==5,1),X(Y==5,2),'b*','LineWidth',1,'MarkerSize',10);          
hold on;
plot(X(Y==6,1),X(Y==6,2),'c*','LineWidth',1,'MarkerSize',10);         
hold on;
plot(X(Y==7,1),X(Y==7,2),'b+','LineWidth',1,'MarkerSize',10);           
hold on;
plot(X(Y==8,1),X(Y==8,2),'r+','LineWidth',1,'MarkerSize',10);          
hold on;
plot(X(Y==9,1),X(Y==9,2),'k+','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(:,1),Xt(:,2),'ms','MarkerFaceColor','m','LineWidth',1,'MarkerSize',10);         
hold on;
xlabel('x axis');
ylabel('y axis');
%}

% 贝叶斯算法
tic()
Ym = zeros(m,1);  
Py = zeros(9,1);             % 先验概率（类别标签）
PX_Y = zeros(3,2,9);         % 条件概率（X区间，维度，类别标签）
for i = 1:n
    Py(Y(i),1) = Py(Y(i),1) + 1;
    if 0<X(i,1) && X(i,1)<3 && 0<X(i,2) && X(i,2)<3      
       PX_Y(1,1,Y(i)) = PX_Y(1,1,Y(i))+1;
       PX_Y(1,2,Y(i)) = PX_Y(1,2,Y(i))+1;
    end
    if 0<X(i,1) && X(i,1)<3 && 3.5<X(i,2) && X(i,2)<6.5
       PX_Y(1,1,Y(i)) = PX_Y(1,1,Y(i))+1;
       PX_Y(2,2,Y(i)) = PX_Y(2,2,Y(i))+1;
    end
    if 0<X(i,1) && X(i,1)<3 && 7<X(i,2) && X(i,2)<10
       PX_Y(1,1,Y(i)) = PX_Y(1,1,Y(i))+1;
       PX_Y(3,2,Y(i)) = PX_Y(3,2,Y(i))+1;
    end
    if 3.5<X(i,1) && X(i,1)<6.5 && 0<X(i,2) && X(i,2)<3
       PX_Y(2,1,Y(i)) = PX_Y(2,1,Y(i))+1;
       PX_Y(1,2,Y(i)) = PX_Y(1,2,Y(i))+1;
    end
    if 3.5<X(i,1) && X(i,1)<6.5 && 3.5<X(i,2) && X(i,2)<6.5
       PX_Y(2,1,Y(i)) = PX_Y(2,1,Y(i))+1;
       PX_Y(2,2,Y(i)) = PX_Y(2,2,Y(i))+1;
    end
    if 3.5<X(i,1) && X(i,1)<6.5 && 7<X(i,2) && X(i,2)<10
       PX_Y(2,1,Y(i)) = PX_Y(2,1,Y(i))+1;
       PX_Y(3,2,Y(i)) = PX_Y(3,2,Y(i))+1;
    end
    if 7<X(i,1) && X(i,1)<10 && 0<X(i,2) && X(i,2)<3
       PX_Y(3,1,Y(i)) = PX_Y(3,1,Y(i))+1;
       PX_Y(1,2,Y(i)) = PX_Y(1,2,Y(i))+1;
    end
    if 7<X(i,1) && X(i,1)<10 && 3.5<X(i,2) && X(i,2)<6.5
       PX_Y(3,1,Y(i)) = PX_Y(3,1,Y(i))+1;
       PX_Y(2,2,Y(i)) = PX_Y(2,2,Y(i))+1;
    end
    if 7<X(i,1) && X(i,1)<10 && 7<X(i,2) && X(i,2)<10
       PX_Y(3,1,Y(i)) = PX_Y(3,1,Y(i))+1;
       PX_Y(3,2,Y(i)) = PX_Y(3,2,Y(i))+1;
    end
end
Py = Py./n;
PX_Y = PX_Y./n;
Px = zeros(m,9);               % 联合概率（测试点序数，类别标签）
for i = 1:m
    for j = 1:9 
        if 0<Xt(i,1) && Xt(i,1)<3 && 0<Xt(i,2) && Xt(i,2)<3
            Px(i,j) = Py(j,1).*PX_Y(1,1,j).*PX_Y(1,2,j);
        end
        if 0<Xt(i,1) && Xt(i,1)<3 && 3.5<Xt(i,2) && Xt(i,2)<6.5
            Px(i,j) = Py(j,1).*PX_Y(1,1,j).*PX_Y(2,2,j);
        end
        if 0<Xt(i,1) && Xt(i,1)<3 && 7<Xt(i,2) && Xt(i,2)<10
            Px(i,j) = Py(j,1).*PX_Y(1,1,j).*PX_Y(3,2,j);
        end
        if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 0<Xt(i,2) && Xt(i,2)<3
            Px(i,j) = Py(j,1).*PX_Y(2,1,j).*PX_Y(1,2,j);
        end
        if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 3.5<Xt(i,2) && Xt(i,2)<6.5
            Px(i,j) = Py(j,1).*PX_Y(2,1,j).*PX_Y(2,2,j);
        end
        if 3.5<Xt(i,1) && Xt(i,1)<6.5 && 7<Xt(i,2) && Xt(i,2)<10
            Px(i,j) = Py(j,1).*PX_Y(2,1,j).*PX_Y(3,2,j);
        end
        if 7<Xt(i,1) && Xt(i,1)<10 && 0<Xt(i,2) && Xt(i,2)<3
            Px(i,j) = Py(j,1).*PX_Y(3,1,j).*PX_Y(1,2,j);
        end
        if 7<Xt(i,1) && Xt(i,1)<10 && 3.5<Xt(i,2) && Xt(i,2)<6.5
            Px(i,j) = Py(j,1).*PX_Y(3,1,j).*PX_Y(2,2,j);
        end
        if 7<Xt(i,1) && Xt(i,1)<10 && 7<Xt(i,2) && Xt(i,2)<10
            Px(i,j) = Py(j,1).*PX_Y(3,1,j).*PX_Y(3,2,j);
        end
    end
    value = 0;
    index = 0;
    for j = 1:9
        if Px(i,j)>value
            value = Px(i,j);
            index = j;
        end
    end
    Ym(i,1) = index;
end
toc()

% 图三：预测结果
figure(3)
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
plot(X(Y==5,1),X(Y==5,2),'b*','LineWidth',1,'MarkerSize',10);          
hold on;
plot(X(Y==6,1),X(Y==6,2),'c*','LineWidth',1,'MarkerSize',10);            
hold on;
plot(X(Y==7,1),X(Y==7,2),'b+','LineWidth',1,'MarkerSize',10);         
hold on;
plot(X(Y==8,1),X(Y==8,2),'r+','LineWidth',1,'MarkerSize',10);          
hold on;
plot(X(Y==9,1),X(Y==9,2),'k+','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(Ym==1,1),Xt(Ym==1,2),'ro','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(Ym==2,1),Xt(Ym==2,2),'ko','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(Ym==3,1),Xt(Ym==3,2),'bo','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);         
hold on;
plot(Xt(Ym==4,1),Xt(Ym==4,2),'go','MarkerFaceColor','g','LineWidth',1,'MarkerSize',10);           
hold on;
plot(Xt(Ym==5,1),Xt(Ym==5,2),'bo','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(Ym==6,1),Xt(Ym==6,2),'co','MarkerFaceColor','c','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(Ym==7,1),Xt(Ym==7,2),'bo','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);         
hold on;
plot(Xt(Ym==8,1),Xt(Ym==8,2),'ro','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(Ym==9,1),Xt(Ym==9,2),'ko','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);           
hold on;
xlabel('x axis');
ylabel('y axis');

% 结果与错误率
count = 0;
for i=1:m
    if Ym(i)~=Yt(i)
        count = count + 1;
    end
end
accuracy = count / m;
disp(accuracy)