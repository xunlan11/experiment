% 数据样本、噪声点生成
n = 2000;                
X1 = rand(n,2)*10;       
Y1 = zeros(n,1);        
for i=1:n
   if 0<X1(i,1) && X1(i,1)<3 && 0<X1(i,2) && X1(i,2)<3      
       Y1(i) = 1;
   end
   if 0<X1(i,1) && X1(i,1)<3 && 3.5<X1(i,2) && X1(i,2)<6.5
       Y1(i) = 2;
   end
   if 0<X1(i,1) && X1(i,1)<3 && 7<X1(i,2) && X1(i,2)<10
       Y1(i) = 3;
   end
   if 3.5<X1(i,1) && X1(i,1)<6.5 && 0<X1(i,2) && X1(i,2)<3
       Y1(i) = 4;
   end
   if 3.5<X1(i,1) && X1(i,1)<6.5 && 3.5<X1(i,2) && X1(i,2)<6.5
       Y1(i) = 5;
   end
   if 3.5<X1(i,1) && X1(i,1)<6.5 && 7<X1(i,2) && X1(i,2)<10
       Y1(i) = 6;
   end
   if 7<X1(i,1) && X1(i,1)<10 && 0<X1(i,2) && X1(i,2)<3
       Y1(i) = 7;
   end
   if 7<X1(i,1) && X1(i,1)<10 && 3.5<X1(i,2) && X1(i,2)<6.5
       Y1(i) = 8;
   end
   if 7<X1(i,1) && X1(i,1)<10 && 7<X1(i,2) && X1(i,2)<10
       Y1(i) = 9;
   end
end
X1 = X1(Y1>0,:);                                                          
Y1 = Y1(Y1>0,:);              
percent = 1;
nn = round(length(Y1) * percent);
X2 = rand(nn,2)*10;       
Y2 = ceil( rand(nn,1)*9 );   
X = [X1;X2];
Y = [Y1;Y2];
l = length(Y);

% 测试样本生成
m = 100;               
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

% 贝叶斯算法
tic()
Ym1 = zeros(m,1);  
Py = zeros(9,1);             % 先验概率（类别标签）
PX_Y = zeros(3,2,9);         % 条件概率（X区间，维度，类别标签）
for i = 1:l
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
Py = Py./l;
PX_Y = PX_Y./l;
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
    Ym1(i,1) = index;
end
toc()
count1 = 0;
for i=1:m
    if Ym1(i)~=Yt(i)
        count1 = count1 + 1;
    end
end
accuracy1 = count1 / m;
disp(accuracy1)
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
plot(Xt(Ym1==1,1),Xt(Ym1==1,2),'ro','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(Ym1==2,1),Xt(Ym1==2,2),'ko','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(Ym1==3,1),Xt(Ym1==3,2),'bo','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);         
hold on;
plot(Xt(Ym1==4,1),Xt(Ym1==4,2),'go','MarkerFaceColor','g','LineWidth',1,'MarkerSize',10);           
hold on;
plot(Xt(Ym1==5,1),Xt(Ym1==5,2),'bo','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(Ym1==6,1),Xt(Ym1==6,2),'co','MarkerFaceColor','c','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(Ym1==7,1),Xt(Ym1==7,2),'bo','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);         
hold on;
plot(Xt(Ym1==8,1),Xt(Ym1==8,2),'ro','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(Ym1==9,1),Xt(Ym1==9,2),'ko','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);           
hold on;
xlabel('x axis');
ylabel('y axis');

% K-近邻   （预测输出，并与测试数据的真实输出比较，计算错误率）
tic()
Ym2 = zeros(m,1);                    % 预测集
dis = zeros(l,1);                    % 距离集
k = 5;                               % 近邻数目
k_dis = zeros(k,1);                  % 近邻类别集
for i=1:m
    for j=1:l                        % 计算欧氏距离
        dis(j) = norm(Xt(i,:)-X(j,:));
    end
    [a, index] = sort(dis);          % 升序排序，index记录下标
    for j=1:k                        % k个最近邻数据点的类别标签
        k_dis(j) = Y(index(j));
    end
    Ym2(i) = mode(k_dis);            % 标签众数即为所求
end
toc()
count2 = 0;
for i=1:m
    if Ym2(i)~=Yt(i)
        count2 = count2 + 1;
    end
end
accuracy2 = count2 / m;
disp(accuracy2)
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
plot(Xt(Ym2==1,1),Xt(Ym2==1,2),'ro','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);        
hold on;
plot(Xt(Ym2==2,1),Xt(Ym2==2,2),'ko','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);           
hold on;
plot(Xt(Ym2==3,1),Xt(Ym2==3,2),'bo','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);           
hold on;
plot(Xt(Ym2==4,1),Xt(Ym2==4,2),'go','MarkerFaceColor','g','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(Ym2==5,1),Xt(Ym2==5,2),'bo','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);       
hold on;
plot(Xt(Ym2==6,1),Xt(Ym2==6,2),'co','MarkerFaceColor','c','LineWidth',1,'MarkerSize',10);            
hold on;
plot(Xt(Ym2==7,1),Xt(Ym2==7,2),'bo','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);        
hold on;
plot(Xt(Ym2==8,1),Xt(Ym2==8,2),'ro','MarkerFaceColor','r','LineWidth',1,'MarkerSize',10);          
hold on;
plot(Xt(Ym2==9,1),Xt(Ym2==9,2),'ko','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);      
hold on;
xlabel('x axis');
ylabel('y axis');