% 训练验证集导入
Train_Validation = readtable('train.csv'); 
num_t_v = height(Train_Validation); 
X = Train_Validation{:,3:32};
Y = Train_Validation{:,33};

% 验证
tic()
n = 10000;
XTrain = X(1:n,:);
YTrain = Y(1:n);
result = KNB_fun(XTrain,YTrain,X);
disp(length(result(result == 1)))
% 计算真正例（TP）、假正例（FP）、真反例（TN）和假反例（FN，没用，不算）
TP = sum(Y .* result);  
FP = sum((1 - Y) .* result); 
FN = sum(Y .* (1 - result)); 
% 计算精确率和召回率，进而计算F1  
precision = TP / (TP + FP + eps); 
recall = TP / (TP + FN + eps); 
f1 = 2 * (precision * recall) / (precision + recall + eps); 
disp(f1)
toc()

% 测试
Predict = readtable('pred.csv'); 
XTest = Predict{:,3:32};
YTest = KNB_fun(XTrain,YTrain,XTest);
disp(length(YTest(YTest == 1)))

% K近邻函数
function result=KNB_fun(XTrain,YTrain,X)
n = height(XTrain); 
m = height(X); 
YTry = zeros(m,1);                  % 预测集
dis = zeros(n,1);                   % 距离集
k = 1;                              % 近邻数目
k_dis = zeros(k,1);                 % 近邻类别集
for i=1:m
    for j=1:n                       % 计算欧氏距离
        dis(j) = norm(X(i,:)-XTrain(j,:));
    end
    [~, index] = sort(dis);         % 升序排序，index记录下标
    for j=1:k                       % k个最近邻数据点的类别标签
        k_dis(j) = YTrain(index(j));
    end
    YTry(i) = mode(k_dis);          % 标签众数即为所求
end
result = YTry;
end