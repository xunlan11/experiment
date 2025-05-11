% 训练验证集导入
Train_Validation = readtable('train.csv'); 
num_t_v = height(Train_Validation); 
X = Train_Validation{:,3:32};
Y = Train_Validation{:,33};

% 得到x个最重要的维度
tic()
% 小训练集获得w、b，w与对应x的均值之积表示权重
n = 1000;     % 小训练集数据集数目
XTrain = X(1:n,:);
YTrain = Y(1:n);
[w1, b1] = perception_fun(XTrain,YTrain,30);
[threshold1, f11] = calculate_metrics(X,Y,w1,b1);
disp(threshold1)
disp(f11)
X_average = sum(X(:,1:30))./num_t_v;
X_weight = X_average .* w1';
[~, indices] = sort(X_weight, 'descend');  
%{
% 随机森林
rfModel = TreeBagger(100, X, Y, 'Method', 'classification', 'OOBPredictorImportance', 'on');
[~, indices] = sort(rfModel.OOBPermutedVarDeltaError, 'descend');  
%}
x = 5;
indices = indices(1:x); 
toc()

% 全集训练
X = X(:, indices);
[w2, b2] = perception_fun(X,Y,x);
[threshold2, f12] = calculate_metrics(X,Y,w2,b2);
disp(threshold2)
disp(f12)

% 测试
Predict = readtable('pred.csv'); 
num_p = height(Predict); 
XTest = Predict{:,3:32};
XTest = XTest(:, indices);
YTest = XTest * w2 + b2;
YTest(YTest>threshold2) = 1;
YTest(YTest<1) = 0;
disp(length(find(YTest==1)))

% 感知机函数
function [w_result,b_result]=perception_fun(X,Y,w_num)
w = zeros(w_num,1);
b = 0;          
x = 100000;        % 迭代次数
c = 0.0001;         % 步长
for i = 1:x
    a = randi([1,height(Y)]);
    if (X(a,:)*w+b)*Y(a) <= 0
        w = w+(X(a,:)').*c.*Y(a);
        b = b+c*Y(a);
    end
end
w_result=w;
b_result=b;
end

% 最佳F1值
function [best_threshold, best_f1] = calculate_metrics(X,Y,w,b)
best_threshold = 0;
best_f1 = 0;
for threshold = 0:0.0001:0.1
    result = X * w + b;
    result(result>threshold) = 1;
    result(result<1) = 0;
    % 计算真正例（TP）、假正例（FP）、真反例（TN）和假反例（FN，没用，不算）
    TP = sum(Y .* result);  
    FP = sum((1 - Y) .* result); 
    FN = sum(Y .* (1 - result)); 
    % 计算精确率和召回率，进而计算F1  
    precision = TP / (TP + FP + eps); 
    recall = TP / (TP + FN + eps); 
    f1 = 2 * (precision * recall) / (precision + recall + eps); 
    if f1 > best_f1
        best_f1 = f1;
        best_threshold = threshold;
    end
end
end  