%% 输入已知量
cost_parts = [2, 8 ,12, 2, 8, 12, 8, 12];                   %八种零配件的采购成本
cost_check_parts = [1, 1, 2, 1, 1, 2, 1, 2];                %八种零配件的检测成本
defective_parts = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]; %八种零配件的次品率
cost_assembly_half_products=[8, 8, 8];                      %三个半成品的装配成本
cost_check_half_products=[4,4,4];                           %三个半成品的检测成本
cost_disassemble_half_products = [6, 6, 6];                 %三个半成品的拆解费用
defective_half_products = [0.1, 0.1, 0.1];                  %三个半成品的次品率
cost_assembly_product = 8;                                  %成品的装配成本
cost_check_product = 6;                                     %成品的检测成本
cost_disassemble_product = 10;                              %成品的拆解费用
defective_product = 0.1;                                    %成品的次品率
price_product = 200;                                        %成品售价
cost_replace = 40;                                          %不合格成品调换损失

%% 初始化变量
num_parts = 2000; %各零配件个数
num_bad_parts = round(num_parts.*defective_parts); 
num_good_parts=num_parts-num_bad_parts; 
schemes = {}; %方案
profits = []; %方案的利润

%% 遍历所有可能的生产决策组合
for check_parts_bin = 0:255  %零配件检测
    check_parts = num2cell(dec2bin(check_parts_bin, 8) - '0');
    for check_half_products_bin = 0:7 %半成品检测
        check_half_products = num2cell(dec2bin(check_half_products_bin, 3) - '0');
        for disassemble_half_products_bin = 0:7 %半成品拆解
            disassemble_half_products = num2cell(dec2bin(disassemble_half_products_bin, 3) - '0'); 
            for check_product = [0, 1] %成品检测
                for disassemble_product = [0, 1] %成品拆解
                    parts = cell(1, 8); %8种零配件
                    for k = 1:8
                        %每种零配件的良品和次品
                        parts{k} = [ones(1, num_good_parts(k)), zeros(1, num_bad_parts(k))];
                        %随机打乱顺序
                        parts{k} = parts{k}(randperm(num_parts)); 
                    end
                    uses_parts = zeros(1, 8); %每个零配件的使用次数
                    total_sales = 0; %总销售数量
                    cost = 0;        %总成本
                    i = ones(1, 8);  %各个零配件的索引
                    %模拟生产过程
                    num_products_to_assemble = floor(num_parts / 2); %最大成品数量
                    for product_num = 1:num_products_to_assemble
                        %检查是否还有足够的零配件继续生产
                        if any(i > num_parts)
                            break; 
                        end
                        %获取当前零配件
                        parts_current = cellfun(@(x, idx) x(idx), parts, num2cell(i), 'UniformOutput', false);
                        %增加当前零配件的使用次数
                        for k = 1:8
                            uses_parts(k) = uses_parts(k) + 1;
                        end
                        discard = false;
                        for k = 1:8
                            %检查零配件
                            if check_parts{k} == 1 && parts_current{k} == 0
                                discard = true;
                                i(k) = i(k) + 1; %丢弃并跳过该零配件
                                break;
                            end
                        end
                        if discard
                            continue; %跳过本轮生产
                        end
                        %模拟装配和检测半成品
                        half_products = [all([parts_current{1:3}]), all([parts_current{4:6}]), all([parts_current{5:6}])];
                        %检测半成品
                        for k = 1:3
                            if check_half_products{k}
                                if rand() <= defective_half_products(k)
                                    if disassemble_half_products{k}
                                        cost = cost + cost_disassemble_half_products(k); %增加拆解费用
                                    end
                                    half_products(k) = false; %标记为次品
                                end
                            end
                        end
                        %装配成品
                        if all(half_products)
                            product = rand() > defective_product;     
                            cost = cost + cost_assembly_product;  %增加装配成品的成本
                            %检测成品
                            if check_product && ~product
                                cost = cost + cost_check_product; %增加检测成品的成本
                                if disassemble_product
                                    cost = cost + cost_disassemble_product; %增加拆解成品的费用
                                else
                                    cost = cost + cost_replace; %不拆解则增加替换成品的损失
                                end
                                continue; %跳过本轮生产
                            end
                            %记录销售数据
                            if product
                                total_sales = total_sales + 1; %成品合格，增加销售数量
                            else
                                cost = cost + cost_replace; %次品的替换损失
                                if disassemble_product
                                    cost = cost + cost_disassemble_product; %增加拆解费用
                                    continue
                                end
                            end
                        end
                        i = i + 1; %更新零配件索引，继续生产
                    end
                    %计算利润
                    profit = total_sales * price_product - cost;
                    %保存当前方案和利润
                    profits(end+1) = profit;
                    schemes{end+1} = ['零配件检测: ', num2str([check_parts{:}]), ' 半成品检测: ', ...
                                      num2str([check_half_products{:}]), ' 成品检测: ', num2str(check_product), ...
                                      ' 半成品拆解: ', num2str([disassemble_half_products{:}]), ' 成品拆解: ', num2str(disassemble_product)];
                end
            end
        end
    end
end

%% 计算前20个利润最大的方案
[sorted_profits, indices] = sort(profits, 'descend');
top_20_profits = sorted_profits(1:20);
top_20_schemes = schemes(indices(1:20));
%绘制柱状图
figure;
barh(top_20_profits);
set(gca, 'yticklabel', top_20_schemes); %y轴为方案名
xlabel('Profit');
ylabel('Scheme');
title('Top 20 Profit Schemes');