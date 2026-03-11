varP = 0.01:0.1:0.95;
profit_p = zeros(10, 1);
for ext = 1:10
    %% 输入已知量
    cost_parts = [2, 8,12, 2, 8, 12, 8, 12]; %八种零件的采购成本
    cost_check_parts = [1, 1, 2, 1, 1, 2, 1, 2]; %八种零件的检测成本
    defective_parts = [varP(ext), varP(ext), varP(ext), varP(ext), varP(ext), varP(ext), varP(ext), varP(ext)]; %八种零件的次品率
    cost_assembly_half_products=[8, 8, 8]; %三个半成品的装配成本
    cost_check_half_products=[4,4,4]; %三个半成品的检测成本
    cost_disassemble_half_products = [6, 6, 6]; %三个半成品的拆解费用
    defective_half_products = [varP(ext), varP(ext), varP(ext)]; %三个半成品的次品率
    cost_assembly_product = 8; %成品的装配成本
    cost_check_product = 6; %成品的检测成本
    cost_disassemble_product = 10; %成品的拆解费用
    defective_product = varP(ext); %成品的次品率
    price_product = 200; %成品售价
    cost_replace = 40; %不合格成品调换损失
    %% 初始化变量
    num_parts = 500; %每种零件10000个，包括次品和良品
    num_bad_parts = round(num_parts.*defective_parts); %根据次品率计算每种零件的次品数量
    num_good_parts=num_parts-num_bad_parts; %计算每种零件的良品数量
    best_profit=-Inf; %初始化最大利润为负无穷
    best_scheme=[]; %初始化保存最优方案的变量
    %% 遍历所有可能的生产决策组合
    for check_parts_bin = 0:255  %遍历所有检测零件的组合
        check_parts = num2cell(dec2bin(check_parts_bin, 8) - '0'); %生成8个零件的二进制决策
        for check_half_products_bin = 0:7 % 遍历半成品检测的组合
            check_half_products = num2cell(dec2bin(check_half_products_bin, 3) - '0'); %生成3个半成品的二进制决策
            for disassemble_half_products_bin = 0:7 %遍历半成品拆解的组合
                disassemble_half_products = num2cell(dec2bin(disassemble_half_products_bin, 3) - '0'); %生成3个半成品的二进制决策
                for check_product = [0, 1] %是否检测成品
                    for disassemble_product = [0, 1] %是否拆解成品
                        %初始化仓库中的零件和使用次数
                        parts = cell(1, 8); %创建一个包含8种零件的cell数组
                        for k = 1:8
                            %每种零件的良品和次品
                            parts{k} = [ones(1, num_good_parts(k)), zeros(1, num_bad_parts(k))];
                            %随机打乱顺序
                            parts{k} = parts{k}(randperm(num_parts)); 
                        end
                        uses_parts = zeros(1, 8); %使用数值数组记录每个零件的使用次数
                        total_sales = 0; %总销售数量
                        cost = 0;        %总成本
                        i = ones(1, 8);  %各个零件的索引
                        %模拟生产过程
                        %假设生产线能处理的最大成品数量为num_parts/2，确保足够多的零件进行配对
                        num_products_to_assemble = floor(num_parts / 2); 
                        for product_num = 1:num_products_to_assemble
                            %检查是否还有足够的零件继续生产
                            if any(i > num_parts)
                                break; %如果任意一个零件不足，停止生产
                            end
                            %获取当前零件
                            parts_current = cellfun(@(x, idx) x(idx), parts, num2cell(i), 'UniformOutput', false);
                            %增加当前零件的使用次数
                            for k = 1:8
                                uses_parts(k) = uses_parts(k) + 1;
                            end
                            %根据方案检测零件
                            discard = false;
                            for k = 1:8
                                %检查零件是否需要检测，以及是否为次品
                                if check_parts{k} == 1 && parts_current{k} == 0
                                    discard = true;
                                    i(k) = i(k) + 1; %丢弃并跳过该零件
                                    break;
                                end
                            end
                            if discard
                                continue; %跳过本轮生产
                            end
                            %模拟装配和检测半成品
                            half_products = [all([parts_current{1:3}]), all([parts_current{4:6}]), all([parts_current{5:6}])];
                            half_products_costs = arrayfun(@(x) x + cost_assembly_half_products(x), 1:3);
    
                            %检测半成品
                            for k = 1:3
                                if check_half_products{k}
                                    if rand() <= defective_half_products(k)
                                        if disassemble_half_products{k}
                                            cost = cost + cost_disassemble_half_products(k); %增加拆解费用
                                        end
                                        half_products(k) = false; %标记半成品为次品
                                    end
                                end
                            end
                            %装配成品
                            if all(half_products)
                                product = rand() > defective_product; %即使半成品良品，成品也可能是次品
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
                            %更新零件索引，继续生产
                            i = i + 1; %每次处理一个零件
                        end
                        %计算利润
                        profit = total_sales * price_product - cost;
                        %更新最优方案
                        if profit > best_profit
                            best_profit = profit;
                            best_scheme = [check_parts{:}, check_half_products{:}, disassemble_half_products{:}, check_product, disassemble_product];
                        end
                    end
                end
            end
        end
    end
    %输出最优方案
    fprintf('最优方案：\n检测零件：%s\n检测半成品：%s\n拆解半成品：%s\n检测成品：%d\n拆解成品：%d\n最大利润: %.2f\n', ...
        num2str(best_scheme(1:8)), num2str(best_scheme(9:11)), num2str(best_scheme(12:14)), best_scheme(15), best_scheme(16), best_profit);
    profit_p(ext) = best_profit;
end