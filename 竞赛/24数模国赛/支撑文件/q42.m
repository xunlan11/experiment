profit_p = zeros([6, 10, 10, 10]);
best_sol = zeros(6, 10, 10, 10, 4);
p = 0.01:0.1:0.95;
%已知量
for m = 1:10
    for n = 1:10
        for q = 1:10
            data=[p(m) p(m) p(m) p(m) p(m) p(m); %零件1次品率
                  2   2   2   1   8   2   ; %零件1检测成本
                  p(n) p(n) p(n) p(n) p(n) p(n); %零件2次品率
                  3   3   3   1   1   3   ; %零件2检测成本
                  p(q) p(q) p(q) p(q) p(q) p(q); %成品次品率
                  3   3   3   2   2   3   ; %成品检测成本
                  6   6   30  30  10  10  ; %调换损失
                  5   5   5   5   5   40 ]; %拆解费用
            cost_part_1 = 4;     %零件1购买单价
            cost_part_2 = 18;    %零件2的购买单价
            cost_assembly = 6;   %装配成本
            price_product = 56;  %市场售价
            fprintf('p_1=%.2f  ', p(m));
            fprintf('p_2=%.2f  ', p(n));
            fprintf('p_3=%.2f', p(q));
            %六种情况
            for cases =1:6
                %% 初始化变量    
                defective_part_1 = data(1,cases);  %零件1次品率 
                cost_check_part_1 = data(2,cases); %零件1检测成本
                defective_part_2 = data(3,cases);  %零件2次品率
                cost_check_part_2 = data(4,cases); %零件2检测成本
                defective_product = data(5,cases); %成品次品率
                cost_check_product = data(6,cases);%成品检测成本
                cost_replace = data(7,cases);      %调换损失
                cost_disassemble = data(8,cases);  %拆解费用  
                num_parts = 10000;                         %模拟投入生产线总共的零件数
                num_bad_1 = num_parts * defective_part_1;  %次品零件1
                num_bad_1 = ceil(num_bad_1);
                num_good_1 = num_parts - num_bad_1;        %良品零件1
                num_good_1 = ceil(num_good_1);
                num_bad_2 = num_parts * defective_part_2;  %次品零件2
                num_bad_2 = ceil(num_bad_2);
                num_good_2 = num_parts - num_bad_2;        %良品零件2
                num_good_2 = floor(num_good_2);
                max_uses = 3;            %零件最多使用次数为3次
                profits = zeros(1, 16);  %记录各方案的利润
                best_profit = -Inf;      %最大利润
                best_scheme = [];        %最优方案
                index = 1;               %方案索引 
                %% 遍历所有生产决策组合（16种组合，分别表示每个阶段的检测决策）
                for check_part_1 = [0, 1]
                    for check_part_2 = [0, 1]
                        for check_product = [0, 1]
                            for disassemble_product = [0, 1]
                                %初始化仓库中的零件和使用次数
                                parts_1 = [ones(1, num_good_1), zeros(1, num_bad_1)];
                                parts_2 = [ones(1, num_good_2), zeros(1, num_bad_2)];
                                parts_1 = parts_1(randperm(num_parts));  %随机打乱
                                parts_2 = parts_2(randperm(num_parts));
                                uses_part_1 = zeros(1, num_parts);  %初始化零件的使用次数
                                uses_part_2 = zeros(1, num_parts); 
                                profit = 0;       %初始化利润
                                total_sales = 0;  %初始化成交数
                                i = 1;            %初始化零件1索引
                                j = 1;            %初始化零件2索引
                                cost=0;
                                %模拟生产过程，直到某种零件用完
                                while i <= length(parts_1) && j <= length(parts_2)
                                    cost = cost + cost_part_1 + cost_part_2;
                                    %获取当前取出的零件1和零件2
                                    part_1 = parts_1(i);
                                    part_2 = parts_2(j);
                                    %检查零件的使用次数，超出使用次数则更换零件
                                    if uses_part_1(i) >= max_uses
                                        i = i + 1;
                                        continue;  %零件使用超过上限，丢弃
                                    end
                                    if uses_part_2(j) >= max_uses
                                        j = j + 1;
                                        continue; 
                                    end
                                    %增加使用次数
                                    uses_part_1(i) = uses_part_1(i) + 1;
                                    uses_part_2(j) = uses_part_2(j) + 1;
                                    %根据方案决定是否检测零件1
                                    if check_part_1
                                        cost = cost + cost_check_part_1;
                                        if part_1 == 0  %次品
                                            i = i + 1;
                                            continue;  %丢弃次品，取下一个零件1
                                        end
                                    end
                                    %根据方案决定是否检测零件2
                                    if check_part_2
                                        cost = cost + cost_check_part_2;
                                        if part_2 == 0 
                                            j = j + 1;
                                            continue; 
                                        end
                                    end
                                    %装配成品
                                    cost = cost + cost_assembly;
                                    product = part_1 == 1 && part_2 == 1;  %判断零件是否都合格
                                    %即使零件1和零件2都是良品，成品也有10%的概率次品
                                    if product
                                        if rand() <= defective_product
                                            product = 0;  %模拟成品为次品的概率
                                        end
                                    end
                                    %成品检测
                                    if check_product
                                        cost = cost + cost_check_product;
                                        if ~product
                                            %成品检测不合格
                                            if disassemble_product  %决定是否拆解成品
                                                cost = cost + cost_disassemble;
                                                continue;
                                            else
                                                i = i + 1;
                                                j = j + 1;
                                                continue;
                                            end
                                        end
                                    end
                                    %客户收到好货，更新利润和销售数
                                    if product
                                        total_sales = total_sales + 1;
                                        i = i + 1;
                                        j = j + 1;
                                    end  
                                    %如果客户收到次品
                                    if ~product
                                        cost = cost + cost_replace;
                                        if disassemble_product  %决定是否拆解成品
                                            cost = cost + cost_disassemble;
                                            continue
                                        else
                                            i = i + 1;
                                            j = j + 1;
                                            continue
                                        end
                                    end
                                end  %循环到这里
                                profit = total_sales * price_product-cost;
                                profits(index) = profit;  
                                index = index + 1; 
                                %更新最优方案
                                if profit > best_profit
                                    best_profit = profit;
                                    best_scheme = [check_part_1, check_part_2, check_product, disassemble_product];
                                end
                            end
                        end
                    end
                end
                %输出最优方案
                profit_p(cases, m, n, q) = best_profit;
                best_sol(cases, m, n, q, :) = best_scheme;
                fprintf('第%d个情况的最优决策方案：\n', cases);
                fprintf('检测零件1: %d\n', best_scheme(1));
                fprintf('检测零件2: %d\n', best_scheme(2));
                fprintf('检测成品: %d\n', best_scheme(3));
                fprintf('拆解次品: %d\n', best_scheme(4));
                fprintf('最大利润: %.2f\n', best_profit);
            end
        end
    end
end