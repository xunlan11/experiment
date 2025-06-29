import random
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

# 数据
df = pd.read_excel('data.xlsx')
df = df.values
df = df[:, 1:11].astype(np.int64)
# 参数
length = 5         # 个体基因数量——配送点数目
values = [1, 2, 3] # 可能基因型——配送点的可能选择
generations = 10   # 迭代次数
pop_size = 100     # 种群大小
clone = 2          # 克隆因子
pm = 0.01          # 变异概率

# 个体类
class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness_value = None
    def fitness(self):  # 匈牙利法评估适应值，储存以减少评估次数
        if self.fitness_value is None:
            # 抽取基因型对应的行，构成匈牙利法的效率矩阵
            i = np.arange(length)
            r = (3 * i + self.genes - 1).repeat(2)
            cost_matrix = df[r, :]
            # 匈牙利法求min值
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            self.fitness_value = cost_matrix[row_ind, col_ind].sum()
        return self.fitness_value

# 克隆和变异
def clone_and_mutate(selected, clone, pm):
    new_population = []
    for individual in selected:
        for _ in range(clone):
            cloned = Individual(individual.genes.copy())
            for i in range(length):
                if random.random() < pm:
                    cloned.genes[i] = random.choice(values)
            new_population.append(cloned)
    return new_population

# 转化最优解方案
def solution(individual):
    i = np.arange(length)
    r = (3 * i + individual.genes - 1).repeat(2)
    cost_matrix = df[r, :]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for j in range(length):
        print(f"P{j + 1}.L{individual.genes[j]}到S{col_ind[2 * j] + 1},S{col_ind[2 * j + 1] + 1}")

# 主函数
start_time = time.time()
population = [Individual([random.choice(values) for _ in range(length)]) for _ in range(pop_size)]
best_individual = min(population, key=lambda x: x.fitness())  # 保留迭代过程中的最优个体
for generation in range(generations):
    # 选择适应度低的个体
    population.sort(key=lambda x: x.fitness(), reverse=False)
    selected = population[:pop_size // 2]
    # 克隆并变异，获取新种群
    new_population = clone_and_mutate(selected, clone, pm)
    population = new_population[:pop_size]
    # 记录最佳个体
    generation_best_individual = min(population, key=lambda x: x.fitness())
    print(f"第{generation + 1}轮: 最优方案{generation_best_individual.genes}的花费为{generation_best_individual.fitness()}")
    if generation_best_individual.fitness() < best_individual.fitness():
        best_individual = generation_best_individual
# 输出最终结果
print("最优方案", best_individual.genes, "的花费为", best_individual.fitness())
solution(best_individual)
# 运行时间
end_time = time.time()
print("运行时间：%.5f秒" % (end_time - start_time))