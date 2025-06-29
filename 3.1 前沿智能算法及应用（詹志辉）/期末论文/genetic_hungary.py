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
length = 5          # 个体基因数量——配送点数目
values = [1, 2, 3]  # 可能基因型——配送点的可能选择
generations = 10    # 迭代次数
pop_size = 100      # 种群大小
Pc = 0.7            # 交叉概率
Pm = 0.01           # 变异概率

# 个体类
class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness_value = None
    def fitness(self):    # 匈牙利法评估适应值，储存以减少评估次数
        if self.fitness_value is None:
            # 抽取基因型对应的行，构成匈牙利法的效率矩阵
            i = np.arange(length)
            r = (3 * i + self.genes - 1).repeat(2)
            cost_matrix = df[r, :]
            # 匈牙利法求min值，并用与1000的差来表示适应值，以使值大的费用低
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_cost = cost_matrix[row_ind, col_ind].sum()
            self.fitness_value = 1000 - total_cost
        return self.fitness_value

# 选择：轮盘赌
def selection(population):
    total_fitness = sum(individual.fitness() for individual in population)
    selection_probs = [individual.fitness() / total_fitness for individual in population]
    selected = random.choices(population, weights=selection_probs, k=2)
    return selected

# 交叉：单点交叉
def crossover(parent1, parent2):
    if random.random() < Pc:
        point = random.randint(1, length - 1)
        child1_genes = parent1.genes[:point] + parent2.genes[point:]
        child2_genes = parent2.genes[:point] + parent1.genes[point:]
        return Individual(child1_genes), Individual(child2_genes)
    else:
        return parent1, parent2

# 变异：单点变异
def mutate(individual):
    for i in range(length):
        if random.random() < Pm:
            individual.genes[i] = random.choice(values)
    return individual

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
best_individual = max(population, key=lambda x: x.fitness())   # 保留迭代过程中的最优个体
for generation in range(generations):
    new_population = []
    while len(new_population) < pop_size:
        parent1, parent2 = selection(population)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        new_population.append(child1)
        child2 = mutate(child2)
        new_population.append(child2)
    population = new_population
    # 记录最佳个体
    generation_best_individual = max(population, key=lambda x: x.fitness())
    print(f"第{generation + 1}轮: 最优方案{generation_best_individual.genes}的花费为{1000 - generation_best_individual.fitness()}")
    if generation_best_individual.fitness() > best_individual.fitness():
        best_individual = generation_best_individual
# 输出最终结果
print("最优方案", best_individual.genes, "的花费为", 1000 - best_individual.fitness())
solution(best_individual)
# 运行时间
end_time = time.time()
print("运行时间：%.5f秒" % (end_time - start_time))