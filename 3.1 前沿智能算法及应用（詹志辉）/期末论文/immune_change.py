import random
import pandas as pd
import numpy as np
import time

# 数据
df = pd.read_excel('data.xlsx')
df = df.values
df = df[:, 1:11].astype(np.int64)
# 参数
length1 = 10       # 个体基因数量1——零售商数目
length2 = 5        # 个体基因数量2——配送点数目
generations = 100  # 迭代次数
pop_size = 100     # 种群大小
clone_factor = 2   # 克隆因子
pm = 0.05          # 变异概率

# 个体类
class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness_value = None
    def fitness(self):
        fitness_value = 0
        for i in range(length1):
            row_index = 3 * self.genes[i] + self.genes[self.genes[i] + length1]
            col_index = i
            fitness_value += df[row_index, col_index]
        return fitness_value

# 初始化种群
def init_population():
    population = []
    for _ in range(pop_size):
        # 前10位确保0-4每个数字出现两次
        front = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        random.shuffle(front)
        # 后5位在0-2之间随机选取
        back = [random.randint(0, 2) for _ in range(length2)]
        individual = Individual(front + back)
        population.append(individual)
    return population

# 选择
def select(population, num_best):
    fitnesses = [i.fitness() for i in population]
    best_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:num_best]
    return [population[i] for i in best_indices]

# 克隆和变异
def clone_and_mutate(selected, clone_factor, pm, length1, length2):
    new_population = []
    for individual in selected:
        for _ in range(clone_factor):
            cloned = Individual(individual.genes.copy())
            if random.random() < 0.5:
                for i in range(length1):
                    p = random.random()
                    if p < pm: # 选择另一个不同的索引进行交换
                        j = random.choice([x for x in range(10) if x != i])
                        cloned.genes[i], cloned.genes[j] = cloned.genes[j], cloned.genes[i]
                    elif pm <= p <= 1.1 * pm:  # 区间反转
                        start, end = sorted(random.sample(range(10), 2))
                        cloned.genes[start:end + 1] = reversed(cloned.genes[start:end + 1])
                    elif 1.1 * pm <= p <= 1.2 * pm:  # 基因移动
                        j = random.sample(range(10), 1)[0]
                        gene = cloned.genes.pop(i)
                        cloned.genes.insert(j, gene)
            else:
                for i in range(length1, length1 + length2):
                    if random.random() < pm:
                        cloned.genes[i] = random.randint(0, 2)
            new_population.append(cloned)
    return new_population

# 转化最优解方案
def solution(individual):
    front_genes =  np.array(individual.genes[:10])
    for i in range(length1, length1 + length2):
        indices = np.flatnonzero(front_genes == (i - 10))
        print(f"P{i - 9}.L{individual.genes[i]}到S{indices[0]},S{indices[1]}")

# 主函数
start_time = time.time()
population = init_population()
best_individual = min(population, key=lambda x: x.fitness())  # 保留迭代过程中的最优个体
for generation in range(generations):
    selected = select(population, pop_size // 2)
    new_population = clone_and_mutate(selected, clone_factor, pm, length1, length2)
    population = new_population[:pop_size]
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