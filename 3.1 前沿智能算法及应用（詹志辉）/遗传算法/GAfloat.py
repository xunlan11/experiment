import numpy as np
import pandas as pd
import time

n = 2             # 基因数目
population = 100  # 种群大小
generations = 2000# 迭代代数
X = [-3.0, 12.1]  # x取值范围
Y = [4.1, 5.8]    # y取值范围
bounds = [X, Y]   # 边界条件
repeat = 30       # 每组数据重复次数

# 种群初始化
def create_population(size, bounds):
    return np.random.rand(size, len(bounds)) * (np.array(bounds)[:, 1] - np.array(bounds)[:, 0]) + np.array(bounds)[:,0]

# 适应度函数（函数一定非负，所以未进行保证非负的处理）
def get_fitness(pop):
    x = pop[:, 0]
    y = pop[:, 1]
    pred = 21.5 + x * np.sin(4 * np.pi * x) + y * np.sin(20 * np.pi * y)
    return pred

# 交叉：单点交叉
def crossover(pop, pc):
    child_pop = np.empty_like(pop, dtype=pop.dtype)
    for i in range(0, len(pop) - 1, 2):
        if np.random.rand() < pc:
            cross_point = np.random.randint(0, n)
            # 交换交叉点后的基因
            child_pop[i, :] = np.concatenate([pop[i, :cross_point], pop[i + 1, cross_point:]])
            child_pop[i + 1, :] = np.concatenate([pop[i + 1, :cross_point], pop[i, cross_point:]])
        else:
            # 不进行交叉，直接复制
            child_pop[i, :] = pop[i, :]
            child_pop[i + 1, :] = pop[i + 1, :]
    return child_pop

# 变异：单点变异
def mutation(pop, pm, bounds):
    for i in range(len(pop)):
        for j in range(n):
            if np.random.rand() < pm:
                temp=np.random.uniform(bounds[j][0], bounds[j][1])
                pop[i, j] =temp
    return pop

# 选择：轮盘赌选择
def select(pop, fitness):
    idx = np.random.choice(np.arange(population), size=population, replace=True, p=fitness / fitness.sum())
    return pop[idx]

start_time = time.time()
Pcs = np.linspace(0.1, 0.9, 9)
Pms = np.linspace(0.01, 0.09, 9)
fitness_list = np.zeros((9, 9))
for Pc in Pcs:
    for Pm in Pms:
        best_fitnesses = []
        for _ in range(repeat):
            pop = create_population(population, bounds)
            # best_idx = None
            for _ in range(generations):
                fitness = get_fitness(pop)
                # if best_idx is None or fitness[best_idx] < np.max(fitness):
                    # best_idx = np.argmax(fitness)
                pop = crossover(pop, Pc)
                pop = mutation(pop, Pm, bounds)
                pop = select(pop, fitness)
            # x = pop[best_idx, 0]
            # y = pop[best_idx, 1]
            # print("(%.5f,%.5f)" % (x, y))
            best_fitness = np.max(get_fitness(pop))
            best_fitnesses.append(best_fitness)
        mean_best_fitness = np.mean(best_fitnesses)
        fitness_list[int(Pc * 10) - 1, int(Pm * 100) - 1] = mean_best_fitness
        print("在Pc=%.1f、Pm=%.2f的条件下，多次遗传算法的均值为%.5f" % (Pc, Pm, mean_best_fitness))
end_time = time.time()
print("浮点数编码代码运行时间：%.2f秒" % (end_time - start_time))
df = pd.DataFrame(fitness_list, index=range(1, 10), columns=range(1, 10))
file_path = 'results.xlsx'
with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='float', index=True, header=True)