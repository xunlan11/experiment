import numpy as np
import pandas as pd
import time

n = 24            # 基因数目（二进制编码长度）
population = 100  # 种群大小
generations = 2000# 迭代代数
X = [-3.0, 12.1]  # x取值范围
Y = [4.1, 5.8]    # y取值范围
repeat = 30       # 每组数据重复次数

# 二进制与浮点数转化
def translateDNA(pop):
    x_pop = pop[:, 1::2]
    y_pop = pop[:, ::2]
    x = x_pop.dot(2 ** np.arange(n)[::-1]) / float(2 ** n - 1) * (X[1] - X[0]) + X[0]
    y = y_pop.dot(2 ** np.arange(n)[::-1]) / float(2 ** n - 1) * (Y[1] - Y[0]) + Y[0]
    return x, y

# 种群适应度
def get_fitness(pop):
    x, y = translateDNA(pop)
    pred = 21.5 + x * np.sin(4 * np.pi * x) + y * np.sin(20 * np.pi * y)
    return pred

# 交配、变异
def crossover_and_mutation(pop):
    new_pop = []
    for father in pop:
        child = father.copy()
        # 交叉：单点交叉
        if np.random.rand() < Pc:
            mother = pop[np.random.randint(population)]
            cross_point = np.random.randint(low=0, high=2 * n)
            child[cross_point:] = mother[cross_point:]
        # 变异：单点变异
        for point in range(2*n):
            if np.random.rand() < Pm:
                child[point] = child[point] ^ 1
        new_pop.append(child)
    return new_pop

# 选择：轮盘赌
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
            pop = np.random.randint(2, size=(population, n * 2))
            for _ in range(generations):
                x, y = translateDNA(pop)
                pop = np.array(crossover_and_mutation(pop))
                fitness = get_fitness(pop)
                pop = select(pop, fitness)
            best_fitness = np.max(get_fitness(pop))
            best_fitnesses.append(best_fitness)
            max_fitness_index = np.argmax(get_fitness(pop))
            # x, y = translateDNA(pop)
            # print("(%.5f,%.5f)" % (x, y))
        mean_best_fitness = np.mean(best_fitnesses)
        fitness_list[int(Pc * 10) - 1, int(Pm * 100) - 1] = mean_best_fitness
        print("在Pc=%.1f、Pm=%.2f的条件下，多次遗传算法的均值为%.5f" % (Pc, Pm, mean_best_fitness))
end_time = time.time()
print("二进制编码代码运行时间：%.2f秒" % (end_time - start_time))
df = pd.DataFrame(fitness_list, index=range(1, 10), columns=range(1, 10))
file_path = 'results.xlsx'
with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='bin', index=True, header=True)