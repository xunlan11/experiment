import numpy as np

n = 24            # 基因数目（二进制编码长度）
population = 100  # 种群大小
Pc = 0.7          # 交配率
Pm = 0.07         # 变异率
generations = 200 # 迭代代数
X = [-3.0, 12.1]  # x取值范围
Y = [4.1, 5.8]    # y取值范围

# 二进制编码转换为实数域x和y
def translateDNA(pop):
    x_pop = pop[:, 1::2]
    y_pop = pop[:, ::2]
    x = x_pop.dot(2 ** np.arange(n)[::-1]) / float(2 ** n - 1) * (X[1] - X[0]) + X[0]
    y = y_pop.dot(2 ** np.arange(n)[::-1]) / float(2 ** n - 1) * (Y[1] - Y[0]) + Y[0]
    return x, y

# 种群适应度
def get_fitness(pop):
    x, y = translateDNA(pop)
    # 适应度函数
    pred = 21.5 + x * np.sin(4 * np.pi * x) + y * np.sin(20 * np.pi * y)
    return (pred - np.min(pred)) + 1e-3

# 交配、变异
def crossover_and_mutation(pop):
    new_pop = []
    for father in pop:
        child = father.copy()
        # 交配
        if np.random.rand() < Pc:
            mother = pop[np.random.randint(population)]
            cross_points = np.random.randint(low=0, high=n * 2)
            child[cross_points:] = mother[cross_points:]
        # 变异
        if np.random.rand() < Pm:
            mutate_point = np.random.randint(0, n * 2)
            child[mutate_point] = child[mutate_point] ^ 1
        new_pop.append(child)
    return new_pop

# 选择：轮盘赌
def select(pop, fitness):
    idx = np.random.choice(np.arange(population), size=population, replace=True, p=fitness / fitness.sum())
    return pop[idx]

if __name__ == "__main__":
    pop = np.random.randint(2, size=(population, n * 2))
    for _ in range(generations):
        x, y = translateDNA(pop)
        pop = np.array(crossover_and_mutation(pop))
        fitness = get_fitness(pop)
        pop = select(pop, fitness)
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))
