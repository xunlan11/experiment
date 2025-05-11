import itertools
import random
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)  # 使用宋体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置随机种子
random.seed(12345)
np.random.seed(12345)


# 生成数据
def generate_data(length):
    data = np.random.randint(1, 100, size=(3 * length, 2 * length))
    return data


# 代码1
def code1(df, length, generations):
    values = [1, 2, 3]
    pop_size = 100
    clone = 2
    pm = 0.01

    class Individual:
        def __init__(self, genes):
            self.genes = genes
            self.fitness_value = None

        def fitness(self):
            if self.fitness_value is None:
                i = np.arange(length)
                r = (3 * i + self.genes - 1).repeat(2)
                cost_matrix = df[r, :]
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                self.fitness_value = cost_matrix[row_ind, col_ind].sum()
            return self.fitness_value

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

    population = [Individual([random.choice(values) for _ in range(length)]) for _ in range(pop_size)]
    best_individual = min(population, key=lambda x: x.fitness())
    no_improvement_count = 0
    for generation in range(generations):
        population.sort(key=lambda x: x.fitness(), reverse=False)
        selected = population[:pop_size // 2]
        new_population = clone_and_mutate(selected, clone, pm)
        population = new_population[:pop_size]
        generation_best_individual = min(population, key=lambda x: x.fitness())
        if generation_best_individual.fitness() == best_individual.fitness():
            no_improvement_count += 1
        else:
            best_individual = generation_best_individual
            no_improvement_count = 0
        if no_improvement_count >= 10:
            break
    return best_individual.fitness()


# 代码2
def code2(df, length, generations):
    values = [1, 2, 3]
    pop_size = 100
    Pc = 0.7
    Pm = 0.01

    class Individual:
        def __init__(self, genes):
            self.genes = genes
            self.fitness_value = None

        def fitness(self):
            if self.fitness_value is None:
                i = np.arange(length)
                r = (3 * i + self.genes - 1).repeat(2)
                cost_matrix = df[r, :]
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                total_cost = cost_matrix[row_ind, col_ind].sum()
                self.fitness_value = 1000 - total_cost
            return self.fitness_value

    def selection(population):
        total_fitness = sum(individual.fitness() for individual in population)
        selection_probs = [individual.fitness() / total_fitness for individual in population]
        selected = random.choices(population, weights=selection_probs, k=2)
        return selected

    def crossover(parent1, parent2):
        if random.random() < Pc:
            point = random.randint(1, length - 1)
            child1_genes = parent1.genes[:point] + parent2.genes[point:]
            child2_genes = parent2.genes[:point] + parent1.genes[point:]
            return Individual(child1_genes), Individual(child2_genes)
        else:
            return parent1, parent2

    def mutate(individual):
        for i in range(length):
            if random.random() < Pm:
                individual.genes[i] = random.choice(values)
        return individual

    population = [Individual([random.choice(values) for _ in range(length)]) for _ in range(pop_size)]
    best_individual = max(population, key=lambda x: x.fitness())
    no_improvement_count = 0
    for generation in range(generations):
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            new_population.append(child1)
            child2 = mutate(child2)
            new_population.append(child2)
        population = new_population[:pop_size]
        generation_best_individual = max(population, key=lambda x: x.fitness())
        if generation_best_individual.fitness() == best_individual.fitness():
            no_improvement_count += 1
        else:
            best_individual = generation_best_individual
            no_improvement_count = 0
        if no_improvement_count >= 10:
            break
    return 1000 - best_individual.fitness()


# 代码3
def code3(df, length):
    location_combinations = list(itertools.product(range(1, 4), repeat=length))
    solutions = {}
    for combination in location_combinations:
        i = np.arange(length)
        r = (3 * i + combination - 1).repeat(2)
        cost_matrix = df[r, :]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        fitness_value = cost_matrix[row_ind, col_ind].sum()
        solutions[combination] = fitness_value
    best_solution = min(solutions, key=solutions.get)
    best_value = solutions[best_solution]
    return best_value


# 测试不同长度
lengths = range(5, 11)
results = {5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
times = {5: [], 6: [], 7: [], 8: [], 9: [], 10: []}

num_runs = 5  # 运行次数

for length in lengths:
    df = generate_data(length)  # 为每个长度生成一次数据

    # 代码1和代码2运行5次
    results1 = []
    times1 = []
    results2 = []
    times2 = []
    for _ in range(num_runs):
        start_time = time.time()
        result1 = code1(df, length, 100)
        times1.append(time.time() - start_time)
        results1.append(result1)

        start_time = time.time()
        result2 = code2(df, length, 100)
        times2.append(time.time() - start_time)
        results2.append(result2)

    # 计算平均值
    avg_result1 = np.mean(results1)
    avg_time1 = np.mean(times1)
    avg_result2 = np.mean(results2)
    avg_time2 = np.mean(times2)

    results[length].extend([avg_result1, avg_result2])
    times[length].extend([avg_time1, avg_time2])

    # 代码3运行1次
    start_time = time.time()
    result3 = code3(df, length)
    times[length].append(time.time() - start_time)
    results[length].append(result3)

# 提取免疫匈牙利法和遗传匈牙利法的时间数据
immune_hungarian_times = [times[length][0] for length in lengths]
genetic_hungarian_times = [times[length][1] for length in lengths]

# 绘制折线图
plt.figure(figsize=(10, 6))

# 免疫匈牙利法
plt.plot(lengths, immune_hungarian_times, marker='o', linestyle='-', color='black', label='免疫匈牙利法')

# 遗传匈牙利法
plt.plot(lengths, genetic_hungarian_times, marker='*', linestyle='--', color='grey', label='遗传匈牙利法')

plt.xlabel('变量数', fontproperties=font)
plt.ylabel('时间 (秒)', fontproperties=font)
plt.title('两种方法的变量数与耗时关系', fontproperties=font)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()

# 绘制柱状图
plt.figure(figsize=(10, 6))
bar_width = 0.25
r1 = np.arange(len(lengths))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

plt.bar(r1, [results[length][0] for length in lengths], color='black', width=bar_width, edgecolor='black',
        label='免疫匈牙利法')
plt.bar(r2, [results[length][1] for length in lengths], color='grey', width=bar_width, edgecolor='black',
        label='遗传匈牙利法')
plt.bar(r3, [results[length][2] for length in lengths], color='white', width=bar_width, edgecolor='black', hatch='//',
        label='暴力遍历')

# 添加柱状图标注
for i in range(len(lengths)):
    plt.text(r1[i], results[lengths[i]][0] + 0.1, str(round(results[lengths[i]][0], 2)), ha='center', va='bottom',
             fontsize=10, fontproperties=font)
    plt.text(r2[i], results[lengths[i]][1] + 0.1, str(round(results[lengths[i]][1], 2)), ha='center', va='bottom',
             fontsize=10, fontproperties=font)
    plt.text(r3[i], results[lengths[i]][2] + 0.1, str(round(results[lengths[i]][2], 2)), ha='center', va='bottom',
             fontsize=10, fontproperties=font)

plt.xlabel('变量数', fontproperties=font)
plt.ylabel('结果', fontproperties=font)
plt.title('三个方法的结果对比', fontproperties=font)
plt.xticks([r + bar_width for r in range(len(lengths))], lengths)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.show()