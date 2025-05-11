import random
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Set random seed for reproducibility
np.random.seed(42)

# Generate data
def generate_data(length):
    df = pd.DataFrame(np.random.randint(50, 100, size=(3 * length, 2 * length)))
    return df.values

# Individual class
class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness_value = None

    def fitness(self, df, length):  # Hungarian method to evaluate fitness, store to reduce evaluations
        if self.fitness_value is None:
            i = np.arange(length)
            r = (3 * i + self.genes - 1).repeat(2)
            cost_matrix = df[r, :]
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            self.fitness_value = cost_matrix[row_ind, col_ind].sum()
        return self.fitness_value

# Clone and mutate
def clone_and_mutate(selected, clone, pm, df, length):
    new_population = []
    for individual in selected:
        for _ in range(clone):
            cloned = Individual(individual.genes.copy())
            for i in range(length):
                if random.random() < pm:
                    cloned.genes[i] = random.choice([1, 2, 3])
            new_population.append(cloned)
    return new_population

# Convert optimal solution
def solution(individual, df, length):
    i = np.arange(length)
    r = (3 * i + individual.genes - 1).repeat(2)
    cost_matrix = df[r, :]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for j in range(length):
        print(f"P{j + 1}.L{individual.genes[j]} to S{col_ind[2 * j] + 1}, S{col_ind[2 * j + 1] + 1}")

# Brute force search
def brute_force_search(df, length):
    from itertools import product
    location_combinations = list(product(range(1, 4), repeat=length))
    solutions = {}
    for combination in location_combinations:
        i = np.arange(length)
        r = (3 * i + np.array(combination) - 1).repeat(2)
        cost_matrix = df[r, :]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        fitness_value = cost_matrix[row_ind, col_ind].sum()
        solutions[combination] = fitness_value
    best_solution = min(solutions, key=solutions.get)
    best_value = solutions[best_solution]
    return best_solution, best_value

# Test running time for different lengths
lengths = list(range(3, 11))
times_genetic = []
times_brute_force = []

for length in lengths:
    df = generate_data(length)

    # Genetic algorithm
    start_time = time.time()
    population = [Individual([random.choice([1, 2, 3]) for _ in range(length)]) for _ in range(100)]
    best_individual = min(population, key=lambda x: x.fitness(df, length))
    best_fitness_history = [best_individual.fitness(df, length)]
    no_improvement_count = 0

    while True:
        population.sort(key=lambda x: x.fitness(df, length))
        selected = population[:50]
        new_population = clone_and_mutate(selected, 2, 0.01, df, length)
        population = new_population[:100]
        generation_best_individual = min(population, key=lambda x: x.fitness(df, length))
        generation_best_fitness = generation_best_individual.fitness(df, length)

        if generation_best_fitness == best_fitness_history[-1]:
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        best_fitness_history.append(generation_best_fitness)

        if no_improvement_count >= 10:
            break

    best_individual = min(population, key=lambda x: x.fitness(df, length))
    times_genetic.append(time.time() - start_time)

    # Brute force search
    start_time = time.time()
    best_solution, best_value = brute_force_search(df, length)
    times_brute_force.append(time.time() - start_time)

# 设置中文字体
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=12)  # 使用黑体

# Plot the results
plt.figure(figsize=(10, 6))

# 设置线条颜色为黑色
plt.plot(lengths, times_genetic, label='免疫匈牙利法', marker='o', color='black', linestyle='-')
plt.plot(lengths, times_brute_force, label='暴力遍历', marker='x', color='black', linestyle='--')

# 设置图例、标题、轴标签的颜色为黑色
plt.xlabel('变量数（配送点数量）', color='black', fontproperties=font)
plt.ylabel('运行时间（s）', color='black', fontproperties=font)
plt.title('两种方法的变量数与耗时关系', color='black', fontproperties=font)
plt.legend(frameon=False, fontsize=10, loc='upper left', prop=font)  # frameon=False 可以去掉图例周围的框线

# 设置刻度颜色为黑色
plt.tick_params(axis='both', which='both', colors='black')

# 显示网格，设置网格颜色为灰色（更接近黑白）
plt.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray')

plt.show()