import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 参数
length = 34            # 基因数目
population_size = 50   # 种群大小
generations = 500      # 迭代代数
Pc = 0.6               # 交叉因子
Pm = 0.02              # 变异因子
elite = 0.1            # 精英保留比例

# 读取数据
file_path_distances = '城市经纬距离矩阵.xlsx'
df_distances = pd.read_excel(file_path_distances, index_col=0)
distance_matrix = df_distances.iloc[:length, :length].values
city_names = df_distances.index[:length]
file_path_cities = '城市经纬度.xlsx'
df_cities = pd.read_excel(file_path_cities, index_col=0)

# 个体类
class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness_value = None
    def fitness(self): # 个体生成后只评估一次，节省时间
        if self.fitness_value is None:
            self.fitness_value = sum([distance_matrix[self.genes[i]][self.genes[(i + 1) % length]] for i in range(length)])
        return self.fitness_value

# 交叉：单点交叉
def crossover(parent1, parent2, pc):
    if np.random.rand() < pc:
        cross_point = np.random.randint(1, length)
        child1_genes = parent1.genes[:cross_point] + [gene for gene in parent2.genes if gene not in parent1.genes[:cross_point]]
        child2_genes = parent2.genes[:cross_point] + [gene for gene in parent1.genes if gene not in parent2.genes[:cross_point]]
        return Individual(child1_genes), Individual(child2_genes)
    else:
        return Individual(parent1.genes.copy()), Individual(parent2.genes.copy())  # 避免修改原始种群个体

# 变异：交换、区间反转、基因移动、片段移动
def mutation(individual, pm):
    # 创建个体的副本，防止修改原始个体
    new_genes = individual.genes.copy()
    # 方法选择概率
    methods = ['swap', 'inversion', 'insertion', 'shift']
    probabilities = [0.7, 0.1, 0.1, 0.1]
    for i in range(length):
        if np.random.rand() < pm:
            method = np.random.choice(methods, p=probabilities)
            if method == 'swap':
                j = np.random.choice([x for x in range(length) if x != i])
                new_genes[i], new_genes[j] = new_genes[j], new_genes[i]
            elif method == 'inversion':
                start, end = sorted(random.sample(range(length), 2))
                new_genes[start:end + 1] = reversed(new_genes[start:end + 1])
            elif method == 'insertion':
                j = random.sample(range(length), 1)[0]
                gene = new_genes.pop(i)
                new_genes.insert(j, gene)
            elif method == 'shift':
                start, end = sorted(random.sample(range(1,length-1), 2))
                subsequence = new_genes[start:end + 1]
                sublength = len(subsequence)
                k = random.choice([x for x in range(length - sublength)])   # 保证插入位置在剩余序列内
                del new_genes[start:end + 1]
                for item in reversed(subsequence):
                    new_genes.insert(k, item)
    return Individual(new_genes)

# 选择：轮盘赌选择
def select(pop, fitness):
    probabilities = 1 / np.array(fitness)
    probabilities /= probabilities.sum()  # 归一化
    selected_indices = np.random.choice(np.arange(population_size), size=2, replace=True, p=probabilities)
    return pop[selected_indices[0]], pop[selected_indices[1]]

# 演化过程
def evolution(fitness, title):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, generations + 1), fitness, label=title)
    plt.xlabel('迭代次数')
    plt.ylabel('最小距离')
    plt.title('演化过程')
    plt.legend()
    plt.show()

# 绘制路径
def plot_path(best_individual, city_names, distance_matrix):
    # 模拟地图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    # 绘制城市点并标注城市名
    path = [city_names[i] for i in best_individual.genes]
    lats = distance_matrix.loc[path, 'Latitude'].values
    lons = distance_matrix.loc[path, 'Longitude'].values
    ax.scatter(lons, lats, 60, marker='o', color='k', zorder=5, transform=ccrs.PlateCarree(), label='城市')
    for i, name in enumerate(path):
        ax.text(lons[i], lats[i], name, fontsize=10, ha='right', va='bottom', transform=ccrs.PlateCarree())
    # 绘制路径（回到起点）
    ax.plot(lons, lats, 'r-', linewidth=2, label='路径', transform=ccrs.PlateCarree())
    ax.plot([lons[-1], lons[0]], [lats[-1], lats[0]], 'r-', linewidth=2, transform=ccrs.PlateCarree())
    plt.show()

# 主程序
start_time = time.time()
generation_best_fitness = [] # 世代最优
best_fitness = []            # 全局最优
population = [Individual(random.sample(range(length), length)) for _ in range(population_size)]  # 种群初始化
best_individual = min(population, key=lambda x: x.fitness())  # 保留迭代过程中的最优个体
for generation in range(generations):
    sorted_population = sorted(population, key=lambda x: x.fitness())
    new_population = sorted_population[:int(elite * population_size)]
    while len(new_population) < population_size:
        parent1, parent2 = select(population, [ind.fitness() for ind in population])
        child1, child2 = crossover(parent1, parent2, Pc)
        child1 = mutation(child1, Pm)
        child2 = mutation(child2, Pm)
        new_population.extend([child1, child2])
    population = new_population[:population_size]
    # 最佳个体
    generation_best_individual = min(population, key=lambda x: x.fitness())
    generation_best_fitness.append(generation_best_individual.fitness())
    if generation_best_individual.fitness() < best_individual.fitness():
        best_individual = generation_best_individual
    best_fitness.append(best_individual.fitness())
    if (generation + 1) % 100 == 0:
        print(f"第{generation + 1}轮: 当前最优方案{best_individual.genes}的花费为{best_individual.fitness()}")

# 输出最终结果
print(f"最短路径:{best_individual.genes}")
print(f"最短距离:{best_individual.fitness():.5f}经纬距离")
end_time = time.time()
print(f"运行时间:{end_time - start_time:.5f}s")
evolution(generation_best_fitness, '世代最优')
evolution(best_fitness, '全局最优')
plot_path(best_individual, city_names, df_cities)