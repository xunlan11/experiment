import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 参数
length = 34        # 城市数量
num_ants = 30      # 蚂蚁数量
generations = 10   # 迭代次数
Q = 10             # 信息素释放总量
rho = 0.5          # 信息素挥发因子
alpha = 1          # 信息素重要性因子
beta = 5           # 启发式信息重要性因子

# 读取数据
file_path_distances = '城市经纬距离矩阵.xlsx'
df_distances = pd.read_excel(file_path_distances, index_col=0)
distance_matrix = df_distances.iloc[:length, :length].values
city_names = df_distances.index[:length]
file_path_cities = '城市经纬度.xlsx'
df_cities = pd.read_excel(file_path_cities, index_col=0)

# 路径类
class Path:
    def __init__(self, path):
        self.path = path
        self.len = None
    def length(self):
        if self.len is None:
            self.len = sum([distance_matrix[self.path[i]][self.path[(i + 1) % length]] for i in range(length)])
        return self.len

# 选择下一个城市：轮盘赌选择
def select_next_city(current_city, unvisited_cities):
    pheromones = pheromone_matrix[current_city][unvisited_cities]    # 信息素部分
    heuristic = 1 / distance_matrix[current_city][unvisited_cities]  # 启发式信息部分（距离倒数）
    probabilities = (pheromones ** alpha) * (heuristic ** beta)      # 用因子指数化
    probabilities /= probabilities.sum()  # 归一化
    next_city_index = np.random.choice(len(unvisited_cities), p=probabilities) # 轮盘赌选择
    return unvisited_cities[next_city_index]

# 更新信息素矩阵
def update_pheromones(paths):
    global pheromone_matrix        # 全局变量保证更改
    pheromone_matrix *= 1 - rho    # 挥发
    # 增加
    for path in paths:           # 所有蚂蚁
        for j in range(length):  # 所有边
            start, end = path.path[j], path.path[(j + 1) % length]
            pheromone_matrix[start][end] += Q / path.length()  # 释放总量除以距离
            pheromone_matrix[end][start] = pheromone_matrix[start][end]   # 对称矩阵

# 演化过程
def evolution(best_lengths):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(best_lengths) + 1), best_lengths)
    plt.xlabel('迭代次数')
    plt.ylabel('最短距离')
    plt.title('演化过程')
    plt.show()

# 绘制路径
def plot_path(best_path, city_names, distance_matrix):
    # 模拟地图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    # 绘制城市点并标注城市名
    path = [city_names[i] for i in best_path.path]
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
pheromone_matrix = np.ones((length, length))   # 信息素矩阵
best_length_history = []          # 世代最优
best_path = Path(list(range(34))) # 保留迭代过程中的最优个体
for generation in range(generations):
    paths = []
    for ant in range(num_ants):
        # 每个蚂蚁逐步构建自己的路径
        current_city = np.random.randint(0, length)
        unvisited_cities = set(range(length)) - {current_city}
        path = [current_city]
        while unvisited_cities:
            next_city = select_next_city(current_city, list(unvisited_cities))
            path.append(next_city)
            unvisited_cities.remove(next_city)
            current_city = next_city
        full_path = Path(path)
        paths.append(full_path)
        # 比较最优
        if full_path.length() < best_path.length():
            best_path = full_path
    # 更新信息素矩阵（循环结束后全局更新）
    update_pheromones(paths)
    # 最佳路径
    best_length_history.append(best_path.length())
    if (generation + 1) % 10 == 0:
        print(f"第{generation + 1}轮: 当前最优方案{best_path.path}的花费为{best_path.length():.5f}")

# 输出最终结果
print(f"最短路径:{best_path.path}")
print(f"最短距离:{best_path.length():.5f}经纬距离")
end_time = time.time()
print(f"运行时间:{end_time - start_time:.5f}s")
evolution(best_length_history)
plot_path(best_path, city_names, df_cities)