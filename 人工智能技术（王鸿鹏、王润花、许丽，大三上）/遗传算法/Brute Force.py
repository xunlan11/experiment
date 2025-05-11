import math
import pandas as pd
import itertools
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path_distances = '城市经纬距离矩阵.xlsx'
df_distances = pd.read_excel(file_path_distances, index_col=0)
distance_matrix = df_distances.iloc[:10, :10].values
city_names = df_distances.index[:10]
file_path_cities = '城市经纬度.xlsx'
df_cities = pd.read_excel(file_path_cities, index_col=0)

# 暴力算法求最短路径
def tsp_brute_force(distance_matrix):
    n = len(distance_matrix)
    total_permutations = itertools.permutations(range(n))
    min_distance = float('inf')
    best_path = None
    for path in tqdm(total_permutations, total=math.factorial(n), desc="计算最短路径"):
        distance = 0
        for i in range(n - 1):
            distance += distance_matrix[path[i]][path[i + 1]]
        distance += distance_matrix[path[-1]][path[0]]
        if distance < min_distance:
            min_distance = distance
            best_path = path
    return best_path, min_distance

# 绘制路径
def plot_path(best_path, city_names, distance_matrix):
    # 模拟地图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    # 绘制城市点并标注城市名
    path = [city_names[i] for i in best_path]
    lats = distance_matrix.loc[path, 'Latitude'].values
    lons = distance_matrix.loc[path, 'Longitude'].values
    ax.scatter(lons, lats, 60, marker='o', color='k', zorder=5, transform=ccrs.PlateCarree(), label='城市')
    for i, name in enumerate(path):
        ax.text(lons[i], lats[i], name, fontsize=10, ha='right', va='bottom', transform=ccrs.PlateCarree())
    # 绘制路径（回到起点）
    ax.plot(lons, lats, 'r-', linewidth=2, label='路径', transform=ccrs.PlateCarree())
    ax.plot([lons[-1], lons[0]], [lats[-1], lats[0]], 'r-', linewidth=2, transform=ccrs.PlateCarree())
    plt.show()

start_time = time.time()
best_path, min_distance = tsp_brute_force(distance_matrix)
print(f"最短路径: {[city_names[i] for i in best_path]}")
print(f"最短距离: {min_distance:.5f}经纬距离")
end_time = time.time()
print(f"运行时间: {end_time - start_time:.5f}s")
plot_path(best_path, city_names, df_cities)