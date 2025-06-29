import numpy as np
from math import radians, sin, cos, sqrt, asin
import pandas as pd

# 定义城市坐标
cities = {
    "北京": (116.46, 39.92),
    "天津": (117.2, 39.13),
    "上海": (121.48, 31.22),
    "重庆": (106.54, 29.59),
    "拉萨": (91.11, 29.97),
    "乌鲁木齐": (87.68, 43.77),
    "银川": (106.27, 38.47),
    "呼和浩特": (111.65, 40.82),
    "南宁": (108.33, 22.84),
    "哈尔滨": (126.63, 45.75),
    "长春": (125.35, 43.88),
    "沈阳": (123.38, 41.8),
    "石家庄": (114.48, 38.03),
    "太原": (112.53, 37.87),
    "西宁": (101.74, 36.56),
    "济南": (117, 36.65),
    "郑州": (113.65, 34.76),
    "南京": (118.78, 32.04),
    "合肥": (117.27, 31.86),
    "杭州": (120.19, 30.26),
    "福州": (119.3, 26.08),
    "南昌": (115.89, 28.68),
    "长沙": (113, 28.21),
    "武汉": (114.31, 30.52),
    "广州": (113.23, 23.16),
    "台北": (121.5, 25.05),
    "海口": (110.35, 20.02),
    "兰州": (103.73, 36.03),
    "西安": (108.95, 34.27),
    "成都": (104.06, 30.67),
    "贵阳": (106.71, 26.57),
    "昆明": (102.73, 25.04),
    "香港": (114.1, 22.2),
    "澳门": (113.33, 22.13)
}

def haversine(lat1, lon1, lat2, lon2):
    """
    计算两个地理坐标点之间的大圆距离。
    lat:纬度
    lon:经度
    """
    R = 6371  # 地球半径，单位为km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat/2)**2 + cos(lat1) * cos(lat2) * sin(dLon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# 距离矩阵
num_cities = len(cities)
distance_matrix = np.zeros((num_cities, num_cities))
for i, city1 in enumerate(cities):
    for j, city2 in enumerate(cities):
        if i != j:
            lat1, lon1 = cities[city1]
            lat2, lon2 = cities[city2]
            distance_matrix[i][j] = haversine(lat1, lon1, lat2, lon2)

# 保存
city_names = list(cities.keys())
df = pd.DataFrame(distance_matrix, index=city_names, columns=city_names)
output_file = '城市距离矩阵.xlsx'
df.to_excel(output_file)