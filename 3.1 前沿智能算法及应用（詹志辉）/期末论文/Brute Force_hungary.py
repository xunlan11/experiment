import itertools
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

# 数据
df = pd.read_excel('data.xlsx')
df = df.values
df = df[:, 1:11].astype(np.int64)
length = 5         # 个体基因数量——配送点数目

# 转化最优解方案
def solution(combination):
    i = np.arange(length)
    r = (3 * i + np.array(combination) - 1).repeat(2)
    cost_matrix = df[r, :]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for j in range(length):
        print(f"P{j + 1}.L{combination[j]}到S{col_ind[2 * j] + 1},S{col_ind[2 * j + 1] + 1}")

# 主函数
start_time = time.time()
location_combinations = list(itertools.product(range(1, 4), repeat=5))
solutions = {}
for j, combination in enumerate(location_combinations):
    i = np.arange(length)
    r = (3 * i + combination - 1).repeat(2)
    cost_matrix = df[r, :]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    fitness_value = cost_matrix[row_ind, col_ind].sum()
    solutions[combination] = fitness_value
best_solution = min(solutions, key=solutions.get)
best_value = solutions[best_solution]
print("最优方案", best_solution, "的花费为", best_value)
solution(best_solution)
end_time = time.time()
print("运行时间：%.5f秒" % (end_time - start_time))