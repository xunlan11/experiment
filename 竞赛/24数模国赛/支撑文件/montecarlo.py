import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from math import sqrt, ceil
from scipy.stats import binom 

def critical_k(n, alpha, p=0.1):
    p_result = binom.pmf(np.arange(n+1), n, p)  #计算所有k值的概率
    p_sum = np.cumsum(p_result)                 #计算累积概率
    return np.searchsorted(p_sum, 1-alpha, side='right') - 1

alpha = 0.05        #1-a = 0.95
beta = 0.10         #1-b = 0.90
delta = 0.03        #概率值容忍误差
n_init = 1          #初始样本量
#计算置信区间
z_alpha = stats.norm.ppf(1-alpha/2)
z_beta = stats.norm.ppf(1-beta)
#计算系数
u = (z_alpha + z_beta)**2 / (delta**2)
#相关参数
p_0 = 0.1
sight = 10
MAXITER = 1000
#初始化参数
X = []              #样本序列
X_sum = []          #样本和序列
bar_X = []          #样本均值序列
p_max = []          #最大置信区间
p_min = []          #最小置信区间
n = [n_init]        #样本量序列
k = []              #置信区间上界
count = 0           #迭代次数
#迭代
while 1:
    #如果已经迭代足够次数且n的变化趋于稳定，则跳出循环
    if count > sight and (np.abs(n[-1] - np.mean(n[-sight:])) <= 5) and np.var(n[-sight:]) <= 10:
        print(f"n: {n[-1]}, d: {n[-1]-np.mean(n[-sight:])}, var: {np.var(n[-sight:])}, k: {critical_k(n[-1], alpha)}")
        break
    #更新计数器
    count += 1
    #生成一个二项分布的随机样本
    X.append(np.random.binomial(1, p_0))
    #更新参数
    X_sum.append(sum(X))
    bar_X.append(sum(X)/len(X))
    #print(bar_X[-1])  #用于调试
    #更新样本量
    if count > 1:
        n.append(ceil(max(count, u * p_max[-1] * (1-p_max[-1]))))
    #最大迭代次数限制
    if count > MAXITER:
        print("Max iteration reached")
        break
    #更新临界次品数目
    k.append(critical_k(n[-1], alpha, bar_X[-1]))
    #修正项计算并修正参数
    err = k[-1] - bar_X[-1] * n[-1]
    bar_X[-1] -= 0.75 * err / n[-1]
    bar_X[-1] = min(max(bar_X[-1], 0), 1)  #限制置信区间在[0,1]范围内
    #计算置信区间
    p_max.append(min(bar_X[-1]+z_alpha * sqrt(bar_X[-1]*(1-bar_X[-1])/n[-1]), 1))
    p_min.append(max(bar_X[-1]-z_alpha * sqrt(bar_X[-1]*(1-bar_X[-1])/n[-1]), 0))

print(f"bar_X: {bar_X[-1]}, p_min: {p_min[-1]}, p_max: {p_max[-1]}, n: {n[-1]}, k: {k[-1]}, count: {count}")
#绘图，使用两个比例尺
fig, ax1 = plt.subplots()
ax1.plot(n, label='n', color='green')
ax1.plot(k, label='k', color='orange')
ax1.set_xlabel('sample times')
ax1.set_ylabel('Amount of samples', color='green')
ax1.tick_params(axis='y', labelcolor='black')
ax2 = ax1.twinx()
#置信区间绘制
#ax2.plot(p_max, label='p_max', color='purple', linestyle='--')
#ax2.plot(p_min, label='p_min', color='orange', linestyle='--')
#以如下形式展现
ax2.fill_between(range(len(p_max)), p_min, p_max, color='magenta', alpha=alpha, label='Confidence Interval')
ax2.plot(bar_X, label='bar_X', color='red')
ax2.set_ylabel('Probability', color='red')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(0, 0.5)
plt.axhline(y=p_0, linestyle='--', color='black')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')