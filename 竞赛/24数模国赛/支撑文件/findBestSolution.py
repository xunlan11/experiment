import numpy as np  
import matplotlib.pyplot as plt  
import scipy.stats as stats
from math import sqrt, ceil  
from scipy.stats import binom
  
#计算给定置信水平下，二项分布需要达到的临界k值
def critical_k(n, alpha, p=0.1):  
    p_result = binom.pmf(np.arange(n+1), n, p)  #计算从0到n的所有k值的二项分布概率
    p_sum = np.cumsum(p_result)                 #计算累积概率
    return np.searchsorted(p_sum, 1-alpha, side='right') - 1  #找到累积概率大于或等于1-alpha的最小k值
  
#找到最优样本量n的迭代过程
def findBestSolution(alpha, beta, delta, n_init, MAXITER, p=0.1):  
    #计算正态分布的分位数
    z_alpha = stats.norm.ppf(1-alpha/2)  
    z_beta = stats.norm.ppf(1-beta)
    #计算样本量n的公式中的u
    u = (z_alpha + z_beta)**2 / (delta**2)
    #初始化存储结果的列表
    ave = []            #平均迭代次数
    ave_n = []          #平均样本量
    result = []         #最优样本量的列表
    result_1 = []       #最优样本量对应的迭代次数的列表
    #迭代过程
    for i in range(MAXITER):
        #初始化变量
        X = []          #每次迭代的样本均值
        X_sum = []      #每次迭代的样本总和
        bar_X = []      #每次迭代的样本均值
        p_max = []      #每次迭代的样本均值+z_alpha*sqrt(bar_X*(1-bar_X)/n)的最大值
        p_min = []      #每次迭代的样本均值-z_alpha*sqrt(bar_X*(1-bar_X)/n)的最小值
        n = [n_init]    #每次迭代的样本量
        k = []          #每次迭代的临界k值
        count = 0       #迭代次数
        while 1:  
            #如果已经迭代足够次数且n的变化趋于稳定，则跳出循环
            if count > 20 and (n[-1] - np.mean(n[-20:]) <= 10) and np.var(n[-20:]) < 10:  
                result.append(n[-1])  
                result_1.append(count)  
                break
            #增加迭代次数
            count += 1
            #生成一个二项分布的随机样本
            #if random.random() < p:
            #    X.append(1)
            #else:
            #    X.append(0)
            X.append(np.random.binomial(1, 0.1))  
            X_sum.append(sum(X))  
            bar_X.append(sum(X)/len(X))
            #更新n的值
            if count > 1:  
                n.append(ceil(max(count, u * p_max[-1] * (1-p_max[-1]))))
            #更新p_max和p_min
            p_max.append(min(bar_X[-1]+z_alpha * sqrt(bar_X[-1]*(1-bar_X[-1])/n[-1]), 1))  
            p_min.append(max(bar_X[-1]-z_alpha * sqrt(bar_X[-1]*(1-bar_X[-1])/n[-1]), 0))
            #计算当前n下的临界k值
            k.append(critical_k(n[-1], alpha))
        #计算平均迭代次数和平均样本量
        ave.append(np.mean(result_1))  
        ave_n.append(np.mean(result))
    #打印结果和最后一次迭代的详细信息
    print(np.mean(result))  
    print(f"bar_X: {bar_X[-1]}, p_max: {p_max[-1]}, n: {n[-1]}, k: {k[-1]}, count: {count}")
    #绘制n和count的变化图
    plt.plot(n, label='n')  
    plt.plot(range(count), label='count')
    plt.legend()  
    plt.show()
  
#调用函数，测试不同的参数
#findBestSolution(alpha=0.05, beta=0.10, delta=0.05, n_init=1, MAXITER=500)
findBestSolution(alpha=0.10, beta=0.10, delta=0.05, n_init=1, MAXITER=10, p=1-0.1)