import numpy as np
import matplotlib. pyplot as plt
from scipy. stats import beta
import seaborn

seaborn.set()
params =[0.25,  1,10]
x = np.linspace(0, 1, 100)
plt.plot(x, beta(0.25, 0.25). pdf(x), color='b', label='$\\alpha=0.25, \\beta=0.25$') 
plt.fill_between(x,0, beta(0.25,0.25).pdf(x), color='b', alpha=0.25)
plt.plot(x, beta(1, 1). pdf(x), color='g', label='$\\alpha=1, \\beta=1$')
plt.fill_between(x, 0, beta(1, 1). pdf(x), color='g', alpha=0.25)
plt.plot(x, beta(10,10).pdf(x),color='r',label='$\\alpha=10,\\beta=10$')
plt.fill_between(x,0, beta(10, 10).pdf(x), color='r', alpha=0.25)
plt.gca().axes. set_ylim(0,10)
plt.gca().axes. set_xlabel('theta')
plt.gca().axes. set_ylabel('p(theta)')
plt.legend()
plt.show()