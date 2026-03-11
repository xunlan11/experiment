import numpy as np
import matplotlib. pyplot as plt
from scipy. stats import beta
import seaborn

seaborn.set()
theta_real = 0.62
n_array = [5, 10, 20, 100, 500, 1000]
y_array = [2, 4, 11, 60, 306, 614]
beta_params = [(0.25, 0.25), (1, 1), (10, 10)]
x = np.linspace(0, 1, 100)
fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
for i in range(2):
    for j in range(3):
        n = n_array[3 * i + j]
        y = y_array[3 * i + j]
        for (a_prior, b_prior), c in zip(beta_params, ('b', 'r', 'g')):
            a_post = a_prior + y
            b_post = b_prior + n - y
            p_theta_given_y = beta.pdf(x, a_post, b_post)
            ax[i, j].plot(x, p_theta_given_y, c)
            ax[i, j].fill_between(x, 0, p_theta_given_y, color=c, alpha=0.25)
        ax[i, j].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax[i, j].set_title('n={},y={}'.format(n, y))
ax[0, 0].set_ylabel('$p(\\theta|y)$')
ax[1, 0].set_ylabel('$p(\\theta|y)$')
ax[1, 1].set_xlabel('$\\theta$')
plt.show()