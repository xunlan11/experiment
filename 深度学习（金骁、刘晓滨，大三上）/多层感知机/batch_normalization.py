import torch
import torch.nn as nn
from d2l import torch as d2l
import time

# 划分数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# relu激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 神经网络
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens1, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens1, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens1, num_hiddens2, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_hiddens2, requires_grad=True))
W3 = nn.Parameter(torch.randn(num_hiddens2, num_outputs, requires_grad=True) * 0.01)
b3 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

def batch_norm(X, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.9):
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    mean = X.mean(0)
    var = ((X - mean) ** 2).mean(0)
    running_mean = momentum * running_mean + (1 - momentum) * mean
    running_var = momentum * running_var + (1 - momentum) * var
    X_hat = (X - mean) / torch.sqrt(var + eps)
    Y = gamma * X_hat + beta
    return Y, running_mean, running_var

gamma1 = nn.Parameter(torch.ones(num_hiddens1))
beta1 = nn.Parameter(torch.zeros(num_hiddens1))
gamma2 = nn.Parameter(torch.ones(num_hiddens2))
beta2 = nn.Parameter(torch.zeros(num_hiddens2))
params = [W1, b1, gamma1, beta1, W2, b2, gamma2, beta2, W3, b3]

def net(X):
    X = X.reshape((-1, num_inputs))
    H1, running_mean1, running_var1 = batch_norm(X @ W1 + b1, gamma1, beta1, torch.zeros(num_hiddens1), torch.ones(num_hiddens1))
    H1 = relu(H1)
    H2, running_mean2, running_var2 = batch_norm(H1 @ W2 + b2, gamma2, beta2, torch.zeros(num_hiddens2), torch.ones(num_hiddens2))
    H2 = relu(H2)
    out = H2 @ W3 + b3
    return out

# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 主函数
start_time = time.time()
updater = torch.optim.SGD(params, lr=0.1)
num_epochs= 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.predict_ch3(net, test_iter)
d2l.plt.show()
end_time = time.time()
print(f"代码运行时间:{end_time - start_time}秒")