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
num_inputs, num_outputs, num_hiddens1, num_hiddens2, num_hiddens3 = 784, 10, 256, 256, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens1, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens1, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens1, num_hiddens2, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_hiddens2, requires_grad=True))
W3 = nn.Parameter(torch.randn(num_hiddens2, num_hiddens3, requires_grad=True) * 0.01)
b3 = nn.Parameter(torch.zeros(num_hiddens3, requires_grad=True))
W4 = nn.Parameter(torch.randn(num_hiddens3, num_outputs, requires_grad=True) * 0.01)
b4 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2, W3, b3, W4, b4]
def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = relu(X @ W1 + b1)
    H2 = relu(H1 @ W2 + b2)
    H3 = relu(H2 @ W3 + b3)
    out = H3 @ W4 + b4
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