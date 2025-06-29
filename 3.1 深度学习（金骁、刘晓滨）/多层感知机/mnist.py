import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt
import time

# 加载、划分数据集
train_data = datasets.MNIST(root="../data", train=True, transform=transforms.ToTensor(), download=False)
test_data = datasets.MNIST(root="../data", train=False, transform=transforms.ToTensor(), download=False)
batch_size = 256
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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
params = [W1, b1, W2, b2, W3, b3]
def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = relu(X @ W1 + b1)
    H2 = relu(H1 @ W2 + b2)
    out = H2 @ W3 + b3
    return out

# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 主函数
start_time = time.time()
updater = torch.optim.SGD(params, lr=0.1)
num_epochs= 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.plt.show()
end_time = time.time()
print(f"代码运行时间:{end_time - start_time}秒")

# 预测函数
def predict(net, test_iter, n=6):
    for X, y in test_iter:
        break
    preds = net(X).argmax(dim=1)
    trues = [str(y_i.item()) for y_i in y[:n]]
    preds = [str(pred_i.item()) for pred_i in preds[:n]]
    titles = ['true: ' + true + '\npred: ' + pred for true, pred in zip(trues, preds)]
    plt.figure(figsize=(n * 1.2, 2.4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i].reshape(28, 28), cmap=plt.cm.binary)
        plt.xlabel(titles[i])
    plt.show()

predict(net, test_iter)