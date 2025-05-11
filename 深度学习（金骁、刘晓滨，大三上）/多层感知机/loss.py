import torch
import torch.nn as nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import time

# 划分数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# relu激活函数
def relu(X):
    return torch.maximum(X, torch.tensor(0.0))

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

# 均方误差损失函数
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
    def forward(self, output, target, params=None):
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_outputs).float()
        mse = torch.square(output - target_one_hot)
        return mse.mean()

# 平均绝对误差
class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
    def forward(self, output, target, params=None):
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_outputs).float()
        mae = torch.abs(output - target_one_hot)
        return mae.mean()

# Huber损失函数
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
    def forward(self, output, target, params=None):
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_outputs).float()
        mask = (output - target_one_hot).abs() < self.delta
        loss = torch.zeros_like(output)
        loss[mask] = 0.5 * (output[mask] - target_one_hot[mask]) ** 2
        loss[~mask] = self.delta * (output[~mask] - target_one_hot[~mask]).abs() - 0.5 * self.delta ** 2
        return loss.mean()

# 焦点损失函数
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, output, target, params=None):
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_outputs).float()
        softmax_output = torch.softmax(output, dim=1)
        probs = (softmax_output * target_one_hot).sum(dim=1)
        loss = -self.alpha * torch.pow(1.0 - probs, self.gamma) * torch.log(probs)
        return loss.mean()

# 带l2正则化的交叉熵损失函数
class CrossEntropy_L2(nn.Module):
    def __init__(self, weight_decay=0.01):
        super(CrossEntropy_L2, self).__init__()
        self.weight_decay = weight_decay
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    def forward(self, output, target, params):
        ce_loss = self.cross_entropy_loss(output, target)
        # L2正则化项
        l2_reg = torch.tensor(0., device=params[0].device)
        for param in params:
            l2_reg += torch.norm(param, p=2) ** 2
        total_loss = ce_loss + self.weight_decay * l2_reg
        return total_loss

# loss = MSELoss()
# loss = MAELoss()
# loss = HuberLoss()
# loss = FocalLoss()
loss = CrossEntropy_L2()

# 正确率评估函数
def accuracy(data_iter, net):
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            cmp = (y_hat.argmax(axis=1) == y).type(y.dtype)
            metric.add(float(cmp.sum()), cmp.numel())
    return metric[0] / metric[1]
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# 训练函数
def train_and_record(train_iter, test_iter, net, loss, trainer, num_epochs, params):
    train_l_history = []
    train_acc_history = []
    test_acc_history = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y, params)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            train_l_sum += l.cpu().item()
            n += y.shape[0]
        train_l_history.append(train_l_sum / n)
        train_acc_sum = accuracy(train_iter, net)
        train_acc_history.append(train_acc_sum)
        test_acc = accuracy(test_iter, net)
        test_acc_history.append(test_acc)
        print(f'epoch {epoch + 1}, test acc {test_acc:.3f}')
    return train_l_history, train_acc_history, test_acc_history

# 主函数
if __name__ == '__main__':
    trainer = torch.optim.SGD(params, lr=0.1)
    num_epochs = 10
    start_time = time.time()
    # 训练
    train_l_history, train_acc_history, test_acc_history = train_and_record(train_iter, test_iter, net, loss, trainer, num_epochs, params)
    # 绘制曲线
    plt.figure(figsize=(10, 5))
    epochs = list(range(1, num_epochs + 1))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_l_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_history, label='Training Accuracy')
    plt.plot(epochs, test_acc_history, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.001)
    # 预测
    d2l.predict_ch3(net, test_iter)
    d2l.plt.show()
    end_time = time.time()
    print(f"代码运行时间:{end_time - start_time}秒")