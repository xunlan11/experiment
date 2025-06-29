import torch
from torch import nn
import torch.nn.functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt
import time

# Inception块
class Inception(nn.Module):
    def __init__(self, in_channels, ch1, ch2_in, ch2_out, ch3_in, ch3_out, ch4_out, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, ch1, kernel_size=1)
        # 1x1卷积层+3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, ch2_in, kernel_size=1)
        self.p2_2 = nn.Conv2d(ch2_in, ch2_out, kernel_size=3, padding=1)
        # 1x1卷积层+5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, ch3_in, kernel_size=1)
        self.p3_2 = nn.Conv2d(ch3_in, ch3_out, kernel_size=5, padding=2)
        # 3x3最大池化层+1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, ch4_out, kernel_size=1)
    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(F.relu(self.p2_1(x)))
        p3 = self.p3_2(F.relu(self.p3_1(x)))
        p4 = self.p4_2(self.p4_1(x))
        return torch.cat((p1, p2, p3, p4), dim=1)

# GoogleNet
net = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    nn.Conv2d(64, 64, kernel_size=1),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    Inception(192, 64, 96, 128, 16, 32, 32),
    Inception(256, 128, 128, 192, 32, 96, 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    Inception(480, 192, 96, 208, 16, 48, 64),
    Inception(512, 160, 112, 224, 24, 64, 64),
    Inception(512, 128, 128, 256, 24, 64, 64),
    Inception(512, 112, 144, 288, 32, 64, 64),
    Inception(528, 256, 160, 320, 32, 128, 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    Inception(832, 256, 160, 320, 32, 128, 128),
    Inception(832, 384, 192, 384, 48, 128, 128),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(1024, 10))

# 评估
def evaluate(net, data_iter, device):
    net.eval()
    metric = d2l.Accumulator(2)    # 正确预测的数量，总预测的数量
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 训练
def train(net, train_iter, test_iter, num_epochs, lr, device):
    start_time = time.time()
    # 初始化权重
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    # 训练初始化
    net.apply(init_weights)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 优化器：随机梯度下降
    loss = nn.CrossEntropyLoss()                          # 损失函数：交叉熵损失
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    # 训练
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)    # 训练损失之和，训练准确率之和，样本数
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            # 清理不再需要的张量
            torch.cuda.empty_cache()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate(net, test_iter, device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')
    end_time = time.time()
    plt.show()
    return end_time - start_time

# 主函数
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
lr, num_epochs = 0.01, 20
time = train(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
print(f"代码运行时间:{time}秒")