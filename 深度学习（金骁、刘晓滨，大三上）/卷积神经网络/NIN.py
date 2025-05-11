import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import time

# 定义基本卷积块
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),  # 1x1 卷积
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),  # 1x1 卷积
        nn.ReLU()
    )
    return blk

# 定义NIN模型
# nn.MaxPool2d(kernel_size=3, stride=2)*3->nn.MaxPool2d(kernel_size=2, stride=2)*2
class NIN(nn.Module):
    def __init__(self, num_classes=10):
        super(NIN, self).__init__()
        self.net = nn.Sequential(
            nin_block(1, 96, kernel_size=5, stride=1, padding=2),  # 输出: 28x28
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 14x14
            nin_block(96, 256, kernel_size=3, stride=1, padding=1),  # 输出: 14x14
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 7x7
            nin_block(256, 384, kernel_size=3, stride=1, padding=1),  # 输出: 7x7
            nn.Dropout(0.5),
            nn.Conv2d(384, num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    def forward(self, X):
        return self.net(X)

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

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
lr, num_epochs = 0.01, 50
time = train(NIN(), train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
print(f"代码运行时间:{time}秒")