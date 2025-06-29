import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
import time

# 定义卷积块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# VGGNet
# conv_arch：((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))->((1, 1, 32), (1, 32, 64))
class VGGNet(nn.Module):
    def __init__(self, conv_arch=((1, 1, 32), (1, 32, 64)), num_classes=10):
        super(VGGNet, self).__init__()
        self.conv_arch = conv_arch
        self.features = self.make_layers(conv_arch)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def make_layers(self, conv_arch):
        layers = []
        in_channels = 1
        for (num_convs, in_channel, out_channel) in conv_arch:
            layers.append(vgg_block(num_convs, in_channel, out_channel))
            in_channels = out_channel
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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

# 主函数
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
lr, num_epochs = 0.1, 20
time = train(VGGNet(), train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
print(f"代码运行时间:{time}秒")