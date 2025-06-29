import torch
import torch.nn as nn
from d2l import torch as d2l
import time
import matplotlib.pyplot as plt

# 初始化权重
def init_weights(m):
    if isinstance(m, nn.Linear):
        # 高斯分布随机初始化
        # nn.init.normal_(m.weight, mean=0.0, std=0.01)
        # Xavier初始化
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# 预热
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

# 训练函数
def train(net, train_iter, test_iter, num_epochs, loss):
    start_time = time.time()
    net.to(device)
    # 初始化模型参数
    params = [p for p in net.parameters() if p.requires_grad]
    # 优化器
    # SGD
    optimizer = torch.optim.SGD(params, lr=learning_rate)
    # SGD+动量
    # optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
    # SGD+动量+权重衰减
    # optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    # SGD+动量+nesterov
    # optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, nesterov=True)
    # Adam
    # optimizer = torch.optim.Adam(params, lr=learning_rate)
    # Adam+amsgrad
    # optimizer = torch.optim.Adam(params, lr=learning_rate, amsgrad=True)
    # Adamw
    # optimizer = torch.optim.AdamW(params, lr=learning_rate)

    # 学习率策略
    # 固定学习率：循环末尾不更新学习率
    # 指数衰减
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # 分段学习率
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    # 多项式衰减
    # lr_lambda = lambda epoch: (1 - float(epoch) / num_epochs) ** 0.9
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # 线性衰减
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch: 1 - epoch / num_epochs)
    # 余弦衰减
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # 预热
    # warmup_epochs = 5
    # scheduler = warmup_lr_scheduler(optimizer, warmup_epochs, warmup_factor=0.001)

    train_loss_list, train_acc_list, test_acc_list = [], [], []
    for epoch in range(num_epochs):
        # 训练
        net.train()
        train_loss, correct, total_samples = 0.0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y).mean()
            l.backward()
            optimizer.step()
            train_loss += l.item() * y.shape[0]
            _, predicted = torch.max(y_hat.data, 1)
            total_samples += y.size(0)
            correct += (predicted == y).sum().item()
        avg_train_loss = train_loss / total_samples
        train_acc = correct / total_samples
        # 测试
        net.eval()
        with torch.no_grad():
            test_correct, test_total = 0, 0
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                _, predicted = torch.max(y_hat.data, 1)
                test_total += y.size(0)
                test_correct += (predicted == y).sum().item()
            test_acc = test_correct / test_total
        # 学习率变化
        # scheduler.step()
        # 结果
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"Epoch[{epoch + 1}/{num_epochs}],Loss:{avg_train_loss:.4f},Train Acc:{train_acc:.4f},Test Acc:{test_acc:.4f}")
    best_test_acc = max(test_acc_list)
    print(f"最优验证正确率:{best_test_acc:.4f}")
    # 训练耗时
    end_time = time.time()
    training_time = end_time - start_time
    # 绘图
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    epochs = range(1, num_epochs + 1)
    ax1.plot(epochs, train_loss_list, 'bo-', label="Train Loss")
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_xlim(1, num_epochs)
    ax1.set_ylim(0, max(train_loss_list) * 1.1)
    ax1.legend(loc='upper left')
    ax2.plot(epochs, train_acc_list, 'r+-', label="Train Acc")
    ax2.plot(epochs, test_acc_list, 'g+-', label="Test Acc")
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right')
    plt.title('Training Progress')
    plt.show()
    return training_time

# 主函数
if __name__ == "__main__":
    # 参数
    num_epochs = 10
    learning_rate = 0.001 # (SGD)为0.1，Adam为0.001
    batch_size = 256
    # 数据
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        nn.Linear(num_hiddens1, num_hiddens2),
        nn.ReLU(),
        nn.Linear(num_hiddens2, num_outputs)
    )
    # net.apply(init_weights)
    # 损失函数
    loss = nn.CrossEntropyLoss(reduction='mean')
    # 训练
    training_time = train(net, train_iter, test_iter, num_epochs, loss)
    print(f"代码运行时间:{training_time}秒")