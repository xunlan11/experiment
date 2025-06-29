import torch
from torch import nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# 定义模型类
class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        super(MLP, self).__init__()
        # 初始化模型参数
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens1, requires_grad=True) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens1, requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(num_hiddens1, num_hiddens2, requires_grad=True) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(num_hiddens2, requires_grad=True))
        self.W3 = nn.Parameter(torch.randn(num_hiddens2, num_outputs, requires_grad=True) * 0.01)
        self.b3 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
        params = [self.W1,self.b1, self.W2, self.b2, self.W3, self.b3]
        self.num_inputs = num_inputs
    def forward(self, X ):
        X = X.reshape((-1, self.num_inputs))
        H = relu(X @ self.W1 + self.b1)
        C = relu(H @ self.W2 + self.b2)
        return softmax(C @ self.W3 + self.b3)
# 定义激活函数
def relu(X):
    return torch.max(input=X, other=torch.zeros_like(X))
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition
# 定义交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(net, data_iter):
    correct = 0
    total = 0
    net.eval()
    for X, y in data_iter:
        y_hat = net(X)
        correct += (y_hat.argmax(dim=1) == y).type(torch.float).sum().item()
        total += y.shape[0]
    return correct / total
# 定义训练函数
def train(net, train_iter, test_iter, loss, num_epochs, updater):
    train_losses = []
    test_accs = []
    train_accs = []
    # 训练模式
    net.train()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            updater.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y).mean()
            l.backward()
            updater.step()

        train_loss = [loss(net(X), y).mean().item() for X, y in train_iter]
        test_acc = accuracy(net, test_iter)
        train_acc = accuracy(net, train_iter)

        train_accs.append(train_acc)
        train_losses.append(sum(train_loss) / len(train_loss))
        test_accs.append(test_acc)
        print(f'Epoch {epoch + 1}, Loss: {sum(train_loss) / len(train_loss):.3f}, Test Acc: {test_acc:.3f}')
    return train_losses,test_accs,train_accs

# 画出损失函数和准确率随 epoch 变化的曲线
def plot_metrics(train_losses, test_accs, train_accs, num_epochs):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(epochs, train_losses, 'r-', label='Training Loss')
    plt.plot(epochs, test_accs, 'b-', label='Test Accuracy')
    plt.plot(epochs, train_accs, 'g-', label='Train Accuracy')
    plt.yscale('linear')
    plt.xscale('linear')
    plt.xlabel('Epoch')
    plt.xlim([1, num_epochs])
    plt.ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.title('TrainingLoss, Test Accuracy and Train Accuracy vs. Epoch')
    plt.legend()
    plt.show()

def main():
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 数据预处理
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 加载MNIST数据集
    # 加载、划分数据集
    train_data = datasets.MNIST(root="../data", train=True, transform=transforms.ToTensor(), download=False)
    test_data = datasets.MNIST(root="../data", train=False, transform=transforms.ToTensor(), download=False)
    batch_size = 256
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # 定义参数
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    # 实例化模型
    net = MLP(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

    # 训练模型
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(net.parameters(), lr=lr)
    train_losses,test_accs,train_accs = train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    plot_metrics(train_losses,test_accs,train_accs,num_epochs)

if __name__ == '__main__':
    main()