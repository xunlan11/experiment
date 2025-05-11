import torch
import torch.nn as nn
from torch import optim

# 神经网络
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2, 5)
        self.output = nn.Linear(5, 1)
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

# 训练数据
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 训练配置
model = XORNet()
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 随机梯度下降优化器

# 训练
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 交互
def get_user_input():
    x1 = int(input("请输入第一个数字（0或1）："))
    x2 = int(input("请输入第二个数字（0或1）："))
    input_tensor = torch.tensor([[x1, x2]], dtype=torch.float32)
    return input_tensor

# 预测
def predict_xor(model, input_tensor):
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
        rounded_prediction = torch.round(prediction).int().item()
    return rounded_prediction

# 主函数
input_tensor = get_user_input()
prediction = predict_xor(model, input_tensor)
print(f"预测结果：{prediction}")