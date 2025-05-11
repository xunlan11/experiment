import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import TextCNN


# 加载、处理数据集
class TextDataset(Dataset):
    def __init__(self, df):
        self.labels = df['category'].values
        self.texts = df['indices'].apply(lambda x: np.array(list(map(int, x.replace('\n', '').split(','))))).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# 训练模型函数
def train(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    model.to(device)
    model.train()
    best_val_acc = 0
    for epoch in range(epochs):
        running_loss = 0.0
        train_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for inputs, labels in train_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            # 清除之前的梯度
            optimizer.zero_grad()
            # 前向传播，得到预测输出
            outputs = model(inputs)
            # 损失
            loss = criterion(outputs, labels)
            # 反向传播损失，计算梯度并累计
            loss.backward()
            running_loss += loss.item()
            # 使用优化器更新模型权重
            optimizer.step()
            train_tqdm.set_postfix(loss=running_loss / len(train_loader))
        # 验证
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss:{running_loss / len(train_loader)}, Val Loss: {val_loss}, Val Acc: {val_acc}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model.pth')


# 评估函数
def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return total_loss / len(data_loader), correct / len(data_loader.dataset)


train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
# 加载、包装数据
train_dataset = TextDataset(train_df)
val_dataset = TextDataset(val_df)
train_loader = DataLoader(train_dataset, 64, shuffle=True)
val_loader = DataLoader(val_dataset, 64)
# 标签种类，词汇表长度（+2）
model = TextCNN(num_classes=len(train_df['category'].unique()), num_embeddings=2002)
# 损失函数为交叉熵损失
criterion = nn.CrossEntropyLoss()
# Adam优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练轮次
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train(model, criterion, optimizer, train_loader, val_loader, epochs, device)
