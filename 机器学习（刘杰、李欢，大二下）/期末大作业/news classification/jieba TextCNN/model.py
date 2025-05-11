import torch
import torch.nn as nn
import torch.nn.functional as f


class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()

    def forward(self, x):
        return f.max_pool1d(x, kernel_size=x.size(2)).squeeze(-1)


class TextCNN(nn.Module):
    def __init__(self, num_classes, num_embeddings, embedding_dim=128, kernel_sizes=[3, 4, 5], num_channels=[100, 100, 100]):
        # 调用父类的初始化方法
        super(TextCNN, self).__init__()
        # 类别数量
        self.num_classes = num_classes
        # 嵌入层
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # 卷积层（一维卷积、批量归一化层、ReLU激活函数）
        self.cnn_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(in_channels=embedding_dim if self.embedding is not None else 1,
                                     out_channels=c, kernel_size=k), nn.BatchNorm1d(c), nn.ReLU(inplace=True))
             for c, k in zip(num_channels, kernel_sizes)])
        # 池化层
        self.pool = Pool()
        # 分类层（Dropout层、全连接层）
        self.classify = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(sum(num_channels), self.num_classes))

    # 前向传播函数
    def forward(self, input):
        input = self.embedding(input)
        input = input.permute(0, 2, 1)
        y = [self.pool(layer(input)).squeeze(-1) for layer in self.cnn_layers]
        y = torch.cat(y, dim=1)
        return self.classify(y)
