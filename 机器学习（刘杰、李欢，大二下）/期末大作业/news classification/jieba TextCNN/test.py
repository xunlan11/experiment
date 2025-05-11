import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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


# 导入测试集和相关文件
test_df = pd.read_csv('test.csv')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)
test_dataset = TextDataset(test_df)
test_loader = DataLoader(test_dataset, 64)
model = TextCNN(num_classes=len(label_encoder.classes_), num_embeddings=2002)
model.load_state_dict(torch.load('model.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# 测试
model.eval()
results = []
labels = []
with torch.no_grad():
    for texts, true_labels in test_loader:
        texts = texts.to(device)
        outputs = model(texts)
        _, predicted = torch.max(outputs, 1)
        predicted_classes = predicted.cpu().numpy()
        results.extend(predicted_classes)
        true_labels_cpu = true_labels.cpu().numpy()
        labels.extend(true_labels_cpu)
# 测试正确率
results = np.array(results)
labels = np.array(labels)
accuracy = np.mean(results == labels)
print(f"Test Accuracy: {accuracy:.4f}")
