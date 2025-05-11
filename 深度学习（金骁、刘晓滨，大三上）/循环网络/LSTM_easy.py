import torch
from torch import nn
from d2l import torch as d2l
import time

# 数据与参数
batch_size = 32         # 批量大小
num_steps = 35          # 序列长度
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps) # 数据集
num_epochs = 500        # 训练轮数
lr = 1                  # 学习率
vocab_size = len(vocab) # 词汇表大小
hiddens = 256           # 隐藏层大小
device = d2l.try_gpu()  # 训练设备
num_inputs = vocab_size # 输入大小
# 模型
lstm = nn.LSTM(num_inputs, hiddens)         # lstm层
model = d2l.RNNModel(lstm, len(vocab)).to(device) # 模型实例
# 训练
start_time = time.time()
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
end_time = time.time()
print("运行时间：%.5f秒" % (end_time - start_time))
# 预测
model.eval()
num_predicts = 100      # 预测字符数
start_time = time.time()
phrases = [
    "time traveller",
    "traveller",
    "the time traveller says that",
    "when the time traveller returns to the garden",
    "the time traveller begins learning the language",
    "the time traveller determines that",
    "the time traveller knows he will have to stop",
    "when he wakes up",
    "the time traveller finds himself",
    "the time traveller tells the narrator to wait for him"
]
for phrase in phrases:
    print(d2l.predict_ch8(phrase, num_predicts, model, vocab, device))
end_time = time.time()
print("运行时间：%.5f秒" % (end_time - start_time))