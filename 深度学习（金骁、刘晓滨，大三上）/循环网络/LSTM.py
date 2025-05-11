import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import time

# 获取模型参数
def get_params(vocab_size, hiddens, device):
    num_inputs = num_outputs = vocab_size # 输入和输出与词汇表同维
    def three():
        return (torch.randn(size=(num_inputs, hiddens), device=device) * 0.01,
                torch.randn(size=(hiddens, hiddens), device=device) * 0.01,
                torch.zeros(hiddens, device=device))
    W_xi, W_hi, b_i = three() # 输入门
    W_xf, W_hf, b_f = three() # 遗忘门
    W_xo, W_ho, b_o = three() # 输出门
    W_xc, W_hc, b_c = three() # 候选记忆元
    # 输出层
    W_hq = torch.randn(size=(hiddens, num_outputs), device=device) * 0.01
    b_q = torch.zeros(num_outputs, device=device)
    # 添加梯度
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 初始化状态：为一个batch的数据初始化隐藏状态和细胞状态
def init_state(batch_size, hiddens, device):
    return torch.zeros((batch_size, hiddens), device=device), torch.zeros((batch_size, hiddens), device=device)

# LSTM前向传播
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state # 隐藏状态和细胞状态
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)    # 输入门
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)    # 遗忘门
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)    # 输出门
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c) # 候选记忆元
        C = F * C + I * C_tilda # 更新细胞状态
        H = O * torch.tanh(C)   # 更新隐藏状态
        Y = (H @ W_hq) + b_q    # 输出
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)

# RNN类
class RNN:
    def __init__(self, vocab_size, hiddens, device, params, init_state, forward_fn):
        self.vocab_size = vocab_size
        self.hiddens = hiddens
        self.params = params(vocab_size, hiddens, device)
        self.init_state = init_state # 初始化状态函数
        self.forward_fn = forward_fn # 前向传播函数
    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32) # 输入独热编码
        return self.forward_fn(X, state, self.params) # 前向传播
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.hiddens, device) # 初始化状态

# 梯度裁剪，防止梯度爆炸
def grad_clipping(net, theta):
    params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params)) # L2范数
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm # 缩放梯度

# 训练
def train(net, train_iter, vocab, lr, epochs, device):
    loss = nn.CrossEntropyLoss()              # 交叉熵损失函数
    updater = torch.optim.SGD(net.params, lr) # SGD优化器
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', legend=['train'], xlim=[10, epochs])
    for epoch in range(epochs):
        state = None
        timer = d2l.Timer()
        metric = d2l.Accumulator(2) # 累计损失和样本数量
        for X, Y in train_iter:
            if state is None: # 初始化状态
                state = net.begin_state(batch_size=X.shape[0], device=device)
            else: # 断开历史依赖
                if isinstance(state, tuple): # 如果state是元组（LSTM的状态）
                    for s in state:
                        s.detach_() # 创建新张量，切断与旧梯度的联系
                else:
                    state.detach_()
            y = Y.T.reshape(-1) # 展平
            X, y = X.to(device), y.to(device)
            y_hat, state = net(X, state) # 前向传播
            l = loss(y_hat, y.long()).mean() # 计算平均损失
            updater.zero_grad() # 清空梯度
            l.backward() # 反向传播
            grad_clipping(net, 1) # 梯度裁剪
            updater.step() # 参数更新
            metric.add(l * d2l.size(y), d2l.size(y)) # 累计损失和样本数量
        ppl = math.exp(metric[0] / metric[1]) # 计算困惑度
        speed = metric[1] / timer.stop() # 计算处理速度
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller', 50, net, vocab, device))
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')

# 预测
def predict(prefix, len_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device) # 初始化
    outputs = [vocab[prefix[0]]] # 初始化输出列表
    # 获取输入张量表示，调整形状以适应模型输入要求
    get_input = lambda: d2l.reshape(d2l.tensor([outputs[-1]], device=device), (1, 1))
    # 更新状态并记录字符索引
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # 生成指定长度的新字符
    for _ in range(len_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    # 将索引转换回字符，连接成完整字符串
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# 设置数据、参数与模型
batch_size = 32        # 批量大小
num_steps = 35         # 序列长度
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps) # 数据集
num_epochs = 500       # 训练轮数
lr = 1                 # 学习率
hiddens = 256          # 隐藏层大小
device = d2l.try_gpu() # 训练设备
model = RNN(len(vocab), hiddens, device, get_params, init_state, lstm) # 模型实例
# 训练
start_time = time.time()
train(model, train_iter, vocab, lr, num_epochs, device)
end_time = time.time()
print("运行时间：%.5f秒" % (end_time - start_time))
# 预测
num_predicts = 100 # 预测字符数
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
    print(predict(phrase, num_predicts, model, vocab, device))
end_time = time.time()
print("运行时间：%.5f秒" % (end_time - start_time))