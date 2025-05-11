import re
import jieba
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# 分词与筛选
def process(in_text):
    source_list = list(jieba.cut(in_text, cut_all=False))
    # 设定虚词为停用词
    stopwords = {'的', '了', '在', '是', '我', '有', '和', '也', '就', '都', '不', '你', '他', '她', '它', '我们',
                 '你们', '他们', '自己', '这', '那', '上', '下', '要', '说', '着', '么', '吧', '会', '过', '去',
                 '还', '很', '太', '都', '又', '就', '也', '还', '了', '啊', '吗', '吧', '呢', '着', '啦', '的',
                 '得', '地', '个', '把', '对', '等', '但', '而', '和', '还', '或', '可', '却', '如果', '虽然',
                 '但是', '因为', '所以', '因此', '于是', '因此', '这样', '那样', '比如', '例如', '同时', '然后', '然后',
                 '还有', '另外', '此外', '同样', '既然', '除了', '只有', '只要', '不仅', '不管'}
    # 去掉虚词、数字，只保留剩下的汉字
    filtered_list = [word for word in source_list if
                     word not in stopwords and not word.isdigit() and not re.match(r'[^\u4e00-\u9fa5]', word)]
    return filtered_list


# 构建词汇表
def build_vocab(tokens):
    # 统计词汇出现频次
    word_counts = Counter(tokens)
    # 选取频次最高的部分词汇
    vocab_size = 2000
    vocab = [word for word, count in word_counts.most_common(vocab_size)]
    # 在词汇表列表末尾添加两个特殊标记
    vocab.append('<PAD>')
    vocab.append('<UNK>')
    # 返回词汇表和索引
    return vocab, {word: idx for idx, word in enumerate(vocab)}


# 填充到相同长度
def build_indices(tokens, word_to_idx, max_seq_length):
    indices = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
    indices += [word_to_idx['<PAD>']] * (max_seq_length - len(indices))
    indices = indices[:max_seq_length]
    return np.array(indices)


# 将原文件转换为xlsx格式后导入，并标注列名
df = pd.read_excel('../cnews.train.xlsx', header=None, names=['category', 'content'], engine='openpyxl')
# 应用上述函数处理文本列，最后获得字符串
df['tokens'] = df['content'].apply(process)
all_tokens = [token for news_tokens in df['tokens'] for token in news_tokens]
vocab, vocabulary = build_vocab(all_tokens)
df['indices'] = df['tokens'].apply(lambda tokens: build_indices(tokens, vocabulary, 200))
df['indices'] = df['indices'].apply(lambda x: ','.join(map(str, x)))
# 类别列转换为数值编码
label_encoder = LabelEncoder()
df['category'] = label_encoder.fit_transform(df['category'])
# 保留有用列，划分训练集和测试集
df = df[['category', 'indices']]
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
# 保存数据
np.save('classes.npy', label_encoder.classes_)
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
test_df.to_csv('test.csv', index=False)
