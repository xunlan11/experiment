import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 导入数据集并划分训练集和验证集
t_v = pd.read_csv('train.csv')
x_col = ['V' + str(i) for i in range(1, 31)]
X = t_v[x_col]
X = X.astype('float32')
y = t_v['Label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(30,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# 找到预测阈值（高于阈值的为1，低于的为0）
predictions_p = model.predict(X)
t_v['Label_p'] = predictions_p
threshold = 0
threshold_best = 0
f1_best = 0
while threshold < 1:
    predictions_p_binary = np.where(predictions_p > threshold, 1, 0)
    f1 = f1_score(y, predictions_p_binary, average='macro')
    if f1 > f1_best:
        threshold_best = threshold
        f1_best = f1
    threshold += 0.0001
predictions_p_binary = np.where(predictions_p > threshold_best, 1, 0)
t_v['Label_b'] = predictions_p_binary
t_v.to_csv('train_with_labels.csv', index=False)
print(f"f1_best={f1_best}")


# 测试
t = pd.read_csv('pred.csv')
X_test = t[x_col]
X_test = X_test.astype('float32')
predictions = model.predict(X_test)
predictions_binary = np.where(predictions > threshold_best, 1, 0)
t['Label'] = predictions_binary
t.to_csv('pred_with_labels.csv', index=False)


# 绘制损失和验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()