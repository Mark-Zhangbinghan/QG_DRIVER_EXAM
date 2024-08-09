import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. 读取数据
file_path = './广州市交通指数.txt'
data = pd.read_csv(file_path, delimiter='\t', encoding='utf-8')

# 2. 数据预处理
data['交通指数'] = data['交通指数,拥堵级别,统计时间'].apply(lambda x: float(x.split(',')[0]))
data['统计时间'] = data['交通指数,拥堵级别,统计时间'].apply(
    lambda x: pd.to_datetime(x.split(',')[2], format='%Y年%m月'))

# 按时间排序
data = data.sort_values(by='统计时间').reset_index(drop=True)

# 将数据划分为训练集和验证集
train_data = data[data['统计时间'] < '2024-05']
val_data = data[data['统计时间'] == '2024-05']

# 提取特征和标签
X_train = train_data['交通指数'].values.reshape(-1, 1)
y_train = X_train.copy()  # 因为我们是用前一时间步预测下一时间步，直接复制即可
X_val = val_data['交通指数'].values.reshape(-1, 1)
y_val = X_val.copy()


# 改进的数据归一化函数，防止除以零的情况
def min_max_scaler(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)  # 如果最大值和最小值相等，返回全零数组
    return (data - min_val) / (max_val - min_val)


# 应用归一化
X_train_scaled = min_max_scaler(X_train)
X_val_scaled = min_max_scaler(X_val)

# 将数据转换为LSTM的输入格式 (samples, time_steps, features)
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, 1)
X_val_lstm = X_val_scaled.reshape(X_val_scaled.shape[0], 1, 1)

# 转换为PyTorch张量
X_train_tensor = torch.Tensor(X_train_lstm)
y_train_tensor = torch.Tensor(y_train)
X_val_tensor = torch.Tensor(X_val_lstm)
y_val_tensor = torch.Tensor(y_val)


# 3. 定义LSTM模型
class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        c_0 = torch.zeros(2, x.size(0), 50).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


# 4. 初始化模型、损失函数和优化器
model = TrafficLSTM()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. 训练模型
num_epochs = 100
train_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # 打印损失值
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 6. 定义标签生成函数*
def get_label(traffic_index):
    if 0 <= traffic_index < 2:
        return "畅通"
    elif 2 <= traffic_index < 4:
        return "基本畅通"
    elif 4 <= traffic_index < 6:
        return "轻度拥堵"
    elif 6 <= traffic_index < 8:
        return "中度拥堵"
    elif 8 <= traffic_index <= 10:
        return "严重拥堵"
    else:
        return "无效值"


# 7. 模型评估
model.eval()
with torch.no_grad():
    predictions = model(X_val_tensor)
    val_loss = criterion(predictions, y_val_tensor)
    print(f'Validation Loss: {val_loss.item():.4f}')

    # 反归一化预测值
    predictions = predictions.numpy()
    y_val_tensor = y_val_tensor.numpy()

    predictions_denorm = predictions * (np.max(X_val) - np.min(X_val)) + np.min(X_val)
    y_val_denorm = y_val_tensor * (np.max(X_val) - np.min(X_val)) + np.min(X_val)

    # 打印预测结果和标签
    print("验证集预测结果和标签:")
    for i in range(len(predictions_denorm)):
        predicted_value = predictions_denorm[i][0]
        actual_value = y_val_denorm[i][0]
        predicted_label = get_label(predicted_value)
        actual_label = get_label(actual_value)
        print(
            f"预测值: {predicted_value:.2f}, 预测标签: {predicted_label}, 实际值: {actual_value:.2f}, 实际标签: {actual_label}")
