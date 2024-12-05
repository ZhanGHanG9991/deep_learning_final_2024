import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class TimeSeriesModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TimeSeriesModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出
        return out

def create_dataset(data, input_len, output_len, feature_cols):
    X, y = [], []
    for i in range(0, len(data) - input_len - output_len):
        X.append(data[feature_cols].iloc[i:i + input_len].values)
        y.append(data[data.columns[-1]].iloc[i + input_len:i + input_len + output_len].values)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Load the dataset
data = pd.read_csv('./datasets/weather/weather.csv', parse_dates=['date'])

target_col = data.columns[-1]
feature_cols = [col for col in data.columns if col != target_col and col != 'date']

# Split the data into train, validation, and test sets
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)
test_size = len(data) - train_size - val_size

train_data = data[:train_size]
valid_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

# 对输入特征和目标列进行标准化
scaler_x = StandardScaler()
scaler_y = StandardScaler()

train_data.loc[:, feature_cols] = scaler_x.fit_transform(train_data[feature_cols])
train_data.loc[:, target_col] = scaler_y.fit_transform(train_data[[target_col]])

valid_data.loc[:, feature_cols] = scaler_x.transform(valid_data[feature_cols])
valid_data.loc[:, target_col] = scaler_y.transform(valid_data[[target_col]])

test_data.loc[:, feature_cols] = scaler_x.transform(test_data[feature_cols])
test_data.loc[:, target_col] = scaler_y.transform(test_data[[target_col]])

input_len = 192  # 输入序列长度
output_len = 96  # 输出预测的长度

train_X, train_y = create_dataset(train_data, input_len, output_len, feature_cols)
valid_X, valid_y = create_dataset(valid_data, input_len, output_len, feature_cols)
test_X, test_y = create_dataset(test_data, input_len, output_len, feature_cols)

input_size = len(feature_cols)  # 每个时间步的特征数
hidden_size = 64  # LSTM的隐藏层大小
output_size = output_len  # 输出的长度

model = TimeSeriesModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def compute_loss(model, data_X, data_y):
    model.eval()
    with torch.no_grad():
        output = model(data_X)
        loss = criterion(output, data_y)
    return loss.item()

def train(model, train_X, train_y, valid_X, valid_y, test_X, test_y, epochs=10, batch_size=32):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0  # 用于累积每个 epoch 的总损失
        for i in range(0, len(train_X), batch_size):
            batch_X = train_X[i:i + batch_size]
            batch_y = train_y[i:i + batch_size]
            
            optimizer.zero_grad()
            output = model(batch_X)  # LSTM已处理了特征维度
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()  # 累积每个批次的损失

        # 计算并输出训练集、验证集和测试集的损失
        train_loss = compute_loss(model, train_X, train_y)
        valid_loss = compute_loss(model, valid_X, valid_y)
        test_loss = compute_loss(model, test_X, test_y)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}, Test Loss: {test_loss:.4f}")

def test(model, test_X, test_y):
    model.eval()
    with torch.no_grad():
        output = model(test_X)
        mse = mean_squared_error(test_y.numpy(), output.numpy())
    print(f"Test MSE: {mse}")

train(model, train_X, train_y, valid_X, valid_y, test_X, test_y, epochs=20)
test(model, test_X, test_y)