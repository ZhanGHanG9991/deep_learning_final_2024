import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.data_preprocessing import load_and_preprocess_data
from models.rnn import RNN

# Define the argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train an RNN model for time series forecasting.")
    
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset.")
    parser.add_argument('--input_len', type=int, default=192, help="Input sequence length.")
    parser.add_argument('--output_len', type=int, default=96, help="Output sequence length.")
    parser.add_argument('--hidden_size', type=int, default=64, help="Hidden layer size of the RNN.")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate for the optimizer.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train the model.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    
    return parser.parse_args()

# Hyperparameters
opt = parse_args()

# Load and preprocess the data
train_X, train_y, valid_X, valid_y, test_X, test_y, target_col, feature_cols = load_and_preprocess_data(opt.data_path, opt.input_len, opt.output_len)

input_size = len(feature_cols)  # 每个时间步的特征数
hidden_size = opt.hidden_size
output_size = opt.output_len  # 输出的长度

model = RNN(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

def compute_loss(model, data_X, data_y):
    model.eval()
    with torch.no_grad():
        output = model(data_X)
        loss = criterion(output, data_y)
    return loss.item()

def train(model, train_X, train_y, valid_X, valid_y, test_X, test_y, epochs, batch_size):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0  # 用于累积每个 epoch 的总损失
        for i in range(0, len(train_X), batch_size):
            batch_X = train_X[i:i + batch_size]
            batch_y = train_y[i:i + batch_size]

            if len(batch_X) == 0:
                continue  # 跳过空批次
            
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
        mae = mean_absolute_error(test_y.numpy(), output.numpy())  # MAE computation
    print(f"Test MSE: {mse}, Test MAE: {mae}")

train(model, train_X, train_y, valid_X, valid_y, test_X, test_y, opt.epochs, opt.batch_size)
test(model, test_X, test_y)