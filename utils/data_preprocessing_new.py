import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

def load_and_preprocess_data(csv_file, input_len, output_len):
    # Load the dataset
    data = pd.read_csv(csv_file, parse_dates=['date'])
    print("1")
    target_col = data.columns[-1]
    feature_cols = [col for col in data.columns if col != target_col and col != 'date']

    # Split the data into train, validation, and test sets
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.15)
    test_size = len(data) - train_size - val_size

    train_data = data[:train_size]
    valid_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Standardize the input features and target column
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    train_data.loc[:, feature_cols] = scaler_x.fit_transform(train_data[feature_cols])
    train_data.loc[:, target_col] = scaler_y.fit_transform(train_data[[target_col]])

    valid_data.loc[:, feature_cols] = scaler_x.transform(valid_data[feature_cols])
    valid_data.loc[:, target_col] = scaler_y.transform(valid_data[[target_col]])

    test_data.loc[:, feature_cols] = scaler_x.transform(test_data[feature_cols])
    test_data.loc[:, target_col] = scaler_y.transform(test_data[[target_col]])
    print("2")
    # Create datasets for training, validation, and testing
    train_X, train_y = create_dataset(train_data, input_len, output_len, feature_cols)
    print("3")
    valid_X, valid_y = create_dataset(valid_data, input_len, output_len, feature_cols)
    print("4")
    test_X, test_y = create_dataset(test_data, input_len, output_len, feature_cols)
    print("5")

    

    return train_X, train_y, valid_X, valid_y, test_X, test_y, target_col, feature_cols


# def create_dataset(data, input_len, output_len, feature_cols):
#     X, y = [], []
#     for i in range(0, len(data) - input_len - output_len):
#         X.append(data[feature_cols].iloc[i:i + input_len].values)
#         y.append(data[data.columns[-1]].iloc[i + input_len:i + input_len + output_len].values)
#         X_array = np.array(X)
#         y_array = np.array(y)
#     return torch.tensor(X_array, dtype=torch.float32), torch.tensor(y_array, dtype=torch.float32)

def create_dataset(data, input_len, output_len, feature_cols):
    num_samples = len(data) - input_len - output_len

    # 获取特征矩阵和目标向量
    features = data[feature_cols].values  # 转为 NumPy 数组
    targets = data[data.columns[-1]].values

    # 构造滑动窗口的索引
    X_indices = np.arange(input_len) + np.arange(num_samples).reshape(-1, 1)
    y_indices = np.arange(output_len) + input_len + np.arange(num_samples).reshape(-1, 1)
    
    # 提取数据
    X_array = features[X_indices]
    y_array = targets[y_indices]

    # 转换为 PyTorch 张量
    return torch.tensor(X_array, dtype=torch.float32), torch.tensor(y_array, dtype=torch.float32)
