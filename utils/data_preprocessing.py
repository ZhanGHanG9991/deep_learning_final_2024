import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

def load_and_preprocess_data(csv_file, input_len, output_len):
    # Load the dataset
    data = pd.read_csv(csv_file, parse_dates=['date'])

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

    # Create datasets for training, validation, and testing
    train_X, train_y = create_dataset(train_data, input_len, output_len, feature_cols)
    valid_X, valid_y = create_dataset(valid_data, input_len, output_len, feature_cols)
    test_X, test_y = create_dataset(test_data, input_len, output_len, feature_cols)

    return train_X, train_y, valid_X, valid_y, test_X, test_y, target_col, feature_cols


def create_dataset(data, input_len, output_len, feature_cols):
    X, y = [], []
    for i in range(0, len(data) - input_len - output_len):
        X.append(data[feature_cols].iloc[i:i + input_len].values)
        y.append(data[data.columns[-1]].iloc[i + input_len:i + input_len + output_len].values)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
