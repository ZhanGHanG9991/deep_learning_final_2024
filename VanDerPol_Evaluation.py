import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.VanDerPol import RNNwithODE
from utils.data_preprocessing import load_and_preprocess_data
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from umap import UMAP

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



def compute_loss(model, criterion, data_X, data_y, device):
    model.eval()
    with torch.no_grad():
        data_X, data_y = data_X.to(device), data_y.to(device)
        output, hidden = model(data_X)
        loss = criterion(output, data_y)
    return loss.item()

def evaluate(model, test_X, test_y, device):
    model.eval()
    activations = []
    with torch.no_grad():
        test_X = test_X.to(device)
        test_y = test_y.to(device)
        output, hidden = model(test_X)
        # Extract and store hidden states
        hidden_extract = hidden.clone().cpu().numpy()
        activations.append(hidden_extract)
        
        output_np = output.detach().cpu().numpy()
        test_y_np = test_y.detach().cpu().numpy()
        mse = mean_squared_error(test_y_np, output_np)
        mae = mean_absolute_error(test_y_np, output_np)
    return mse, mae, activations

def train(model, criterion, optimizer, train_X, train_y, valid_X, valid_y, test_X, test_y, epochs, batch_size, device, hidden_size,t_span):
    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # Training loop
        for batch_X, batch_y in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_size_actual = batch_X.size(0)

            hidden1_initial = torch.randn(batch_size_actual, hidden_size)
            hidden2_initial = torch.randn(batch_size_actual, hidden_size)
            hidden = torch.stack([hidden1_initial, hidden2_initial]).to(device)
            sequence_length = batch_X.size(1)

            
            for t in range(sequence_length - 1):
                output, hidden = model(batch_X, hidden, t_span)
                loss = criterion(output, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # hidden = hidden[-1]

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        valid_loss = compute_loss(model, criterion, valid_X, valid_y, device)
        test_loss = compute_loss(model, criterion, test_X, test_y, device)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        test_losses.append(test_loss)

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Test Loss: {test_loss:.4f}")

        # 如果需要保存最优模型，可以在此处进行
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # torch.save(model.state_dict(), "best_model.pth")
    
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].set_title('Train Loss')
    axes[0].set_ylabel('Loss')

    axes[1].plot(valid_losses, label='Valid Loss')
    axes[1].set_title('Valid Loss')
    axes[1].set_ylabel('Loss')

    axes[2].plot(test_losses, label='Test Loss')
    axes[2].set_title('Test Loss')
    axes[2].set_ylabel('Loss')

    plt.savefig('pics/losses.png')

    mse, mae, activations = evaluate(model, test_X, test_y, device)
    print(f"Final Test MSE: {mse:.4f}, Final Test MAE: {mae:.4f}")
    
    # Perform PCA analysis
    PCA_analysis(activations)    
    print(f"Final Test MSE: {mse:.4f}, Final Test MAE: {mae:.4f}")

def PCA_analysis(activations):
    all_activations = np.concatenate(activations, axis=0)
    print(f"all_activations shape: {all_activations.shape}")
    
    # Reshape activations for PCA
    steps, batch, num_neurons = all_activations.shape
    reshaped_activations = all_activations.reshape(-1, num_neurons)
    
    # PCA reduction
    pca = PCA(n_components=105)
    reduced_activations = pca.fit_transform(reshaped_activations)
    
    # UMAP reduction
    reducer = UMAP(n_components=3, random_state=42)
    umap_result = reducer.fit_transform(reduced_activations)

    # Create 3D visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color coding based on time steps
    time_steps = np.arange(reshaped_activations.shape[0])
    color_values = time_steps
    norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())
    color_values = norm(color_values)
    cmap = plt.cm.viridis

    # Create scatter plot
    ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2], 
              alpha=0.7, c=color_values, cmap=cmap, s=1)
    ax.set_title("UMAP of Hidden Activations")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_zlabel("UMAP3")
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([]) 
    
    plt.savefig('pics/hidden_state_visualization.png')

if __name__ == "__main__":
    opt = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_X, train_y, valid_X, valid_y, test_X, test_y, target_col, feature_cols = load_and_preprocess_data(
        opt.data_path, opt.input_len, opt.output_len
    )

    input_size = len(feature_cols)
    output_size = opt.output_len
    hidden_size = opt.hidden_size

    train_X, train_y = train_X.to(device), train_y.to(device)
    valid_X, valid_y = valid_X.to(device), valid_y.to(device)
    test_X, test_y = test_X.to(device), test_y.to(device)

    alpha1 = 0.1
    alpha2 = 0.1
    W = 0.1
    t_span = torch.linspace(0, 1, 5, device=device)

    model = RNNwithODE(input_size, hidden_size, output_size, alpha1=alpha1, alpha2=alpha2, W=W).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    train(model, criterion, optimizer, train_X, train_y, valid_X, valid_y, test_X, test_y, opt.epochs, opt.batch_size, device, hidden_size,t_span)
