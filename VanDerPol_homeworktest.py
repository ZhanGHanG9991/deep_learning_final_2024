import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import wandb
#TODO: odeint_adjoint vs odeint
from torchdiffeq import odeint
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from umap import UMAP
import time 
import torchode

with open("/scratch/gpfs/kf1298/code/RNN/car_racing_data_32x32_120.pkl", "rb") as f:
    data = pickle.load(f)

wandb.init(project="Test_Car")

class VanDerPol(nn.Module):
    def __init__(self, alpha1, alpha2, W):
        super(VanDerPol, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.W = W

    def forward(self, t, hidden):
        h1, h2 = hidden[0], hidden[1]
        #TODO: residual or ...?
        dh1dt = self.alpha1 * h1 * (1 - h2**2) + self.alpha2 * h2 + self.W*h1
        dh2dt = -h1
        print(f"alpha1: {self.alpha1}, alpha2: {self.alpha2}, W: {self.W}") 
        return torch.stack([dh1dt, dh2dt], dim = 0)

class RNNwithODE(nn.Module):
    def __init__(self, action_size, hidden_size, output_size, alpha1, alpha2, W):
        super(RNNwithODE, self).__init__()
        self.hidden_dim = hidden_size
        self.output_size = output_size
        self.ode_func = VanDerPol(alpha1, alpha2, W)
        self.rnn = nn.RNNCell(input_size=3 * 32 * 32 + action_size, hidden_size=hidden_size)
        # TODO: initialization
        self.fc = nn.Linear(hidden_size, output_size)
    
    def input_to_hidden(self, x):
        #TODO: initialization?
        h1_initial = torch.randn(x.size(0), self.hidden_dim)
        h2_initial = torch.randn(x.size(0), self.hidden_dim)
        hidden = torch.stack([h1_initial, h2_initial])
        print(f"input_to_hidden shape: {hidden.shape}") 
        #TODO: how many features
        return hidden
    
    def forward(self, state, action, hidden = None, t_span = None):
        # initialization
        if hidden is None:
            hidden = self.input_to_hidden(state, x)

        if t_span is None:
            t_span = torch.linspace(0, 1, 5, device=hidden.device)
            
        state_flat = state.view(state.size(0), -1)
        x = torch.cat((state_flat, action), dim=1).unsqueeze(1)

        start = time.time()
        hidden = torchode.ODEProblem(self.ode_func, hidden, t_span)
        end=time.time() 
        print(f"ODE_Time: {end-start}")
        # print(f"ODE trajectory shape: {hidden.shape}") 
        #TODO: how to get the last hidden layer
        next_state_pred = self.fc(hidden[-1, 0])
        return next_state_pred, hidden

class WorldModelDataLoader:
    def __init__(self, data, batch_size, sequence_length, device):
        self.data = data
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device

        # 拆分数据为 train, valid, test 集合
        split_train = int(0.8 * len(self.data))
        split_valid = int(0.1 * len(self.data))
        self.train_data = self.data[:split_train]
        self.valid_data = self.data[split_train:split_train + split_valid]
        self.test_data = self.data[split_train + split_valid:]

        self.set_train()

    def set_train(self):
        self.current_data = self.train_data
        self.index = 0
        self.sub_index = 0  # 子序列的起始索引

    def set_valid(self):
        self.current_data = self.valid_data
        self.index = 0
        self.sub_index = 0

    def set_test(self):
        self.current_data = self.test_data
        self.index = 0
        self.sub_index = 0

    def get_batch(self):
        states, actions = [], []
        batch_data = self.current_data[self.index: self.index + self.batch_size]

        for sequence in batch_data:
            state_seq = [torch.tensor(step[0]) for step in sequence[self.sub_index:self.sub_index + self.sequence_length]]
            action_seq = [torch.tensor(step[1]) for step in sequence[self.sub_index:self.sub_index + self.sequence_length]]
            if len(state_seq) < self.sequence_length:
                pad_len = self.sequence_length - len(state_seq)
                state_seq += [torch.zeros_like(state_seq[0])] * pad_len
                action_seq += [torch.zeros_like(action_seq[0])] * pad_len

            states.append(torch.stack(state_seq))
            actions.append(torch.stack(action_seq))

        self.sub_index += self.sequence_length
        if self.sub_index >= len(self.current_data[self.index]):
            self.index += self.batch_size  
            self.sub_index = 0  
        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)

        end_flag = self.index >= len(self.current_data)

        return states, actions, end_flag


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_size = 3
hidden_size = 128
output_size = 32 * 32 * 3
batch_size = 16
sequence_length = 10
num_epochs = 10
learning_rate = 3e-4
alpha1 = 0.1
alpha2 = 0.1
W = 0.1
t_span = torch.linspace(0, 1, 5, device=device)
activations = []

data_loader = WorldModelDataLoader(data, batch_size, sequence_length, device)
model = RNNwithODE(action_size=action_size, hidden_size=hidden_size, output_size=output_size, alpha1 = alpha1, alpha2 = alpha2, W = W).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(num_epochs=6):
    best_val_loss = float('inf') 

    for epoch in range(num_epochs):
        model.train()
        data_loader.set_train()
        total_train_loss = 0
        total_train_samples = 0

        while True:
            states, actions, end_flag = data_loader.get_batch()
            batch_size_actual = states.size(0)
            # Initialize the hidden state (h_0, c_0) for the LSTM, resetting it for each new batch
            hidden1_initial = torch.randn(batch_size_actual, hidden_size)
            hidden2_initial = torch.randn(batch_size_actual, hidden_size)
            hidden = torch.stack([hidden1_initial, hidden2_initial]).to(device)
            # Loop through each time step in the sequence
            for t in range(sequence_length - 1):
                start=time.time()
                current_state = states[:, t]
                action = actions[:, t]
                next_state = states[:, t + 1].view(batch_size_actual, -1)

                next_state_pred, hidden = model(current_state, action, hidden, t_span)
                # print(f"hidden shape output1: {hidden.shape}")
                loss = criterion(next_state_pred, next_state)
                end=time.time() 
                # print(f"forward_Time: {end-start}")

                optimizer.zero_grad()
                start=time.time()
                loss.backward()
                end=time.time()
                # print(f"backward_Time: {end-start}")
                optimizer.step()

                # hidden = torch.stack([h.detach() for h in hidden])

                #TODO: whether detach is necessary
                hidden = hidden[-1]
                # print(f"hidden shape after detach: {hidden.shape}")

                total_train_loss += loss.item()
                wandb.log({"Training Loss": loss.item()})
                total_train_samples += 1

            if end_flag:
                break

        avg_train_loss = total_train_loss / total_train_samples
        val_loss = evaluate()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "world_model_best.pth")
            print("Best model saved.")


def evaluate():
    model.eval()
    data_loader.set_valid()
    total_val_loss = 0
    total_val_samples = 0

    with torch.no_grad():
        while True:
            states, actions, end_flag = data_loader.get_batch()
            batch_size_actual = states.size(0)

            hidden1_initial = torch.randn(batch_size_actual, hidden_size)
            hidden2_initial = torch.randn(batch_size_actual, hidden_size)
            hidden = torch.stack([hidden1_initial, hidden2_initial]).to(device)
            # print(f"device:", device)

            for t in range(sequence_length - 1):
                current_state = states[:, t]
                action = actions[:, t]
                next_state = states[:, t + 1].view(batch_size_actual, -1)

                next_state_pred, hidden = model(current_state, action, hidden, t_span)
                loss = criterion(next_state_pred, next_state)

                hidden = hidden[-1]
                total_val_loss += loss.item()
                wandb.log({"Validation Loss": loss.item()})
                total_val_samples += 1

            if end_flag:
                break

    avg_val_loss = total_val_loss / total_val_samples
    return avg_val_loss

def test():
    model.eval()
    data_loader.set_test()
    total_test_loss = 0
    total_test_samples = 0

    with torch.no_grad():
        while True:
            states, actions, end_flag = data_loader.get_batch()
            batch_size_actual = states.size(0)

            hidden1_initial = torch.randn(batch_size_actual, hidden_size)
            hidden2_initial = torch.randn(batch_size_actual, hidden_size)
            hidden = torch.stack([hidden1_initial, hidden2_initial]).to(device)

            for t in range(sequence_length - 1):
                current_state = states[:, t]
                action = actions[:, t]
                next_state = states[:, t + 1].view(batch_size_actual, -1)
                next_state_pred, hidden = model(current_state, action, hidden)
                loss = criterion(next_state_pred, next_state)

                #TODO extract hidden?
                hidden_extract_1 = hidden[:, 0, :, :]
                activations.append(hidden_extract_1.clone().cpu().numpy())
                hidden = hidden[-1]

                total_test_loss += loss.item()
                total_test_samples += 1

            if end_flag:
                break

    avg_test_loss = total_test_loss / total_test_samples
    print(f"Test Loss: {avg_test_loss:.4f}")
    return activations

def PCA_analysis(activations):
    all_activations = np.concatenate(activations, axis=0)
    print(f"all_activations shape: {all_activations.shape}")
    # sliced_activations = all_activations[:, 10, :]  
    # print(f"sliced_activations shape: {sliced_activations.shape}")
    # n, num_neurons = sliced_activations.shape
    steps, batch, num_neurons = all_activations.shape
    reshaped_activations = all_activations.reshape(-1, num_neurons)
    # reshaped_activations = all_activations.reshape(n, num_neurons).T 
    #TODO: component number
    # pca = PCA()
    # pca.fit(reshaped_activations)
    # explained_variance_ratio = pca.explained_variance_ratio_
    # cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    # plt.plot(
    #     range(1, len(cumulative_variance_ratio) + 1),
    #     cumulative_variance_ratio,
    #     marker='o',
    #     linestyle='--',
    #     label='Cumulative Explained Variance'
    # )
    # plt.savefig("/scratch/gpfs/kf1298/code/NeurODE/Cumulative_Explained_Variance.png")
    # n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    # print(f"n_components: {n_components}")
    pca = PCA(n_components=105)
    reduced_activations = pca.fit_transform(reshaped_activations)
    reducer = UMAP(n_components=3, random_state=42)
    umap_result = reducer.fit_transform(reduced_activations)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # color_values = umap_result[:, 0] 
    time_steps = np.arange(reshaped_activations.shape[0])
    color_values = time_steps
    norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())
    color_values = norm(color_values)
    # norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())
    # color_values = norm(color_values)
    cmap = plt.cm.viridis

    ax.scatter(umap_result[:, 0], umap_result[:, 1], umap_result[:, 2], alpha=0.7, c=color_values, cmap=cmap, s = 1)
    ax.set_title("UMAP of Hidden Activations")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_zlabel("UMAP3")
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([]) 
    plt.savefig("/scratch/gpfs/kf1298/code/NeurODE/PCA.png")



train()
evaluate()
activations = test()
PCA_analysis(activations)