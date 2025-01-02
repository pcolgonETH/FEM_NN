# Import necessary libraries
import time  # For measuring execution time
import torch  # PyTorch for building and training neural networks
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation
from sklearn.preprocessing import MinMaxScaler  # For scaling data
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset  # Data utilities for PyTorch
import matplotlib.pyplot as plt  # For visualization
import joblib  # For saving/loading Python objects (not used here)
from sklearn.model_selection import train_test_split  # For splitting datasets
import random  # For random number generation
import json  # For working with JSON files
import os  # For file system operations

# Function to load data from a file and reshape it into simulations, increments, and features
def load_data_with_columns(input_path, num_simulations, num_increments, num_features):
    data = pd.read_csv(input_path, header=None)  # Load CSV data without headers
    data = data.iloc[:, 0:num_features]  # Select the first `num_features` columns
    data_np = data.to_numpy()  # Convert DataFrame to a NumPy array
    reshaped_data = data_np.reshape(num_simulations, num_increments, num_features)  # Reshape into 3D (simulations, increments, features)
    return reshaped_data

# Manual MinMaxScaler implementation for scaling data
def manual_minmax_scaler(data, feature_range=(-1, 1)):
    data_min = np.min(data, axis=(0, 1))  # Minimum value for each feature across all simulations/increments
    data_max = np.max(data, axis=(0, 1))  # Maximum value for each feature
    data_range = data_max - data_min  # Range = max - min
    scale = np.where(data_range != 0, (feature_range[1] - feature_range[0]) / data_range, 0)  # Scale to the target range
    data_scaled = np.where(data_range != 0, scale * (data - data_min) + feature_range[0], feature_range[0])  # Apply scaling
    return data_scaled, data_min, data_max, scale  # Return scaled data and parameters

# Function to revert data back to its original range using scaling parameters
def manual_inverse_transform(scaled_data, data_min, data_max, scale, feature_range=(-1, 1)):
    return np.where(scale != 0, (scaled_data - feature_range[0]) / scale + data_min, data_min)  # Apply inverse scaling

# Wrapper for normalizing data with the manual MinMaxScaler and storing scaling parameters
def normalize_data(data, scaler_function=manual_minmax_scaler):
    normalized_data, data_min, data_max, scale = scaler_function(data)  # Apply scaling
    scaling_params = {'data_min': data_min, 'data_max': data_max, 'scale': scale}  # Store scaling parameters
    return normalized_data, scaling_params

# Define a custom LSTM cell for single timestep processing
class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Define weights for the input, forget, cell, and output gates
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)

    # Forward pass through the LSTM cell
    def forward(self, x, h_prev, c_prev):
        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))  # Input gate
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))  # Forget gate
        c_hat_t = torch.tanh(self.W_c(x) + self.U_c(h_prev))  # Candidate cell state
        c_t = f_t * c_prev + i_t * c_hat_t  # Update cell state
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))  # Output gate
        h_t = o_t * torch.tanh(c_t)  # Hidden state
        return h_t, c_t

# Define an LSTM model using multiple custom LSTM cells
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim] * num_layers
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Create a list of LSTM cells
        self.lstm_cells = nn.ModuleList([
            CustomLSTMCell(
                input_size=input_dim if i == 0 else self.hidden_dim[i - 1],  # Input size for the first layer
                hidden_size=self.hidden_dim[i]  # Hidden size for each layer
            ) for i in range(num_layers)
        ])
        self.fc = nn.Linear(self.hidden_dim[-1], output_dim)  # Fully connected layer for final output

    # Forward pass through the LSTM model
    def forward(self, x_t, h_t, c_t):
        for i, lstm_cell in enumerate(self.lstm_cells):
            if i == 0:
                h_t[i], c_t[i] = lstm_cell(x_t, h_t[i], c_t[i])  # First layer processes input
            else:
                h_t[i], c_t[i] = lstm_cell(h_t[i - 1], h_t[i], c_t[i])  # Subsequent layers process previous hidden state
        output = self.fc(h_t[-1])  # Map the last hidden state to the output
        return output, h_t, c_t

# Function to train the LSTM model one timestep at a time
def train_custom_lstm_timestep(model, train_loader, num_epochs, learning_rate, loss_save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
    model.to(device)  # Move model to the selected device
    criterion = nn.SmoothL1Loss()  # Loss function for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Adam optimizer
    epoch_losses = []

    for epoch in range(num_epochs):  # Loop over epochs
        model.train()  # Set model to training mode
        running_loss = 0.0
        for inputs, targets in train_loader:  # Loop over batches
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
            batch_size, seq_len, _ = inputs.size()
            # Initialize hidden and cell states for each layer
            h_t = [torch.zeros(batch_size, model.hidden_dim[i], device=device) for i in range(model.num_layers)]
            c_t = [torch.zeros(batch_size, model.hidden_dim[i], device=device) for i in range(model.num_layers)]
            optimizer.zero_grad()  # Clear gradients
            outputs = []

            # Loop over time steps
            for t in range(seq_len):
                x_t = inputs[:, t, :]  # Input at time step t
                y_t = targets[:, t, :]  # Target at time step t
                output_t, h_t, c_t = model.forward(x_t, h_t, c_t)  # Forward pass
                outputs.append(output_t.unsqueeze(1))  # Collect outputs

            outputs = torch.cat(outputs, dim=1)  # Concatenate outputs for all time steps
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            running_loss += loss.item() * inputs.size(0)  # Accumulate loss

        epoch_loss = running_loss / len(train_loader.dataset)  # Compute average loss per epoch
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.8f}")

    # Save the loss history to a file
    with open(loss_save_path, 'w') as f:
        for epoch, loss in enumerate(epoch_losses, 1):
            f.write(f"Epoch {epoch}: {loss:.8f}\n")
    print(f"Losses saved to {loss_save_path}")
    return model  # Return the trained model

# Function to save scaling parameters to a JSON file
def save_scaling_params(scaler, filepath):
    scaler = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in scaler.items()}  # Convert arrays to lists
    with open(filepath, 'w') as f:
        json.dump(scaler, f)  # Save to JSON file

# Function to validate and plot predictions for a single simulation
def validate_and_plot_single_simulation(model, val_loader, target_scaler, timestep_mode=True, save_directory="."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model.to(device)  # Move model to the selected device
    model.eval()  # Set model to evaluation mode
    predictions, targets, h_t_list, c_t_list = [], [], [], []  # Initialize storage for outputs

    with torch.no_grad():  # Disable gradient computation
        for inputs, target in val_loader:  # Loop over batches
            inputs, target = inputs.to(device), target.to(device)
            batch_size, seq_len, input_dim = inputs.size()

            if timestep_mode:  # Process timestep by timestep
                h_t = [torch.zeros(batch_size, model.hidden_dim[layer], device=device) for layer in range(model.num_layers)]
                c_t = [torch.zeros(batch_size, model.hidden_dim[layer], device=device) for layer in range(model.num_layers)]
                timestep_outputs = []
                for t in range(seq_len):  # Loop over time steps
                    x_t = inputs[:, t, :]  # Input at time step t
                    output_t, h_t, c_t = model.forward(x_t, h_t, c_t)  # Forward pass
                    timestep_outputs.append(output_t.unsqueeze(1))  # Collect outputs
                    h_t_list.append([h.detach().cpu().numpy() for h in h_t])  # Save hidden states
                    c_t_list.append([c.detach().cpu().numpy() for c in c_t])  # Save cell states
                outputs = torch.cat(timestep_outputs, dim=1)  # Combine outputs
            else:  # Process entire sequence at once
                outputs = model(inputs)

            predictions.append(outputs.cpu().numpy())  # Save predictions
            targets.append(target.cpu().numpy())  # Save targets

    predictions = np.concatenate(predictions, axis=0)  # Combine all predictions
    targets = np.concatenate(targets, axis=0)  # Combine all targets

    # Rescale predictions and targets back to their original range
    data_min = target_scaler['data_min']
    data_max = target_scaler['data_max']
    scale = target_scaler['scale']
    predictions_original = manual_inverse_transform(predictions, data_min, data_max, scale, feature_range=(-1, 1))
    targets_original = manual_inverse_transform(targets, data_min, data_max, scale, feature_range=(-1, 1))

    # Select a random simulation for visualization
    random_simulation_index = random.randint(0, predictions_original.shape[0] - 1)
    num_outputs = predictions_original.shape[2]  # Number of output features
    time_increments = np.arange(predictions_original.shape[1])  # Time increments for the plot
    simulation_folder = os.path.join(save_directory, f"Simulation_V1_{random_simulation_index}")
    os.makedirs(simulation_folder, exist_ok=True)  # Create folder for plots

    # Generate and save plots for each output feature
    for i in range(num_outputs):
        plt.figure()
        plt.plot(time_increments, predictions_original[random_simulation_index, :, i], label="Predicted", linestyle="-", marker="o")
        plt.plot(time_increments, targets_original[random_simulation_index, :, i], label="True", linestyle="--", marker="x")
        plt.xlabel("Time Increments")
        plt.ylabel(f"Output {i + 1}")
        plt.title(f"Predicted vs True for Output {i + 1} - Simulation {random_simulation_index}")
        plt.legend()
        plot_path = os.path.join(simulation_folder, f"output_{i + 1}.png")
        plt.savefig(plot_path)
        plt.close()
    print(f"Plots saved to {simulation_folder}")

# Main function for loading data, training, and validating the model
def main():
    # Define file paths and dataset parameters
    input_path = 'path/to/features.data'
    target_path = 'path/to/targets.data'
    num_simulations, num_increments, num_features, num_targets = 863, 1000, 12, 6

    # Load and normalize feature and target data
    feature_data = load_data_with_columns(input_path, num_simulations, num_increments, num_features)
    target_data = load_data_with_columns(target_path, num_simulations, num_increments, num_targets)
    feature_data_normalized, feature_scaler = normalize_data(feature_data)
    target_data_normalized, target_scaler = normalize_data(target_data)

    # Split data into training and testing sets
    train_features, test_features, train_targets, test_targets = train_test_split(
        feature_data_normalized, target_data_normalized, test_size=0.1, random_state=42)

    # Create DataLoader objects for training and validation
    batch_size = 10
    train_dataset = TensorDataset(torch.Tensor(train_features), torch.Tensor(train_targets))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(torch.Tensor(test_features), torch.Tensor(test_targets))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the LSTM model
    model = LSTMModel(input_dim=12, hidden_dim=70, num_layers=4, output_dim=6)

    # Train the model and save training losses
    loss_save_path = 'path/to/losses.txt'
    model = train_custom_lstm_timestep(model, train_loader, num_epochs=200, learning_rate=0.0002, loss_save_path=loss_save_path)

    # Save the trained model
    save_path = 'path/to/model.pth'
    torch.save(model.state_dict(), save_path)
    save_path_pt = 'path/to/model_scripted.pt'
    scripted_model = torch.jit.script(model)
    scripted_model.save(save_path_pt)

    # Validate the model and generate plots
    validate_and_plot_single_simulation(model, val_loader, target_scaler, timestep_mode=True, save_directory='path/to/plots')

# Run the main function
if __name__ == "__main__":
    main()
