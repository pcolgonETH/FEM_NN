import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import joblib
import json
import mplcursors
import os

# Utility functions
def load_data(input_path, output_path, threshold=1e-08):
    input_data = np.loadtxt(input_path, delimiter=',')
    output_data = np.loadtxt(output_path, delimiter=',')

    # Apply the threshold to round values smaller than threshold to 0
    #input_data[np.abs(input_data) < threshold] = 0
    #output_data[np.abs(output_data) < threshold] = 0

    return input_data, output_data


# Manual MinMaxScaler implementation
def manual_minmax_scaler(data, feature_range=(-1, 1)):
    # Calculate min and max for each column (feature)
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    # Calculate the range, avoiding division by zero
    data_range = data_max - data_min

    # To avoid division by zero, set scale to zero where range is zero
    scale = np.where(data_range != 0, (feature_range[1] - feature_range[0]) / data_range, 0)

    # Apply the transformation
    data_scaled = np.where(data_range != 0, scale * (data - data_min) + feature_range[0], feature_range[0])

    # Return the scaled data along with parameters (for possible inverse transform)
    return data_scaled, data_min, data_max, scale


def manual_inverse_transform(scaled_data, data_min, data_max, scale, feature_range=(-1, 1)):
    # Ensure data_min, data_max, and scale are numeric arrays
    data_min = np.array(data_min, dtype=float)
    data_max = np.array(data_max, dtype=float)
    scale = np.array(scale, dtype=float)

    # Calculate the range
    data_range = data_max - data_min

    # Inverse transform: Handle zero scale (set the original value to data_min where scale is 0)
    data_original = np.where(scale != 0, (scaled_data - feature_range[0]) / scale + data_min, data_min)

    return data_original


# Modified normalize_data function
def normalize_data(data, scaler_function=manual_minmax_scaler):
    normalized_data, data_min, data_max, scale = scaler_function(data)

    scaling_params = {
        'data_min': data_min,
        'data_max': data_max,
        'scale': scale
    }
    return normalized_data, scaling_params


# Save scaler parameters to JSON
def save_scaler_to_json_manual(scaler_params, filename):
    # Ensure the input is a dictionary and contains NumPy arrays or lists
    if isinstance(scaler_params, dict):
        data_min = scaler_params['data_min']
        data_max = scaler_params['data_max']
        scale = scaler_params['scale']

        data = {
            'min': data_min.tolist(),
            'max': data_max.tolist(),
            'scale': scale.tolist(),
        }

        try:
            with open(filename, 'w') as f:
                json.dump(data, f)
            print(f"Successfully saved scaler parameters to {filename}")
        except Exception as e:
            print(f"Error saving JSON: {e}")
    else:
        print(f"Error: scaler_params is not in the expected format. Received: {scaler_params}")



# Load scaler parameters from JSON
def load_scaler_from_json_manual(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    data_min = np.array(data['min'])
    data_max = np.array(data['max'])
    scale = np.array(data['scale'])

    # Sanity check to ensure you're loading the correct values
    print(f"Loaded scaling parameters: data_min={data_min}, data_max={data_max}, scale={scale}")

    return data_min, data_max, scale

def calculate_r_squared(true, predicted):
    ss_res = np.sum((true - predicted) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        target = self.targets[idx]
        return input, target

# Attention mechanism class
class Attention(nn.Module):
    def __init__(self, input_dim, bias_feature_idx=1, bias_strength=10.0):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim, input_dim))
        self.bias_feature_idx = bias_feature_idx
        self.bias_strength = bias_strength
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights)

        # Introduce a bias towards the second feature
        attention_scores[:, self.bias_feature_idx] += self.bias_strength

        attention_scores = self.softmax(attention_scores)

        weighted_sum = attention_scores * x
        return weighted_sum


# Neural network model class
class StrainToStressNN(nn.Module):
    def __init__(self, activation_function):
        super(StrainToStressNN, self).__init__()
        self.layer1 = nn.Linear(34, 50)
        self.layer2 = nn.Linear(50, 100)
        #self.attention = Attention(6)  # Attention applied after the second layer
        self.layer3 = nn.Linear(100, 100)
        self.layer4 = nn.Linear(100, 50)
        self.layer5 = nn.Linear(50, 50)
        self.output_layer = nn.Linear(50, 48)
        self.activation_function = activation_function

    def forward(self, x):
        x = self.activation_function(self.layer1(x))
        x = self.activation_function(self.layer2(x))
        #x = self.attention(x)  # Attention applied here
        x = self.activation_function(self.layer3(x))
        x = self.activation_function(self.layer4(x))
        x = self.activation_function(self.layer5(x))
        x = self.output_layer(x)
        return x

class StrainToStressCNN(nn.Module):
    def __init__(self, activation_function):
        super(StrainToStressCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1, stride=1)
        #self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.activation_function = activation_function
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 6, 16)
        self.fc2 = nn.Linear(16, 6)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.activation_function(self.conv1(x))
        #x = self.activation_function(self.conv2(x))
        x = self.activation_function(self.conv3(x))
        x = self.flatten(x)
        x = self.activation_function(self.fc1(x))
        x = self.fc2(x)
        return x

# Training and evaluation functions
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Print the current epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}]')

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            predicted_output = model(inputs)
            loss = criterion(predicted_output, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        # Print loss after every epoch
        print(f'Training Loss: {train_loss:.5f} | Validation Loss: {val_loss:.5f}')
    return train_losses, val_losses


def validate_model(model, val_loader, criterion, output_scaler_params, device):
    model.eval()
    val_losses = []
    r_squared_values = []
    true_outputs_list = []
    predicted_outputs_list = []
    original_inputs_list = []

    output_data_min, output_data_max, output_data_scale = output_scaler_params['data_min'], output_scaler_params['data_max'], output_scaler_params['scale']

    # Debugging: print the scaling parameters
    print("Checking output scaling parameters before inverse transformation:")
    print("data_min:", output_data_min)
    print("data_max:", output_data_max)
    print("scale:", output_data_scale)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_losses.append(loss.item() * inputs.size(0))

            true_outputs_list.append(targets.cpu().numpy())
            predicted_outputs_list.append(outputs.cpu().numpy())
            original_inputs_list.append(inputs.cpu().numpy())

            # Manually inverse transform the scaled outputs
            true_outputs = np.concatenate(true_outputs_list, axis=0)
            predicted_outputs = np.concatenate(predicted_outputs_list, axis=0)

            true_outputs[:, :48] = manual_inverse_transform(true_outputs[:, :48], output_data_min, output_data_max,
                                                            output_data_scale)
            predicted_outputs[:, :48] = manual_inverse_transform(predicted_outputs[:, :48], output_data_min,
                                                                 output_data_max, output_data_scale,)

            # Calculate R-squared
            r_squared = calculate_r_squared(true_outputs, predicted_outputs)
            r_squared_values.append(r_squared)

            print("Features (Input) at Original Scale:")
            print(original_inputs_list[-1])

            print("Predicted Outputs at Original Scale:")
            print(predicted_outputs)
            print("-" * 30)

    val_loss = sum(val_losses) / len(val_loader.dataset)
    r_squared = np.mean(r_squared_values)
    return val_loss, r_squared, true_outputs, predicted_outputs, true_outputs, predicted_outputs

def plot_predicted_vs_true(true_outputs, predicted_outputs, target_names, ids, output_dir):
    num_targets = true_outputs.shape[1]
    for i in range(num_targets):
        plt.figure(figsize=(10, 5))

        # Filter out NaN or infinite values before fitting
        valid_mask = np.isfinite(true_outputs[:, i]) & np.isfinite(predicted_outputs[:, i])
        true_valid = true_outputs[valid_mask, i]
        predicted_valid = predicted_outputs[valid_mask, i]

        # Scatter plot only for valid data points
        scatter = plt.scatter(true_valid, predicted_valid, alpha=0.5, label='Data Points')

        # Add IDs as hover labels using mplcursors
        cursor = mplcursors.cursor(scatter, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(f'ID: {ids[sel.index]}'))

        # Only perform polyfit if there are enough valid points
        if len(true_valid) > 1 and len(predicted_valid) > 1:
            fit = np.polyfit(true_valid, predicted_valid, 1)
            fit_fn = np.poly1d(fit)
            plt.plot(true_valid, fit_fn(true_valid), '--k')
        else:
            print(f"Not enough valid data points for polyfit on target {i + 1}")

        plt.xlabel('True Outputs')
        plt.ylabel('Predicted Outputs')
        plt.title(f'Predicted vs True Outputs for {target_names[i]}')
        plt.legend()

        # Save the plot to the specified folder
        plot_filename = os.path.join(output_dir, f'pred_vs_true_{target_names[i]}.png')
        plt.savefig(plot_filename)

        #plt.show()
        #time.sleep(1.5)

# Sequential evaluation of a single simulation
def evaluate_single_simulation(model, validation_loader, scaler_params, device):
    model.eval()

    # Select a random simulation from the validation set
    for inputs, targets in validation_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        print(f"Batch inputs shape: {inputs.shape}")  # Debugging: Check shape of batch
        break  # Take the first batch (random by shuffle in DataLoader)

    # Select one simulation
    simulation_inputs = inputs[0, :, :]  # Take all time steps and features for the first simulation
    simulation_targets = targets[0, :, :]  # Corresponding target data

    # Sequential prediction
    predictions = []
    with torch.no_grad():
        for t in range(simulation_inputs.size(0)):  # Iterate over time steps
            input_step = simulation_inputs[t, :].unsqueeze(0)  # Add batch dimension
            output_step = model(input_step)  # Predict output for current step
            predictions.append(output_step.cpu().numpy())

    # Convert predictions to a single array
    predictions = np.array(predictions).squeeze()

    # Inverse transform predictions and true targets
    data_min, data_max, scale = scaler_params['data_min'], scaler_params['data_max'], scaler_params['scale']
    predictions = manual_inverse_transform(predictions, data_min, data_max, scale)
    simulation_targets = manual_inverse_transform(simulation_targets.cpu().numpy(), data_min, data_max, scale)

    return predictions, simulation_targets




# Plot predictions vs true values for the selected simulation
def plot_simulation_results(predictions, targets, output_dir):
    time_steps = np.arange(targets.shape[0])
    num_targets = targets.shape[1]

    for i in range(num_targets):
        plt.figure()
        plt.plot(time_steps, targets[:, i], label="True", linestyle="--")
        plt.plot(time_steps, predictions[:, i], label="Predicted", linestyle="-")
        plt.xlabel("Time Step")
        plt.ylabel(f"Target {i + 1}")
        plt.legend()
        plt.title(f"Simulation Prediction for Target {i + 1}")
        plt.savefig(os.path.join(output_dir, f"simulation_target_{i + 1}.png"))
        plt.close()

# Main execution block
if __name__ == "__main__":
    # User input for scaler choice
    scaler_choice = 'MinMaxScaler'  # Set to 'MinMaxScaler' as per the new requirement

    # User input for activation function choice
    activation_choice = input("Choose activation function (ReLU/Tanh/LeakyReLU): ")
    if activation_choice == 'ReLU':
        activation_function = nn.ReLU()
    elif activation_choice == 'Tanh':
        activation_function = nn.Tanh()
    elif activation_choice == 'LeakyReLU':
        activation_function = nn.LeakyReLU()
    elif activation_choice == 'Softplus':
        activation_function = nn.Softplus()
    elif activation_choice == 'Softmax':
        activation_function = nn.Softmax(dim=1)
    else:
        print("Invalid choice. Defaulting to ReLU activation function.")
        activation_function = nn.ReLU()

    # User input for number of epochs
    num_epochs = int(input("Enter the number of epochs: "))

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the input (features) and output (targets) data
    input_path = 'path/to/input/data'
    output_path = 'path/to/output/data'

    #feature_data = pd.read_csv(input_path, delimiter=',')
    #target_data = pd.read_csv(output_path, delimiter=',')

    #feature_data = np.loadtxt(input_path, delimiter=',')
    #target_data = np.loadtxt(output_path, delimiter=',')

    feature_data, target_data = load_data(input_path, output_path)

    # Normalize both input and output data (excluding the first 4 columns) before splitting
    feature_data_normalized, input_scaling_params = normalize_data(feature_data[:, 4:])
    output_data_normalized, output_scaling_params = normalize_data(target_data[:, 4:])

    print(feature_data_normalized[0])

    feature_data = pd.concat([pd.DataFrame(feature_data[:, :4]), pd.DataFrame(feature_data_normalized)], axis=1)
    target_data = pd.concat([pd.DataFrame(target_data[:, :4]), pd.DataFrame(output_data_normalized)], axis=1)

    # Extract unique IDs and split based on those IDs
    unique_ids = feature_data.iloc[:, 0].unique()  # Assuming first column is the ID
    ids = feature_data.iloc[:, 0].values
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=24)

    # Use the IDs to filter out the respective training and validation data for inputs and outputs
    train_input_data = feature_data[feature_data.iloc[:, 0].isin(train_ids)]
    val_input_data = feature_data[feature_data.iloc[:, 0].isin(val_ids)]

    train_target_data = target_data[target_data.iloc[:, 0].isin(train_ids)]
    val_target_data = target_data[target_data.iloc[:, 0].isin(val_ids)]

    # Normalize only the relevant columns (excluding the ID) for both inputs and outputs
    input_train = train_input_data.iloc[:, 4:].values  # Assuming features start from the 5th column
    output_train = train_target_data.iloc[:, 4:].values  # Assuming the next 6 columns are the outputs

    input_val = val_input_data.iloc[:, 4:].values
    output_val = val_target_data.iloc[:, 4:].values

    #Saving the input and output data to a file
    feature_train_path = 'path/to/feature_train/data'
    feature_val_path = 'path/to/feature_val/data'
    target_train_path = 'path/to/target_train/data'
    target_val_path = 'path/to/target_val/data'

    format_str = '%.8f'
    np.savetxt(feature_train_path, train_input_data, delimiter=',', fmt=format_str)
    np.savetxt(feature_val_path, val_input_data, delimiter=',', fmt=format_str)
    np.savetxt(target_train_path, train_target_data, delimiter=',', fmt=format_str)
    np.savetxt(target_val_path, val_target_data, delimiter=',', fmt=format_str)

    # Create custom datasets using the CustomDataset class
    train_dataset = CustomDataset(torch.tensor(input_train, dtype=torch.float32),
                                  torch.tensor(output_train, dtype=torch.float32))
    val_dataset = CustomDataset(torch.tensor(input_val, dtype=torch.float32),
                                torch.tensor(output_val, dtype=torch.float32))

    # Calculate the covariance matrix
    cov_matrix = np.cov(input_train, rowvar=False)

    # Print the covariance matrix
    print("Covariance Matrix:\n", cov_matrix)

    # Visualize the covariance matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, annot=True, fmt='.2f', cmap='coolwarm', xticklabels=True, yticklabels=True)
    plt.title('Covariance Matrix of Input Features')
    plt.show()

    # User input for learning rate and batch size
    learning_rate = float(input("Enter the learning rate: "))
    batch_size = int(input("Enter the batch size: "))

    # Training the model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = StrainToStressNN(activation_function).to(device)
    criterion = nn.SmoothL1Loss()
    #criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-7)
    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    val_loss, r_squared, true_outputs, predicted_outputs, true_outputs_original, predicted_outputs_original = validate_model(model, val_loader, criterion, output_scaling_params, device)
    print(f'Final Validation Loss: {val_loss:.4f}, R-squared: {r_squared:.4f}')

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()

    plot_directory = 'path/to/plot/directory'
    # Plotting the predicted vs true outputs for each target
    target_names = ['Target1', 'Target2', 'Target3', 'Target4', 'Target5', 'Target6', 'Target7', 'Target8', 'Target9', 'Target10', 'Target11', 'Target12', 'Target13', 'Target14', 'Target15', 'Target16', 'Target17', 'Target18', 'Target19', 'Target20', 'Target21', 'Target22', 'Target23', 'Target24', 'Target25', 'Target26', 'Target27', 'Target28', 'Target29', 'Target30', 'Target31', 'Target32', 'Target33', 'Target34', 'Target35', 'Target36', 'Target37', 'Target38', 'Target39', 'Target40', 'Target41', 'Target42', 'Target43', 'Target44', 'Target45', 'Target46', 'Target47', 'Target48']
    plot_predicted_vs_true(true_outputs_original, predicted_outputs_original, target_names, ids, plot_directory)

    torch.save(model.state_dict(), 'path/to/model.pth')
    #torch.save(model, 'model_PLASTIC_V3.h5')
    joblib.dump(input_scaling_params, 'path/to/input_scaler.pkl')
    #joblib.dump(input_scaler_plastic, 'input_scaler_PLASTIC_plastic_strains.pkl')
    joblib.dump(output_scaling_params, 'path/to/output_scaler.pkl')

    # Save scalers to JSON
    save_scaler_to_json_manual(input_scaling_params, 'path/to/input_scaler_minmax.json')
    save_scaler_to_json_manual(output_scaling_params, 'path/to/output_scaler_minmax.json')

    #save_scaler_to_json(input_scaler_plastic, 'input_scaler_minmax_PLASTIC_plastic_strains.json')

    # Evaluate a random simulation from the validation set
    predictions, true_targets = evaluate_single_simulation(
        model, val_loader, output_scaling_params, device
    )

    # Save and plot the results
    output_directory = "path/to/output/directory"
    os.makedirs(output_directory, exist_ok=True)
    plot_simulation_results(predictions, true_targets, output_directory)

    print("Evaluation and plotting completed for a single simulation.")