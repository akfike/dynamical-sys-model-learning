import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define PyTorch model
class NeuralNetwork(nn.Module):
    def __init__(self, architecture):
        super(NeuralNetwork, self).__init__()
        layers = []
        input_size = 3  # Assuming 3 input features as in your Keras example
        for units in architecture:
            layers.append(nn.Linear(input_size, units))
            layers.append(nn.ReLU())
            input_size = units  # Update input size for the next layer
        layers.append(nn.Linear(input_size, 1))  # Output layer
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Prepare datasets
def prepare_datasets(X_train, y_train, X_val, y_val, X_test, y_test):
    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
    val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).float())
    test_ds = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())
    return train_ds, val_ds, test_ds

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            outputs = model(inputs)
            predictions.extend(outputs.view(-1).cpu().numpy())
            actuals.extend(targets.view(-1).cpu().numpy())
    mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
    mae = mean_absolute_error(np.array(actuals), np.array(predictions))
    return mse, mae

# Load datasets using NumPy (paths need to be set according to your dataset locations)
train_data = np.load('datasets/RLC/train_data_rlc.npy')
val_data = np.load('datasets/RLC/val_data_rlc.npy')
test_data = np.load('datasets/RLC/test_data_rlc.npy')
new_test_data = np.load('datasets/RLC/final_test_data_RLC.npy')

# Split the datasets into features (X) and target (y)
X_train, y_train = train_data[:, :3], train_data[:, 3]
X_val, y_val = val_data[:, :3], val_data[:, 3]
X_test, y_test = test_data[:, :3], test_data[:, 3]
X_new_test, y_new_test = new_test_data[:, :3], new_test_data[:, 3]

# Prepare datasets for PyTorch by converting them into TensorDatasets
train_ds, val_ds, test_ds = prepare_datasets(X_train, y_train, X_val, y_val, X_test, y_test)
new_test_ds = TensorDataset(torch.tensor(X_new_test).float(), torch.tensor(y_new_test).float())

# Create DataLoader objects for each dataset
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)
new_test_loader = DataLoader(new_test_ds, batch_size=32)

# Training function adjusted for end-of-training plotting
def train_model(model, architecture, train_loader, val_loader, optimizer, criterion, epochs=100, patience=10):
    best_val_loss = np.inf
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'architecture': architecture
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Plotting after training is complete
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss for Architecture: {architecture}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Define your network architectures
network_architectures = [
    (16,),                # Single layer with fewer neurons
    (32,),                # Single layer with more neurons
    (16, 16),             # Two layers, fewer neurons
    (32, 32),             # Two layers, moderate neurons
    (64,),                # Single layer, more neurons to capture potential non-linearities
    (16, 32, 16),         # Three layers, with a bottleneck
    (32, 64, 32),         # Three layers, more capacity than above
    (64, 32),             # Two layers, reducing complexity from original architectures
]

best_mse = np.inf
best_mae = np.inf

for architecture in network_architectures:
    model = NeuralNetwork(architecture).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    print(f"Training model with architecture: {architecture}")
    train_model(model, architecture, train_loader, val_loader, optimizer, criterion)
    # Load best model for evaluation to ensure the correct architecture is used
    checkpoint = torch.load('best_model.pth')
    model = NeuralNetwork(checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    mse, mae = evaluate_model(model, test_loader)
    print(f"Test MSE: {mse}, Test MAE: {mae} for architecture {architecture}")
    
    if mse < best_mse or (mse == best_mse and mae < best_mae):
        best_mse = mse
        best_mae = mae
        best_architecture = architecture

# Assuming best_model.pth has the best model, load it for final evaluation
checkpoint = torch.load('best_model.pth')
best_model = NeuralNetwork(checkpoint['architecture']).to(device)
best_model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on new test data
new_test_mse, new_test_mae = evaluate_model(best_model, new_test_loader)
print(f"New Test MSE: {new_test_mse}, New Test MAE: {new_test_mae} for the best architecture {best_architecture}")

# Plotting
def plot_predictions(model, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.view(-1).cpu().numpy())
            actuals.extend(targets.numpy())
    
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.title('Actual vs. Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'k--', lw=3)  # Diagonal line
    plt.show()
    
    # Time series plot
    plt.figure(figsize=(10, 6))
    plt.plot(actuals[:100], label='Actual', marker='.')
    plt.plot(predictions[:100], label='Predicted', marker='x')
    plt.title('Comparison of Actual and Predicted Values (First 100 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


plot_predictions(best_model, new_test_loader)
