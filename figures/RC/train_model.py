import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

plt.ion()  # Turn on interactive mode

class LivePlot(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 5))
        self.epochs = []
        self.mse = []
        self.val_mse = []
        self.mae = []
        self.val_mae = []
        
    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.mse.append(logs.get('loss'))
        self.val_mse.append(logs.get('val_loss'))
        self.mae.append(logs.get('mae'))
        self.val_mae.append(logs.get('val_mae'))
        self.draw()
        
    def draw(self):
        # Clear previous plot
        for ax in self.axs:
            ax.clear()
        
        # Plot MSE
        self.axs[0].plot(self.epochs, self.mse, label='Training MSE')
        self.axs[0].plot(self.epochs, self.val_mse, label='Validation MSE')
        self.axs[0].set_title('MSE Over Epochs')
        self.axs[0].set_xlabel('Epoch')
        self.axs[0].set_ylabel('MSE')
        self.axs[0].legend()
        
        # Plot MAE
        self.axs[1].plot(self.epochs, self.mae, label='Training MAE')
        self.axs[1].plot(self.epochs, self.val_mae, label='Validation MAE')
        self.axs[1].set_title('MAE Over Epochs')
        self.axs[1].set_xlabel('Epoch')
        self.axs[1].set_ylabel('MAE')
        self.axs[1].legend()
        
        # Use IPython display utilities
        clear_output(wait=True)
        display(self.fig)
        plt.pause(0.001)  # Pause to ensure the plot updates

# Load datasets
train_data = np.load('datasets/train_data.npy')
val_data = np.load('datasets/val_data.npy')
test_data = np.load('datasets/test_data.npy')

# Extract features and labels
X_train, y_train = train_data[:, :2], train_data[:, 2]
X_val, y_val = val_data[:, :2], val_data[:, 2]
X_test, y_test = test_data[:, :2], test_data[:, 2]

# Define a list of neural network architectures to iterate over
# Each architecture is defined by a tuple of layer sizes
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


early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Directory for saving models
model_save_dir = 'saved_models'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

training_histories = {}

best_model = None
best_model_path = ''
lowest_mse = np.inf
lowest_mae = np.inf

for architecture in network_architectures:
    # Create model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(architecture[0], activation='relu', input_shape=(2,)))
    for units in architecture[1:]:
        model.add(keras.layers.Dense(units, activation='relu'))
    model.add(keras.layers.Dense(1))  # Output layer
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Model summary
    model.summary()
    
    # Train model
    print(f"Training model with architecture: {architecture}")
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping, LivePlot()], verbose=2)

    # Save training history
    training_histories[str(architecture)] = history

    # Save model
    # Updated to use the native Keras format:
    model_path = os.path.join(model_save_dir, f'model_architecture_{architecture}.keras')
    model.save(model_path)
    print(f"Model saved at {model_path}")
    
    # Evaluate on the testing set
    test_metrics = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_mae = test_metrics[0], test_metrics[1]
    print(f"Test Loss for architecture {architecture}: {test_loss}")
    print(f"Test MAE for architecture {architecture}: {test_mae}")

    # Predict and calculate R-squared
    y_pred = model.predict(X_test).flatten()
    r_squared = r2_score(y_test, y_pred)
    print(f"R-squared for architecture {architecture}: {r_squared}\n")

    # Load a new dataset for final evaluation
new_test_data = np.load('datasets/final_test_data.npy')
X_new_test, y_new_test = new_test_data[:, :2], new_test_data[:, 2]

plt.ioff()

# Iterate over trained models to evaluate on the new dataset
for architecture, history in training_histories.items():
    # Load the model if saved or use the trained model
    model = keras.models.load_model(f'saved_models/model_architecture_{architecture}.keras')  # Assuming models were saved
    
    # Evaluate model on the new dataset
    new_test_metrics = model.evaluate(X_new_test, y_new_test, verbose=0)
    new_test_mse, new_test_mae = new_test_metrics[0], new_test_metrics[1]
    print(f"New Test MSE for architecture {architecture}: {new_test_mse}, New Test MAE for architecture {architecture}: {new_test_mae}")

    # Check if this model has the lowest MSE so far
    if new_test_mse < lowest_mse or (new_test_mse == lowest_mse and new_test_mae < lowest_mae):
        lowest_mse = new_test_mse
        lowest_mae = new_test_mae
        best_model_path = model_path

# After identifying the best model, load it
if best_model_path:
    best_model = keras.models.load_model(best_model_path)
    print(f"The best model is saved at: {best_model_path}")
    print(f"Best Model MSE: {lowest_mse}, Best Model MAE: {lowest_mae}")

    y_pred = best_model.predict(X_new_test).flatten()

    # Scatter plot for actual vs predicted values
    plt.scatter(y_new_test, y_pred, alpha=0.5)
    plt.title('Actual vs. Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([y_new_test.min(), y_new_test.max()], [y_new_test.min(), y_new_test.max()], 'k--', lw=3)
    plt.show()

    # Time series plot for actual and predicted (for a selected segment or full series if feasible)
    plt.figure(figsize=(10, 6))
    plt.plot(y_new_test[:100], label='Actual', marker='.')
    plt.plot(y_pred[:100], label='Predicted', marker='x')
    plt.title('Comparison of Actual and Predicted Values')
    plt.xlabel('Time Step')
    plt.ylabel('Output')
    plt.legend()
    plt.show()
else:
    print("No best model identified.")