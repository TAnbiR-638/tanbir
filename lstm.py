import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model

# Step 1: Load Dataset
data = pd.read_csv("dataset.csv")
values = data['value'].values.reshape(-1, 1)

# Step 2: Preprocess Data
scaler = MinMaxScaler(feature_range=(0, 1))
values_scaled = scaler.fit_transform(values)

def create_dataset(data, time_steps=3):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

time_steps = 3
X, y = create_dataset(values_scaled, time_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM input: [samples, time steps, features]

# Step 3: Build LSTM Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(time_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Step 4: Define Custom Callback
class CustomCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_losses = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch + 1} - Loss: {logs.get('loss'):.4f}")
        self.epoch_losses.append(logs.get('loss'))

# Step 5: Train Model with Custom Callback
callback = CustomCallback()
history = model.fit(
    X, y, 
    epochs=100,  # Increased to 100 epochs
    batch_size=1, 
    verbose=0,  # Suppress default output
    callbacks=[callback]
)

# Step 6: Predict Future Data
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)
y_actual = scaler.inverse_transform(y.reshape(-1, 1))

# Step 7: Calculate Loss for Final Predictions
losses = []
for actual, predicted in zip(y_actual, predictions):
    loss = (actual[0] - predicted[0]) ** 2
    losses.append(loss)

# Step 8: Find the Best Prediction (Minimum Loss) Among Last 100 Predictions
best_loss_idx = np.argmin(losses[-100:]) + len(losses) - 100  # Adjusting index for last 100
best_loss_value = losses[best_loss_idx]
best_loss_actual = y_actual[best_loss_idx][0]
best_loss_predicted = predictions[best_loss_idx][0]

print("\nBest Final Prediction Among Last 100 Predictions:")
print(f"Index: {best_loss_idx}")
print(f"Actual Value: {best_loss_actual:.4f}")
print(f"Predicted Value: {best_loss_predicted:.4f}")
print(f"Loss: {best_loss_value:.4f}")

# Step 9: Visualize Training Loss Over Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(callback.epoch_losses) + 1), callback.epoch_losses, label='Epoch Loss', color='blue')
plt.title('Epoch-wise Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# Step 10: Visualization of Actual vs Predicted Data
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_actual)), y_actual, label='Actual', color='blue')
plt.plot(range(len(predictions)), predictions, label='Predicted', color='red')
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.title("LSTM Time Series Prediction")
plt.legend()
plt.grid()
plt.show()

# Step 11: Output Final Results
print("\nFinal Predictions and Losses:")
for i, (actual, predicted, loss) in enumerate(zip(y_actual, predictions, losses)):
    print(f"Index {i} - Actual: {actual[0]:.4f}, Predicted: {predicted[0]:.4f}, Loss: {loss:.4f}")

# Step 12: Model Visualization
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
print("\nModel architecture saved as 'model_architecture.png'")

