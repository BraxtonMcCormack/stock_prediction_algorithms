import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf

# Create output directories if they do not exist
os.makedirs('outputs/graphs', exist_ok=True)
os.makedirs('outputs/csv', exist_ok=True)

# Load the dataset
# Replace 'AAPL' with the desired stock ticker symbol
stock_symbol = 'AAPL'
data = yf.download(stock_symbol, start='2010-01-01', end='2024-10-31')

# Prepare the data
# Using Close price for prediction
df = data[['Close']]
# df = data[['Open', 'High', 'Low', 'Close', 'Volume']]


# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df)

# Split the data into training and testing sets
train_size = int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - train_size
train_data, test_data = data_scaled[0:train_size, :], data_scaled[train_size:len(data_scaled), :]

# Create a function to prepare the dataset for LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Set the time step
# The time step determines how many previous time units are used to predict the next value.
# Increasing this value allows the model to capture more long-term dependencies but can increase complexity and risk overfitting.
time_step = 120

# Prepare the training and testing datasets
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))  # This LSTM layer has 50 units, which determines the model's capacity to learn complex patterns. Increasing the units may improve the model's ability to learn, but also increases computation and risk of overfitting.
model.add(Dropout(0.2))  # Dropout is used to prevent overfitting by randomly ignoring a fraction of neurons during training. Increasing this value (e.g., to 0.5) provides stronger regularization, while decreasing it allows the model to learn more but may lead to overfitting.
model.add(LSTM(50, return_sequences=True))  # This LSTM layer has 50 units, which determines the model's capacity to learn complex patterns. Increasing the units may improve the model's ability to learn, but also increases computation and risk of overfitting.
model.add(Dropout(0.2))  # Dropout is used to prevent overfitting by randomly ignoring a fraction of neurons during training. Increasing this value (e.g., to 0.5) provides stronger regularization, while decreasing it allows the model to learn more but may lead to overfitting.
model.add(LSTM(50))  # This LSTM layer has 50 units, which determines the model's capacity to learn complex patterns. Increasing the units may improve the model's ability to learn, but also increases computation and risk of overfitting.
model.add(Dropout(0.2))  # Dropout is used to prevent overfitting by randomly ignoring a fraction of neurons during training. Increasing this value (e.g., to 0.5) provides stronger regularization, while decreasing it allows the model to learn more but may lead to overfitting.
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')  # The optimizer determines how the model adjusts its weights based on the loss function. Adam is adaptive and generally works well for most cases, but other optimizers like SGD may sometimes lead to better generalization.

# Train the model
# Epochs determine how many times the model will iterate over the entire training dataset.
# Increasing the epochs can lead to better learning but also increases the risk of overfitting if too high.
# The batch size determines how many samples are used to update the model weights in each iteration.
# A smaller batch size may lead to better generalization but slower training, while a larger batch size speeds up training but may reduce generalization.
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predict the stock prices
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_inv = scaler.inverse_transform([y_train])
y_test_inv = scaler.inverse_transform([y_test])

# Calculate RMSE for evaluation
train_rmse = np.sqrt(mean_squared_error(y_train_inv[0], train_predict[:, 0]))
test_rmse = np.sqrt(mean_squared_error(y_test_inv[0], test_predict[:, 0]))
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(df.index[time_step:len(train_predict) + time_step], train_predict, label='Train Predictions')
plt.plot(df.index[len(train_predict) + (time_step * 2) + 1:len(data_scaled) - 1], test_predict, label='Test Predictions')
plt.plot(df.index, scaler.inverse_transform(data_scaled), label='Original Data')
plt.legend()
plt.savefig('outputs/graphs/train_test_predictions.png')
plt.show()

# Save predictions to CSV
valid_index = min(len(train_predict), len(test_predict))
train_test_results = pd.DataFrame({
    'Date': df.index[time_step + 1:time_step + 1 + valid_index],
    'Train Predictions': train_predict[:valid_index, 0],
    'Test Predictions': test_predict[:valid_index, 0],
    'Original Data': scaler.inverse_transform(data_scaled)[time_step + 1:time_step + 1 + valid_index, 0]
})
train_test_results.to_csv('outputs/csv/train_test_predictions.csv', index=False)

# Future Prediction
# Number of future days to predict
future_days = 30

# Use the last `time_step` days from the dataset for predicting the future
future_input = data_scaled[-time_step:].reshape(1, time_step, 1)

# Store future predictions
future_predictions = []

# Predict future days iteratively
for _ in range(future_days):
    # Make prediction
    future_pred = model.predict(future_input)
    
    # Store the predicted value
    future_predictions.append(future_pred[0, 0])
    
    # Append the predicted value to the input sequence for the next prediction
    future_input = np.append(future_input[:, 1:, :], [[[future_pred[0, 0]]]], axis=1)

# Inverse transform the predictions to original scale
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create date index for future predictions
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, periods=future_days + 1, inclusive='right')

# Plotting future predictions
plt.figure(figsize=(14, 5))
plt.plot(df.index, scaler.inverse_transform(data_scaled), label='Original Data')
plt.plot(future_dates, future_predictions, label='Future Predictions', color='red')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'Future {future_days} Day Stock Price Prediction')
plt.savefig('outputs/graphs/future_predictions.png')
plt.show()

# Save future predictions to CSV
future_results = pd.DataFrame({
    'Date': future_dates,
    'Future Predictions': future_predictions.flatten()
})
future_results.to_csv('outputs/csv/future_predictions.csv', index=False)
