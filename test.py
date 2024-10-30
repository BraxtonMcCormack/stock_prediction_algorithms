import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf

# Load the stock data
symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-01-01'
data = yf.download(symbol, start=start_date, end=end_date)

# Prepare the data
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Set parameters for LSTM
sequence_length = 60
x_data, y_data = [], []
for i in range(sequence_length, len(scaled_data)):
    x_data.append(scaled_data[i-sequence_length:i])
    y_data.append(scaled_data[i, 3])  # Using 'Close' price as target

x_data, y_data = np.array(x_data), np.array(y_data)

# Split data into training and testing sets
train_size = int(x_data.shape[0] * 0.8)
x_train, x_test = x_data[:train_size], x_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Predict the test set results
predicted_stock_price = model.predict(x_test)
predicted_stock_price = scaler.inverse_transform(np.concatenate((np.zeros((predicted_stock_price.shape[0], 4)), predicted_stock_price), axis=1))[:, 3]
y_test_actual = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 4)), y_test.reshape(-1, 1)), axis=1))[:, 3]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_actual, predicted_stock_price))
print(f'Root Mean Squared Error: {rmse}')

# Plot the results
plt.figure(figsize=(14, 5))
plt.plot(y_test_actual, color='blue', label='Actual Stock Price')
plt.plot(predicted_stock_price, color='red', label='Predicted Stock Price')
plt.title(f'{symbol} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Pip install command for all required libraries
# Run the following command to install all dependencies:
# !pip install numpy pandas matplotlib scikit-learn tensorflow yfinance
