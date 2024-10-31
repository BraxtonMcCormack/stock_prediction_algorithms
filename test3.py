import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# Step 1: Data Collection
# Downloading historical stock price data using yfinance
def download_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df

stock_data = download_stock_data('^GSPC', start='2015-01-01', end='2023-01-01')

# Check if data is downloaded properly
if stock_data.empty:
    raise ValueError("Failed to download stock data. Please check the ticker symbol or data availability.")

# Step 2: Feature Engineering
# Calculating SMA, MACD, RSI, Bollinger Bands, Stochastic Oscillator
stock_data['SMA'] = stock_data['Close'].rolling(window=15).mean()
stock_data['EMA12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
stock_data['EMA26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
stock_data['MACD'] = stock_data['EMA12'] - stock_data['EMA26']

# RSI Calculation
delta = stock_data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()
rs = average_gain / average_loss
stock_data['RSI'] = 100 - (100 / (1 + rs))

# Bollinger Bands
stock_data['BB_upper'] = stock_data['Close'].rolling(window=20).mean() + (stock_data['Close'].rolling(window=20).std() * 2)
stock_data['BB_lower'] = stock_data['Close'].rolling(window=20).mean() - (stock_data['Close'].rolling(window=20).std() * 2)

# Stochastic Oscillator
low_14 = stock_data['Low'].rolling(window=14).min()
high_14 = stock_data['High'].rolling(window=14).max()
stock_data['Stochastic'] = 100 * (stock_data['Close'] - low_14) / (high_14 - low_14)

# Dropping NaN values
stock_data.dropna(inplace=True)

# Step 3: Data Preprocessing
# Using MinMaxScaler to normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[['Close', 'SMA', 'MACD', 'RSI', 'BB_upper', 'BB_lower', 'Stochastic']])

# Preparing data for LSTM
X = []
y = []
sequence_length = 60
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Splitting the data into training and testing sets
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Step 4: Building the LSTM Model
model = Sequential()
model.add(LSTM(units=80, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))  # Increased units to capture more complexity
model.add(Dropout(0.3))  # Increased dropout to reduce overfitting
model.add(LSTM(units=100, activation='relu', return_sequences=True))  # Added another LSTM layer to increase model depth
model.add(Dropout(0.3))
model.add(LSTM(units=60, activation='relu', return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=50, activation='relu'))  # Added a dense layer to improve feature extraction
model.add(Dense(units=1))

# Compile model
model.compile(optimizer='RMSprop', loss='mean_squared_error')  # Changed optimizer to RMSprop for potentially better performance in time-series

# Step 5: Training the Model
# EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # patience=10: waits for 10 epochs without improvement

model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])  # batch_size=16 may improve learning, epochs=200 for better convergence

# Step 6: Evaluation and Prediction
y_pred = model.predict(X_test)

# Inverse transform to get actual stock prices
y_test_actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), 6))), axis=1))[:, 0]
y_pred_actual = scaler.inverse_transform(np.concatenate((y_pred, np.zeros((len(y_pred), 6))), axis=1))[:, 0]

# Calculate performance metrics
mae = mean_absolute_error(y_test_actual, y_pred_actual)
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred_actual)
accuracy = 100 - (mae / np.mean(y_test_actual) * 100)

# Print performance metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
print(f"Prediction Accuracy: {accuracy:.2f}%")

# Plotting the results
plt.figure(figsize=(14, 5))
plt.plot(y_test_actual, color='blue', label='Actual Stock Price')
plt.plot(y_pred_actual, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Baseline Comparison
baseline_accuracy = 93.0
if accuracy < baseline_accuracy:
    print(f"The model's prediction accuracy ({accuracy:.2f}%) is below the baseline value of {baseline_accuracy}%. Consider improving the model.")
else:
    print(f"The model's prediction accuracy ({accuracy:.2f}%) meets or exceeds the baseline value of {baseline_accuracy}%.")
