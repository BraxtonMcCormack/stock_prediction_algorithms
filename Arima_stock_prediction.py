import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import itertools
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

# Suppress warnings
warnings.filterwarnings('ignore')

# Step 1: Download historical stock price data
def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data = data.asfreq('B')  # Set frequency to business days
    data = data.fillna(method='ffill')  # Fill any missing values with forward fill
    data = data.fillna(method='bfill')  # Backfill if any NaNs are left at the start
    return data['Close']

# Step 2: Check Stationarity
def check_stationarity(time_series):
    result = adfuller(time_series)
    print("\nAugmented Dickey-Fuller Test:")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] > 0.05:
        print("Warning: Data is likely non-stationary. Consider additional differencing.")
    else:
        print("Data appears to be stationary.")

# Step 3: Select best (p, d, q) parameters
def select_best_arima_params(time_series, p_range, d_range, q_range):
    best_aic = np.inf
    best_order = None
    best_model = None
    
    pdq = list(itertools.product(p_range, d_range, q_range))
    
    for order in pdq:
        try:
            model = ARIMA(time_series, order=order)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = order
                best_model = model_fit
        except Exception as e:
            print(f"Failed to fit ARIMA order {order}: {e}")
            continue
    
    return best_order, best_model

# Step 4: Train the ARIMA model and make predictions
def train_arima_model(time_series, p, d, q):
    model = ARIMA(time_series, order=(p, d, q))
    model_fit = model.fit()
    return model_fit

# Step 5: Visualize actual vs predicted values
def visualize_forecast(actual, forecast, train_size, conf_int=None):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Prices', color='blue')
    plt.plot(forecast, label='Predicted Prices', color='red')
    plt.axvline(x=actual.index[train_size], color='black', linestyle='--', label='Train-Test Split')
    if conf_int is not None:
        plt.fill_between(forecast.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3, label='95% Confidence Interval')
    plt.title('Stock Price Prediction using ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Step 6: Print Model Summary and Performance Metrics
def evaluate_model(model_fit, test_data, forecast):
    print("\nModel Summary:")
    print(model_fit.summary())
    
    # Drop NaN values in test_data and forecast
    test_data = test_data.dropna()
    forecast = forecast[:len(test_data)]
    
    # Calculate and print MAE, RMSE, MAPE
    mae = mean_absolute_error(test_data, forecast)
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100
    print(f"\nModel Performance Metrics:\nMean Absolute Error (MAE): {mae:.2f}\nRoot Mean Squared Error (RMSE): {rmse:.2f}\nMean Absolute Percentage Error (MAPE): {mape:.2f}%")

# Step 7: Residual Analysis
def residual_analysis(model_fit):
    residuals = model_fit.resid
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plt.plot(residuals)
    plt.title('Residuals Over Time')
    plt.subplot(212)
    sm.graphics.tsa.plot_acf(residuals, lags=40, ax=plt.gca())
    plt.title('ACF of Residuals')
    plt.tight_layout()
    plt.show()
    
    # QQ-Plot
    sm.qqplot(residuals, line='s')
    plt.title('QQ Plot of Residuals')
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Step 1: Load Data
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    stock_data = get_stock_data(ticker, start=start_date, end=end_date)

    # Step 2: Check Stationarity
    check_stationarity(stock_data)

    # Step 3: Split Data into Training and Testing
    train_size = int(len(stock_data) * 0.8)
    train_data, test_data = stock_data[:train_size], stock_data[train_size:]

    # Step 4: Find Best ARIMA Parameters
    p_range = range(0, 4)
    d_range = range(0, 2)
    q_range = range(0, 4)
    best_order, best_model = select_best_arima_params(train_data, p_range, d_range, q_range)
    print(f"Best ARIMA order: {best_order}")

    # Step 5: Train ARIMA Model on Full Training Data
    p, d, q = best_order
    model_fit = train_arima_model(train_data, p, d, q)

    # Step 6: Make Forecasts with Confidence Intervals
    forecast_result = model_fit.get_forecast(steps=len(test_data))
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    
    # Step 7: Visualize Results
    visualize_forecast(stock_data, pd.Series(forecast, index=test_data.index), train_size, conf_int=conf_int)
    
    # Step 8: Evaluate Model
    evaluate_model(model_fit, test_data, forecast)
    
    # Step 9: Residual Analysis
    residual_analysis(model_fit)
