import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Load stock data using Yahoo Finance
# Example: Get historical data for a specific stock, e.g., 'AAPL'
stock_symbol = 'AAPL'
data = yf.download(stock_symbol, start='2020-01-01', end='2023-01-01')

# Create technical indicators (SMA, EMA, RSI, MACD, ADX)
data['SMA'] = data['Close'].rolling(window=20).mean()
data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
data['RSI'] = 100 - (100 / (1 + (data['Close'].diff().clip(lower=0).rolling(window=14).mean() / \
                                  -data['Close'].diff().clip(upper=0).rolling(window=14).mean())))
data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
data['ADX'] = 100 * (data['High'] - data['Low']).rolling(window=14).mean() / data['Close'].rolling(window=14).mean()

# Drop NaN values that were created during indicator calculations
df = data.dropna()

# Generate target labels ('up' or 'down')
df['Movement'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Define feature columns and target column
feature_columns = ['SMA', 'EMA', 'RSI', 'MACD', 'ADX']  # Replace with actual feature names
target_column = 'Movement'

X = df[feature_columns]
y = df[target_column]

# Address class imbalance using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the RandomForest model as an alternative to SVM
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(rf, param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

# Get the best estimator
best_rf = grid.best_estimator_

# Make predictions
y_pred = best_rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Save the model for future use
import joblib
joblib.dump(best_rf, 'rf_stock_model.pkl')
