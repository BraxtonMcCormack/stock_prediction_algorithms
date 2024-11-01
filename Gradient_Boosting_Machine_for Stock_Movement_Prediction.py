import numpy as np
import pandas as pd
import yfinance as yf
import re
from sklearn.metrics import roc_auc_score, f1_score

# Load stock data
def load_data(ticker):
    data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
    return data

# Calculate technical indicators
def add_technical_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['STD_20'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['SMA_20'] + (data['STD_20'] * 2)
    data['Lower_Band'] = data['SMA_20'] - (data['STD_20'] * 2)
    data['MACD'] = data['Close'].ewm(span=12, min_periods=12).mean() - data['Close'].ewm(span=26, min_periods=26).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, min_periods=9).mean()

    # RSI Calculation
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['EMA_50'] = data['Close'].ewm(span=50, min_periods=50).mean()
    return data

# Preprocess data
def preprocess_data(data):
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data = data.dropna()
    features = ['SMA_20', 'Upper_Band', 'Lower_Band', 'MACD', 'Signal_Line', 'RSI', 'EMA_50']
    X = data[features]
    y = data['Target']
    return X, y

# Define a simple GBM model from scratch
class GradientBoostingMachine:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.gammas = []

    def _fit_tree(self, X, residuals, depth=0):
        # Fit a simple decision tree to the residuals
        if depth >= self.max_depth:
            return None
        
        tree = {}
        best_split = None
        best_error = float('inf')
        
        for feature in X.columns:
            for value in X[feature].unique():
                left_indices = X[feature] <= value
                right_indices = X[feature] > value
                left_residuals = residuals[left_indices]
                right_residuals = residuals[right_indices]
                error = (np.sum(left_residuals**2) + np.sum(right_residuals**2))
                if error < best_error:
                    best_error = error
                    best_split = (feature, value)
        
        if best_split is None:
            return None
        
        tree['feature'] = best_split[0]
        tree['value'] = best_split[1]
        tree['left'] = self._fit_tree(X[X[best_split[0]] <= best_split[1]], residuals[X[best_split[0]] <= best_split[1]], depth + 1)
        tree['right'] = self._fit_tree(X[X[best_split[0]] > best_split[1]], residuals[X[best_split[0]] > best_split[1]], depth + 1)
        return tree

    def _predict_tree(self, tree, X):
        if tree is None:
            return np.zeros(len(X))
        
        feature, value = tree['feature'], tree['value']
        left_indices = X[feature] <= value
        right_indices = ~left_indices  # Use negation to define right_indices
        predictions = np.zeros(len(X))
        
        if tree['left'] is not None:
            predictions[left_indices] = self._predict_tree(tree['left'], X[left_indices])
        if tree['right'] is not None:
            predictions[right_indices] = self._predict_tree(tree['right'], X[right_indices])
        
        return predictions

    def fit(self, X, y):
        y_pred = np.zeros(len(y))
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = self._fit_tree(X, residuals)
            if tree is not None:
                gamma = self.learning_rate
                y_pred += gamma * self._predict_tree(tree, X)
                self.trees.append(tree)
                self.gammas.append(gamma)

    def predict(self, X):
        y_pred = np.zeros(len(X))
        for tree, gamma in zip(self.trees, self.gammas):
            y_pred += gamma * self._predict_tree(tree, X)
        return np.where(y_pred > 0, 1, 0)

# Train and evaluate the model
def train_evaluate_model(X, y):
    # Split the data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train the model
    gbm = GradientBoostingMachine(n_estimators=50, learning_rate=0.05, max_depth=3)
    gbm.fit(X_train, y_train)
    
    # Make predictions
    y_pred = gbm.predict(X_test)
    
    # Evaluate the model
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"ROC-AUC Score: {roc_auc}")
    print(f"F1 Score: {f1}")

# Main script
def main():
    data = load_data('AAPL')
    data = add_technical_indicators(data)
    X, y = preprocess_data(data)
    train_evaluate_model(X, y)

if __name__ == "__main__":
    main()
