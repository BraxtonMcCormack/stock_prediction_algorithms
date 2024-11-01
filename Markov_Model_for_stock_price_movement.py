import pandas as pd
import numpy as np
import yfinance as yf
import random
import matplotlib.pyplot as plt

# Fetch historical data for a specific stock
def get_historical_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data['Daily Change'] = data['Adj Close'].pct_change()
    return data

# Classify daily changes into 'up', 'down', or 'no change'
def classify_changes(data):
    conditions = [
        (data['Daily Change'] > 0.001),
        (data['Daily Change'] < -0.001),
        (abs(data['Daily Change']) <= 0.001)
    ]
    choices = ['up', 'down', 'no change']
    data['Movement'] = np.select(conditions, choices, default='no change')
    return data

# Calculate the transition matrix
def calculate_transition_matrix(movements):
    states = ['up', 'down', 'no change']
    matrix = pd.DataFrame(0, index=states, columns=states, dtype='float')
    
    for (prev, curr) in zip(movements[:-1], movements[1:]):
        matrix.loc[prev, curr] += 1

    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    return matrix

# Predict the next state based on the current state and transition matrix, and provide trading signals
def predict_next_state_and_signal(current_state, transition_matrix, num_predictions=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    states = list(transition_matrix.columns)
    predictions = []
    signals = []

    for i in range(num_predictions):
        if current_state not in transition_matrix.index:
            current_state = random.choice(states)
        probabilities = transition_matrix.loc[current_state].fillna(0)
        next_state = np.random.choice(states, p=probabilities)
        
        predictions.append(f"Day {i+1}: {next_state}")
        
        # Buy/sell signals based on prediction
        if next_state == 'up':
            signals.append(f"Day {i+1}: Buy Signal")
        elif next_state == 'down':
            signals.append(f"Day {i+1}: Sell Signal")
        else:
            signals.append(f"Day {i+1}: Hold")

        current_state = next_state
    
    return predictions, signals

# Main function to train the Markov Chain, make predictions, and simulate trading
def main(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    ticker = 'MSFT'  # Microsoft Stock
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    # Step 1: Get Historical Data
    data = get_historical_data(ticker, start_date, end_date)
    
    # Step 2: Classify Daily Movements
    data = classify_changes(data)
    
    # Step 3: Calculate Transition Matrix
    movements = data['Movement'].dropna().tolist()
    transition_matrix = calculate_transition_matrix(movements)
    
    # Step 4: Predict Future Movements and Trading Signals
    current_state = movements[-1]  # Start from the last known state instead of a random one
    predictions, signals = predict_next_state_and_signal(current_state, transition_matrix, num_predictions=10, seed=seed)
    
    # Output Results
    print("Transition Matrix:\n")
    print(transition_matrix)
    print("\nPredicted Movements:\n")
    for prediction in predictions:
        print(prediction)
    print("\nTrading Signals:\n")
    for signal in signals:
        print(signal)

if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 42
    main()
