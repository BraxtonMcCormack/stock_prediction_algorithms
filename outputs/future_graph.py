import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data from CSV file
csv_file = 'outputs\\csv\\future_predictions.csv'

# Get the current working directory to verify where the script is running
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

# Get the absolute path of the CSV file
csv_file_path = os.path.join(current_directory, csv_file)
print(f"Absolute path of the CSV file: {csv_file_path}")

# Check if the CSV file exists at the absolute path
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"The file '{csv_file}' was not found at the location: {csv_file_path}")

data = pd.read_csv(csv_file_path)

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Plot the future predictions over time
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Future Predictions'], marker='o', linestyle='-', color='b')
plt.title('Future Predictions Over Time')
plt.xlabel('Date')
plt.ylabel('Future Predictions')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the rate of change (first derivative) over time
data['Rate of Change'] = data['Future Predictions'].diff()
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Rate of Change'], marker='o', linestyle='-', color='r')
plt.title('Rate of Change of Future Predictions')
plt.xlabel('Date')
plt.ylabel('Rate of Change')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the cumulative sum of future predictions
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Future Predictions'].cumsum(), marker='o', linestyle='-', color='g')
plt.title('Cumulative Sum of Future Predictions')
plt.xlabel('Date')
plt.ylabel('Cumulative Sum of Predictions')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
