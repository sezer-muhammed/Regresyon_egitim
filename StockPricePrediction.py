import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import argparse

# Define argparse parameters
parser = argparse.ArgumentParser(description='Stock Price Prediction with Regression')
parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol to be predicted')
parser.add_argument('--start', type=str, default='2010-01-01', help='Start date of historical data')
parser.add_argument('--end', type=str, default='2023-03-11', help='End date of historical data')
parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data used for training')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
args = parser.parse_args()

# Load the data
data = yf.download(args.symbol, start=args.start, end=args.end)

# Normalize the data
scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data[['Close']])

# Split the data into train and test sets
train_size = int(len(data) * args.train_ratio)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Define the PyTorch model
class Regressor(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        self.fc = nn.Linear(degree+1, 1)

    def forward(self, x):
        x_poly = torch.cat([x.pow(i) for i in range(self.degree+1)], dim=1)
        out = self.fc(x_poly)
        return out


# Initialize the model
model = Regressor(5)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

a = 0
# Train the model
for epoch in range(args.num_epochs):
    inputs = torch.tensor(train_data['Close'].values[:-1], dtype=torch.float32).reshape(-1, 1)
    labels = torch.tensor(train_data['Close'].values[1:], dtype=torch.float32).reshape(-1, 1)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, args.num_epochs, loss.item()))

# Evaluate the model
model.eval()
with torch.no_grad():
    inputs = torch.tensor(test_data['Close'].values[:-1], dtype=torch.float32).reshape(-1, 1)
    labels = torch.tensor(test_data['Close'].values[1:], dtype=torch.float32).reshape(-1, 1)
    outputs = model(inputs)
    mse = criterion(outputs, labels)
    print('Mean Squared Error: {:.4f}'.format(mse.item()))

# Inverse transform the predicted and actual values
predicted_prices = scaler.inverse_transform(outputs.numpy())
actual_prices = scaler.inverse_transform(labels.numpy())
training_prices = scaler.inverse_transform(train_data['Close'].values.reshape(-1, 1))

# Visualize the results
import matplotlib.pyplot as plt
plt.plot(training_prices, label='Training Data')
plt.plot(range(train_size, len(data)-1), actual_prices, label='Actual Price')
plt.plot(range(train_size, len(data)-1), predicted_prices, label='Predicted Price')
plt.title(args.symbol + ' Stock Price Prediction with Regression')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
