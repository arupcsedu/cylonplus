#install,
# conda install matplotlib
# conda install scikit-learn
# Command: torchrun --standalone --nproc_per_node=1  lstm-multi-gpu-torchrun.py --epochs 3 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import argparse
import os
import matplotlib.pyplot as plt

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_data(file_name, sequence_length):
    data = pd.read_csv(file_name)
    prices = data['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices = scaler.fit_transform(prices)

    result = []
    for index in range(len(prices) - sequence_length):
        result.append(prices[index: index + sequence_length])
    result = np.array(result)
    train_size = int(len(result) * 0.9)
    train = result[:train_size, :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[train_size:, :-1]
    y_test = result[train_size:, -1]

    return (x_train, y_train), (x_test, y_test), scaler

def prepare_dataloader(x, y, batch_size):
    tensor_x = torch.Tensor(x)
    tensor_y = torch.Tensor(y)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def plot_predictions(model, test_data_loader, scaler, device):
    model.eval() 
    actual_prices = []
    predicted_prices = []
    
    with torch.no_grad():
     for data, targets in test_data_loader:
         data, targets = data.to(device), targets.to(device)
         predictions = model(data)
         actual_prices.extend(scaler.inverse_transform(targets.cpu().numpy()))
         predicted_prices.extend(scaler.inverse_transform(predictions.cpu().numpy()))
    
    actual_prices = np.array(actual_prices)
    predicted_prices = np.array(predicted_prices)
    
    plt.figure(figsize=(20,10))
    plt.plot(actual_prices, label='Actual Prices', color='blue')
    plt.plot(predicted_prices, label='Predicted Prices', color='red')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Prices')
    plt.legend()
    plt.show()
    plt.savefig('prediction_vs_actual.png')

def main(args):
    (x_train, y_train), (x_test, y_test), _ = load_data("rnn/data/nasdaq_data.csv", args.sequence_length)
    train_data_loader = prepare_dataloader(x_train, y_train, args.batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LSTMNet(input_dim=1, hidden_dim=args.hidden_dim, output_dim=1, num_layers=args.num_layers).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_data_loader.dataset)} ({100. * batch_idx / len(train_data_loader):.0f}%)] \tLoss: {loss.item():.6f}')

        print(f'Epoch {epoch} Average Loss: {total_loss / len(train_data_loader):.6f}')
    (x_test, y_test), _, scaler = load_data("rnn/data/nasdaq_data.csv", args.sequence_length)
    test_data_loader = prepare_dataloader(x_test, y_test, args.batch_size)
    plot_predictions(model, test_data_loader, scaler, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM for NASDAQ Data')
    parser.add_argument('--sequence_length', type=int, default=60, help='Length of the input sequences')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=50, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    args = parser.parse_args()

    main(args)
