import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import argparse
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


def load_and_process_data(filename, sequence_length):
    df = pd.read_csv(filename)
    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    x = []
    y = []
    for i in range(sequence_length, len(data_normalized)):
        x.append(data_normalized[i - sequence_length:i])
        y.append(data_normalized[i, 3])  #predict close price
    x, y = np.array(x), np.array(y)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1), scaler


def create_dataloaders(x, y, batch_size):
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def plot_predictions(model, device, test_loader, scaler):
    model.eval()
    predicted_prices = []
    actual_prices = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data).cpu().numpy()
            predicted_prices.extend(output)
            actual_prices.extend(target.cpu().numpy())

    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    actual_prices = np.array(actual_prices).reshape(-1, 1)
    dummy_predicted = np.zeros((predicted_prices.shape[0], 5))
    dummy_actual = np.zeros((actual_prices.shape[0], 5))
    dummy_predicted[:, 3] = predicted_prices.flatten()
    dummy_actual[:, 3] = actual_prices.flatten()
    predicted_prices_transformed = scaler.inverse_transform(dummy_predicted)[:, 3]
    actual_prices_transformed = scaler.inverse_transform(dummy_actual)[:, 3]
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices_transformed, color='blue', label='Actual NASDAQ Close Price')
    plt.plot(predicted_prices_transformed, color='red', linestyle='--', label='Predicted NASDAQ Close Price')
    plt.title('NASDAQ Close Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('NASDAQ Close Price')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='PyTorch LSTM for NASDAQ Prediction')
    parser.add_argument('--sequence-length', type=int, default=50,
                        help='sequence length for input data (default: 50)')
    parser.add_argument('--input-dim', type=int, default=5,
                        help='input dimension size (default: 5)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='number of hidden units in LSTM (default: 128)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of LSTM layers (default: 2)')
    parser.add_argument('--output-dim', type=int, default=1,
                        help='output dimension (default: 1)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    x, y, scaler = load_and_process_data("rnn/data/nasdaq_data.csv", args.sequence_length)
    split_idx = int(len(x) * 0.8)
    x_train, y_train = x[:split_idx], y[:split_idx]
    x_test, y_test = x[split_idx:], y[split_idx:]

    train_loader = create_dataloaders(x_train, y_train, args.batch_size)
    test_loader = create_dataloaders(x_test, y_test, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMNet(args.input_dim, args.hidden_dim, args.num_layers, args.output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
    plot_predictions(model, device, test_loader, scaler)


if __name__ == "__main__":
    main()

