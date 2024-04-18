# Command: torchrun --standalone --nproc_per_node=4  lstm-fsdp.py --epochs 3
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributed import init_process_group, get_rank
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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


def main(args):
    init_process_group(backend='nccl')
    rank = get_rank()
    n_gpus = torch.cuda.device_count()
    device_id = rank % n_gpus
    torch.cuda.set_device(device_id)
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    (x_train, y_train), (x_test, y_test), _ = load_data(args.data_path, args.sequence_length)
    train_data_loader = prepare_dataloader(x_train, y_train, args.batch_size)
    
    model = LSTMNet(input_dim=1, hidden_dim=args.hidden_dim, output_dim=1, num_layers=args.num_layers)
    model = FSDP(model).to(device)
    
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

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM with FSDP')
    parser.add_argument('--data_path', type=str, help='Path to the dataset', default='rnn/data/nasdaq_data.csv')
    parser.add_argument('--sequence_length', type=int, default=60, help='Length of the input sequences')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=50, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    args = parser.parse_args()

    main(args)
