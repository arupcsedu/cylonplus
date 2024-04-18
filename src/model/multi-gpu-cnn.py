#Arup Sarker: DDP implement with multiprocess by using spawn on CNN
#Command: python multi-gpu-cnn.py
from __future__ import print_function
from cloudmesh.common.StopWatch import StopWatch
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
#from PyTorchLoader import IterableParquetDataset, IterableManualParquetDataset
import os
from examples.mnist import DEFAULT_MNIST_DATA_PATH
from petastorm import make_reader, TransformSpec
from petastorm import make_batch_reader
#from PetastormDataLoader import TransformersDataLoader
from petastorm.pytorch import DataLoader


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, rank, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('GPU: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                rank, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, rank, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            #data, target = data['image'].to(device), data['digit'].to(device)  # For fixing build 
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nGPU: {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        rank, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(rank: int, world_size: int, args, device, train_loader, test_loader):
    ddp_setup(rank, world_size)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, train_loader, optimizer, epoch)
        test(model, rank, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    
    destroy_process_group()

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def _transform_row(mnist_row):
    # For this example, the images are stored as simpler ndarray (28,28), but the
    # training network expects 3-dim images, hence the additional lambda transform.
    transform = transforms.Compose([
        transforms.Lambda(lambda nd: nd.reshape(28, 28, 1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # In addition, the petastorm pytorch DataLoader does not distinguish the notion of
    # data or target transform, but that actually gives the user more flexibility
    # to make the desired partial transform, as shown here.
    result_row = {
        'image': transform(mnist_row['image']),
        'digit': mnist_row['digit']
    }

    return result_row


if __name__ == '__main__':
    # Training settings
    StopWatch.start("initialize")
    DEFAULT_MNIST_DATA_PATH = '/tmp/mnist'

    default_dataset_url = 'file://{}'.format(DEFAULT_MNIST_DATA_PATH)
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--dataset-url', type=str, default=default_dataset_url, metavar='S', help='hdfs:// or file:/// URL to the MNIST petastorm dataset (default: %s)' % default_dataset_url)
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    world_size = torch.cuda.device_count()
    print(f"World Size:  {world_size}")
    StopWatch.stop("initialize")



    #dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    #dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    
    """
    transform = TransformSpec(_transform_row, removed_fields=['idx'])
    train_loader = DataLoader(make_reader('{}/train'.format(args.dataset_url), 
                                    num_epochs=args.epochs, 
                                    transform_spec=transform, seed=args.seed, shuffle_rows=True),
                                    batch_size=args.batch_size)
    test_loader = DataLoader(make_reader('{}/test'.format(args.dataset_url),
                                    num_epochs=args.epochs,
                                    transform_spec=transform), 
                                    batch_size=args.batch_size)
    """
  
    
   
    
    #dataset1 = IterableParquetDataset('{}/train'.format(args.dataset_url), process_rows)
    #dataset2 = IterableParquetDataset('{}/test'.format(args.dataset_url),  process_rows)

    #dataloader = DataLoader(dataset, num_workers=4)


    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    
    if use_cuda:
        train_cuda_kwargs = {'num_workers': world_size,
                       'pin_memory': True,
                       'shuffle': True}
        test_cuda_kwargs = {'num_workers': world_size,
                       'pin_memory': True,
                       'shuffle': True}

        train_kwargs.update(train_cuda_kwargs)
        test_kwargs.update(test_cuda_kwargs)
    
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
  
    
    StopWatch.start("spawn")

    mp.spawn(main, args=(world_size, args, device, train_loader, test_loader), nprocs=world_size)

    StopWatch.stop("spawn")

    StopWatch.benchmark()