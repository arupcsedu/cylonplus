"""
Install: PyCylon (Follow: https://cylondata.org/docs/)
Run Program: mpirun -n 2 python cylon_pytorch_dl.py
"""
import os
import socket
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pycylon import CylonContext
from pycylon import Table, CylonEnv

from pycylon.io import CSVReadOptions
from pycylon.io import read_csv
from pycylon.net import MPIConfig
from torch.nn.parallel import DistributedDataParallel as DDP

hostname = socket.gethostname()


def setup(rank, world_size):
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # initialize the process group
    dist.init_process_group('gloo', init_method="env://", timeout=timedelta(seconds=30))
    print(f"Init Process Groups : => [{hostname}]Demo DDP Rank {rank}")
    

    

    #init_process_group(backend="nccl")
    #torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    dist.destroy_process_group()


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(4, 1)
        self.hidden2 = nn.Linear(1, 16)
        # self.hidden3 = nn.Linear(1024, 10)
        # self.hidden4 = nn.Linear(10, 1)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        # x = F.relu(self.hidden4(x))
        x = self.output(x)
        return x


def demo_basic(rank, world_size):
    print(f"Simple Batch Train => [{hostname}]Demo DDP Rank {rank}")
    setup(rank=rank, world_size=world_size)

    env = CylonEnv(config=MPIConfig(), distributed=True)
    csv_read_options = CSVReadOptions() \
        .use_threads(False) \
        .block_size(1 << 30) \
        .na_values(['na', 'none'])


    rank = dist.get_rank()
    rank = rank + 1
    world_size = dist.get_world_size()
    print(f"World Size:  {world_size}")

    print(f"rank: {rank}")
    csv1 = f"/project/bii_dsc_community/djy8hg/arupcsedu/cylonplus/src/cylon-dl/data/user_usage_tm_{rank}.csv"
    csv2 = f"/project/bii_dsc_community/djy8hg/arupcsedu/cylonplus/src/cylon-dl/data/user_device_tm_{rank}.csv"

    user_devices_data: Table = read_csv(env, csv1, csv_read_options)
    user_usage_data: Table = read_csv(env, csv2, csv_read_options)

    print(user_devices_data)

    first_row_count = user_devices_data.row_count

    second_row_count = user_usage_data.row_count


    print(f"Table 1 & 2 Rows [{first_row_count},{second_row_count}], "
            f"Columns [{user_devices_data.column_count},{user_usage_data.column_count}]")

    configs = {'join_type': 'inner', 'algorithm': 'sort'}
    joined_table: Table = user_devices_data.distributed_join(table=user_usage_data,
                                        join_type=configs['join_type'],
                                        algorithm=configs['algorithm'],
                                        left_on=[3],
                                        right_on=[0]
                                        )


    join_row_count = joined_table.row_count

    print(f"First table had : {first_row_count} and Second table had : {second_row_count}, "
            f"Joined has : {join_row_count}")

    join_table_df: pd.DataFrame = joined_table.to_pandas()

    data_ar: np.ndarray = join_table_df.to_numpy()

    data_features: np.ndarray = data_ar[:, 2:6]
    data_learner: np.ndarray = data_ar[:, 6:7]

    x_train, y_train = data_features[0:100], data_learner[0:100]
    x_test, y_test = data_features[100:], data_learner[100:]

    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)


    # create model and move it to GPU with id rank

    print(f"Model object is created with rank {rank} and world size {world_size}")

    x_train = torch.from_numpy(x_train).to(rank)
    y_train = torch.from_numpy(y_train).to(rank)
    x_test = torch.from_numpy(x_test).to(rank)
    y_test = torch.from_numpy(y_test).to(rank)

    print(x_train.size())

    model = Network().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    if rank == 0:
        print("Training A Dummy Model")
    for t in range(20):
        batch_idx = 1
        for (x_batch, y_batch) in zip(x_train, y_train):
            prediction = ddp_model(x_batch)
            loss = loss_fn(prediction, y_batch)

            print('GPU: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                rank, t, batch_idx, len(x_train),
                100. * batch_idx / len(x_train), loss.item()))
            batch_idx = batch_idx + 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    cleanup()


if __name__ == '__main__':
    ctx: CylonContext = CylonContext('mpi')
    rank = ctx.get_rank()
    world_size = ctx.get_world_size()
    demo_basic(rank=rank, world_size=world_size)
    ctx.finalize()
