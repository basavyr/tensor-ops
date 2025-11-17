import torch
import torch.nn.functional as F
from torch import distributed

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import os

from utils import TinyImagenet


def ddp_setup(rank: int, world_size: int):
    """
    Source: https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
    """
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size)


def get_world_size() -> int:
    return torch.cuda.device_count()


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(dataset))


def main(rank: int, world_size: int, train_batch_size: int, test_batch_size: int, num_epochs: int):
    print(f'CUDA device: {rank} (total available: {world_size})')
    ddp_setup(rank, world_size)

    tiny = TinyImagenet(train_batch_size=train_batch_size,
                        test_batch_size=test_batch_size)
    train_dataset, test_dataset = tiny.get_datasets()

    train_loader = prepare_dataloader(train_dataset, train_batch_size)
    test_loader = prepare_dataloader(test_dataset, test_batch_size)

    print(next(iter(train_loader)))
    print(next(iter(test_loader)))

    distributed.destroy_process_group()


if __name__ == "__main__":
    world_size = get_world_size()

    train_batch_size = 1
    test_batch_size = 1
    num_epochs = 5
    args = (world_size, train_batch_size, test_batch_size,
            num_epochs)
    mp.spawn(main, args=args, nprocs=world_size)
