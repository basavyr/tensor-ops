import os
from argparse import ArgumentParser

import torch.distributed as dist
import torch.multiprocessing as mp

import torch
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "12355"


def setup_dist(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def test_all_reduce(rank: int, world_size: int, device: torch.device):
    """Test all_reduce operation"""
    _op = dist.ReduceOp.SUM
    # Each process creates a tensor with its rank value
    t = torch.ones((5,), device=device).float()
    print(f'Before < {_op} > all reduce op: {rank}: {t}')

    dist.all_reduce(t, op=_op)
    print(f'All reduce: {rank}: {t}')


def main(rank: int, world_size: int):
    setup_dist(rank, world_size)
    print(
        f"Process {rank+1} has joined the process group! ({rank+1}/{world_size})")

    device = torch.device("cuda", rank)
    test_all_reduce(rank, world_size, device)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()

    mp.spawn(main, args=(args.world_size,), nprocs=args.world_size)
