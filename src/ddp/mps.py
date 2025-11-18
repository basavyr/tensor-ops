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
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def test_all_reduce(rank: int, world_size: int, device: torch.device):
    """Test all_reduce operation - sums tensors across all processes"""
    real_sum = sum(list(range(world_size)))

    # Each process creates a tensor with its rank value
    t = torch.ones((5,), device=device) * rank
    print(f'Before all reduce: {rank}: {t}')

    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    print(f'All reduce: {rank}: {t}')

    print(f"[Rank {rank}] Expected: {real_sum}, Got: {t[0].item()}")


def main(rank: int, world_size: int):
    setup_dist(rank, world_size)
    print(
        f"Process {rank} has joined the process group! ({rank}/{world_size})")

    device = torch.device("cpu", rank)
    test_all_reduce(rank, world_size, device)

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--world-size", type=int, required=True)
    args = parser.parse_args()

    mp.spawn(main, args=(args.world_size,), nprocs=args.world_size)
