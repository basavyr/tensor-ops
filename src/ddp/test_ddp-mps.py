import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo0'

    dist.init_process_group("gloo", rank=rank, world_size=size)

    if not torch.backends.mps.is_available():
        print(f"Rank {rank}: MPS not available")
        return

    # Try MPS
    device = torch.device("mps")
    tensor = torch.ones((5,)).to(device)
    print(f"Rank {rank}: Tensor {tensor} on {tensor.device}")

    # Manual move to CPU for Gloo
    cpu_tensor = tensor.cpu()
    dist.all_reduce(cpu_tensor, op=dist.ReduceOp.SUM)
    print(f'All reduce: {rank}: {cpu_tensor}')
    tensor.copy_(cpu_tensor)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(run, args=(2,), nprocs=2)
