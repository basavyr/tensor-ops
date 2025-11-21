import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def check_nccl_connectivity(rank, world_size):
    # 1. Setup specific environment for the process
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # 2. Initialize the Process Group
    # We use "nccl" backend to specifically test the GPU communication
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 3. Set the specific GPU for this process
    torch.cuda.set_device(rank)

    # 4. Create a simple tensor on the GPU (e.g., value = 1.0)
    tensor = torch.ones(1).cuda()

    print(f"[GPU {rank}] Tensor value before reduction: {tensor.item()}")

    # 5. Perform All-Reduce (Summing values from all GPUs)
    # If this hangs, your NCCL setup is broken.
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"[GPU {rank}] Tensor value AFTER reduction: {tensor.item()}")

    # If we have 2 GPUs, the result should be 2.0 on all GPUs
    if tensor.item() == float(world_size):
        print(f"[GPU {rank}] SUCCESS: NCCL communication is working!")
    else:
        print(f"[GPU {rank}] FAILURE: Values did not sum correctly.")

    dist.destroy_process_group()


if __name__ == "__main__":
    # Number of GPUs to test (in your case, 2)
    WORLD_SIZE = 2
    print(f"Testing NCCL connectivity on {WORLD_SIZE} GPUs...")

    # Spawns one process per GPU
    mp.spawn(check_nccl_connectivity,
             args=(WORLD_SIZE,),
             nprocs=WORLD_SIZE,
             join=True)
