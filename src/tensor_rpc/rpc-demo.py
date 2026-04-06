import torch.distributed.rpc as rpc


def add(a, b):
    return a + b


def run_worker(rank, world_size):
    rpc.init_rpc(
        f"worker{rank}",
        rank=rank,
        world_size=world_size,
        backend=rpc.BackendType.TENSORPIPE
    )
    if rank == 0:
        # Make a remote call to worker1
        fut = rpc.remote("worker1", add, args=(2, 3))
        result = fut.wait()
        print(f"Result from remote call: {result}")
    rpc.shutdown()


if __name__ == "__main__":
    world_size = 2
    for rank in range(world_size):
        run_worker(rank, world_size)
