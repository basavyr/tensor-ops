from typing import Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize


import os


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def load_train_objs():
    train_set = CIFAR100("./data", train=True, transform=Compose([ToTensor(),
                         Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]), download=True)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    return train_set, model, optimizer, loss_fn


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(dataset)
    )


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        gpu_id: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[gpu_id])
        self.loss_fn = loss_fn

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        self.tloss += loss.item()*source.shape[0]
        self.preds += (output.argmax(dim=1) == targets).sum().item()
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        self.tloss = 0.0
        self.preds = 0
        b_sz = len(next(iter(self.train_loader))[0])
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_loader)}")
        self.train_loader.sampler.set_epoch(epoch)
        for source, targets in self.train_loader:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
        print(
            f'Train Loss/Acc: {self.tloss/len(self.train_loader.dataset):.3f} / {self.preds/len(self.train_loader.dataset)*100}')

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            self._run_epoch(epoch)


def main(rank: int, world_size: int, num_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    print(f'GPU: {rank} / WS: {world_size}')
    train_set, model, optimizer, loss_fn = load_train_objs()
    train_loader = prepare_dataloader(train_set, batch_size)

    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      gpu_id=rank)
    trainer.train(num_epochs)
    destroy_process_group()


if __name__ == "__main__":

    # Configuration
    batch_size = 128
    num_epochs = 5

    # Model
    world_size = torch.cuda.device_count()
    print(f'World size: {world_size}')
    mp.spawn(main, args=(world_size, num_epochs, batch_size), nprocs=world_size)
