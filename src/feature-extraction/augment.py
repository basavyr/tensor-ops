import torch
import torch.nn.functional as F
from torch import distributed
from torchvision.models import resnext50_32x4d, resnet50

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import os
import sys
from typing import Callable


from tqdm import tqdm

from utils import TinyImagenet


def ddp_setup(rank: int, world_size: int):
    """
    Source: https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
    """
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size)


def get_world_size() -> int:
    return torch.cuda.device_count()


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, sampler=DistributedSampler(dataset))


class DDPTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 rank: int,
                 world_size: int):
        # rank is gpu id
        self.gpu_id = rank
        self.model = model.to(rank)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.world_size = world_size
        if self.world_size > 1:
            self.model = DDP(model, device_ids=[rank])

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            preds = 0
            batch_idx = 0
            self.train_loader.sampler.set_epoch(epoch)
            for x, y_true in tqdm(self.train_loader, desc=f"[GPU{self.gpu_id}] Training epoch {epoch+1} / {num_epochs}"):
                x, y_true = x.to(self.gpu_id), y_true.to(self.gpu_id)
                self.optimizer.zero_grad()

                y = self.model(x)
                loss = self.loss_fn(y, y_true)

                preds += (y.argmax(dim=1) == y_true).sum().item()
                epoch_loss += loss.item()*x.shape[0]
                batch_idx += 1

                loss.backward()
                self.optimizer.step()

            epoch_loss /= len(self.train_loader.dataset)
            acc = preds/len(self.train_loader.dataset)*100
            print(
                f'[GPU{self.gpu_id}] Epoch: {epoch}: Loss = {epoch_loss:.4f} Acc = {acc:.2f} %')


def main(rank: int, world_size: int, train_batch_size: int, test_batch_size: int, num_epochs: int):
    print(f'CUDA device: {rank} (total available: {world_size})')
    ddp_setup(rank, world_size)

    tiny = TinyImagenet(train_batch_size=train_batch_size,
                        test_batch_size=test_batch_size)
    train_dataset, _ = tiny.get_datasets()

    train_loader = prepare_dataloader(train_dataset, train_batch_size)

    model = resnext50_32x4d(num_classes=tiny.NUM_CLASSES)
    model.fc = torch.nn.Identity()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = DDPTrainer(model, train_loader, optimizer,
                         loss_fn, rank, world_size)
    trainer.train(num_epochs)

    distributed.destroy_process_group()


if __name__ == "__main__":
    world_size = get_world_size()

    train_batch_size = 256
    test_batch_size = 256
    num_epochs = 5
    args = (world_size, train_batch_size, test_batch_size,
            num_epochs)
    mp.spawn(main, args=args, nprocs=world_size)
