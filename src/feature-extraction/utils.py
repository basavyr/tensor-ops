from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from typing import Tuple


class TinyImagenet:
    ROOT_DIR = "/home/robert/dev/gitlab/tiny-imagenet-pytorch/tiny-imagenet-200"
    NUM_CLASSES = 200
    TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    def __init__(self, train_batch_size: int, test_batch_size: int, num_workers: int = 8, pin_memory: bool = True):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_datasets(self):
        train_dataset = ImageFolder(
            f'{self.ROOT_DIR}/train', transform=self.TRANSFORM)
        test_dataset = ImageFolder(
            f'{self.ROOT_DIR}/test', transform=self.TRANSFORM)
        return train_dataset, test_dataset

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        train_dataset = ImageFolder(
            f'{self.ROOT_DIR}/train', transform=self.TRANSFORM)
        test_dataset = ImageFolder(
            f'{self.ROOT_DIR}/test', transform=self.TRANSFORM)

        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size,
                                  shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size,
                                 shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

        return train_loader, test_loader
