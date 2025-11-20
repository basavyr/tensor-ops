from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn as nn


def get_input_size(x: torch.Tensor) -> int:
    assert len(x.shape) > 3, "Incorrect input tensor shape"
    prod = 1
    for s in list(x.shape[1:]):
        prod *= s
    return prod


class DatasetGenerator(Dataset):
    def __init__(self, num_samples: int = 10, n_channels: int = 3, x_height: int = 32, x_width: int = 32, num_y_classes: int = 10, device: str = "mps"):
        self.x_shape = (n_channels, x_height, x_width)  # C, H, W
        self.x = torch.randn(num_samples, *self.x_shape,
                             device=device)  # N, C, H, W
        self.y = torch.randint(
            0, num_y_classes, (num_samples,), device=device)  # targets

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Model(nn.Module):
    def __init__(self, input_size: int):
        super(Model, self).__init__()
        self.input_size = input_size

        self.model = nn.Sequential(nn.Linear(input_size, 10),
                                   nn.ReLU(),
                                   nn.Linear(10, 15),
                                   nn.Softshrink(),
                                   nn.Linear(15, 1),)

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        x = self.model(x)
        return x


while True:
    torch.mps.profiler.start(mode='interval', wait_until_completed=True)
    N = 1000
    batch_size = 128
    input_size = 3*32*32
    device = "mps"
    dataset = DatasetGenerator(
        num_samples=N, n_channels=3, x_height=32, x_width=32, num_y_classes=10, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.autograd.profiler.record_function("My Model Forward Pass"):
        model = Model(input_size=input_size).to(device)
        for x, y_true in dataloader:
            y = model(x)

    torch.mps.profiler.stop()
