import torch

import torch.nn as nn

import os
import sys

DEBUG = int(os.getenv("DEBUG", "0"))


class MLP(nn.Module):
    def __init__(self, in_features: int, activation_fn=None) -> None:
        super(MLP, self).__init__()

        self.activation_fn = nn.ReLU() if activation_fn is None else activation_fn
        if DEBUG >= 1:
            print(f'-------\nUsing activation: {self.activation_fn}\n-------')
        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x: torch.Tensor):
        print(f'0: {x.shape}')
        x = x.view(x.shape[0], -1)
        print(f'v: {x.shape}')
        x = self.fc1(x)
        print(f'fc1: {x.shape}')
        x0 = x[0]
        x = self.activation_fn(x)
        x00 = x[0]
        print(x0, x00)
        print(f'act+fc1: {x.shape}')
        logits = self.fc2(x)
        print(f'fc2: {logits.shape}')
        return logits


def main():
    torch.manual_seed(1137)
    device = torch.device("mps")

    batch_size = 128
    c = 1
    h = w = 28
    num_classes = 10
    in_features = c*h*w
    # fake data
    x = torch.randn(batch_size, c, h, w)
    x = x.to(device)
    y_true = torch.randn((batch_size, num_classes))
    y_true = y_true.to(device)

    # change activation function
    act_fn = torch.nn.functional.relu

    model = MLP(in_features, act_fn)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters())

    y = model(x)
    loss = loss_fn(y, y_true)
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    main()
