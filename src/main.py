import torch

from utils import select_optimal_device

from repeat_interleave.repeat_interleave import test_repeat_interleave


def main():
    torch.manual_seed(1137)
    device = select_optimal_device()

    t = torch.randn(1, 3, dtype=torch.bfloat16).to(device)
    test_repeat_interleave(t, 10000)


if __name__ == "__main__":
    main()
