import torch


def test_repeat_interleave(t: torch.Tensor):
    print(f'Applying {__name__}')
    print(f'Tensor -> {t}\n(shape: {t.shape})')
    t = t.repeat_interleave(3, dim=0)
    print(f'Repeat -> {t}\n(shape: {t.shape})')
