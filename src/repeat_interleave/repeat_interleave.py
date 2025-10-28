import time

import torch

import sys


def test_repeat_interleave(t: torch.Tensor, reps: int = 1, ignore_timings: bool = False):
    print(f'Applying {__name__}')
    # print(f'Tensor -> {t}\n(shape: {t.shape})')
    start = time.monotonic()

    tt = torch.tensor([1], dtype=torch.bfloat16).to(t.device)
    print(tt)
    print(tt.shape)
    print(tt.nelement())

    sys.exit(1)
    for _ in range(reps):
        t = t.repeat_interleave(3, dim=0)
        buffer_size = t.nelement()*t.element_size()/(1024**3)
        print(buffer_size)

        # print(f'Repeat -> {t}\n(shape: {t.shape})')

    duration = time.monotonic() - start
    if not ignore_timings:
        print(f'Op: {duration:.6f} on < {t.device}')
