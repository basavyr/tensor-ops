import torch
import torch.nn as nn

from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.flop_counter import FlopCounterMode

import os

DEBUG = os.getenv("DEBUG", "0")
BIAS = os.getenv("BIAS", "0")


class Model(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_bias: bool):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=use_bias)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        return x


def test_fma(use_bias: bool, use_profile: bool = False):
    """
    P1

    We use the documentation from NVIDIA to define the shapes of the matrices that will be multiplied. According to [this](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#fullyconnected-layer) guide, for the forward propagation they denote `(M , N , K)` as such:

    * M - output size (number of outputs)
    * K - input size (number of inputs)
    * N - batch size
    # M, N, K == outputs, batch size, inputs

    A basic GEMM for the forward propagation is thus represented as the matrix multiplication between i) transform weight W^T with shape `(M, K)` and ii) the input A with shape `(K, N)`.

    According to NVIDIA: *this convention is used by PyTorch and Caffe where A contains the weights and B the activations.*

    ==========

    P2

    However, when they describe a GEMM through the [Fused Multiply-Adds operations (FMA) guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-mem), the notations are changed:
    * C - output matrix
    * A - first input matrix (denotes `W^T` from P1)
    * B - second matrix to perform the GEMM (denotes the input matrix from P1)
    * FMA: **C = alpha AB + beta C (one multiply + one add = 2 FLOP)** 

    Shapes for A, B, C:
    * C: `(M , N)`
    * A: `(M , K)`
    * B: (`K, N)`

    Understand MLP dimensions: https://stackoverflow.com/questions/55348647/weight-matrix-dimension-intuition-in-a-neural-network
    """
    print(f'Testing FMA for a Linear layer with bias={use_bias}')
    device = "cpu"
    # matmul: (M, K) (K, N)
    # output: (M , N)
    M = 3
    K = 5
    N = 4
    print(f'M={M}')
    print(f'N={N}')
    print(f'K={K}')
    # denote guide G https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-mem
    B = torch.ones(K, N).to(device).float()
    # weight (transposed) is denoted by A in G
    model = Model(N, M, use_bias).to(device)
    model.linear.weight.data = torch.ones_like(
        model.linear.weight).float()

    print(f'B: (K, N) -> {B.shape}')
    print(f'W: (M, N) -> {model.linear.weight.shape}')
    print(f'W^T: (N, M) -> {model.linear.weight.T.shape}')
    if use_bias:
        model.linear.bias.data = torch.ones_like(
            model.linear.bias).float()+0.5
        print(f'b: (M) -> {model.linear.bias.shape}')

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            O = model(B)
    print(f'O (BW^T + b): (K, M) -> {O.shape})')
    if use_profile:
        print(prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=100))
        prof.export_chrome_trace("trace.json")

    if DEBUG == "1":
        print(f'Weight W: {model.linear.weight.data}')
        print(f'Bias b: {model.linear.bias.data}')
        print(f'Output O: {O}')

    test_flop_2d_tensor(B, O, bias=use_bias)


def test_flop_2d_tensor(input: torch.Tensor, output: torch.Tensor, bias: bool = True):
    """
    This evaluates the number of FLOPs for a given input/output tensors.

    **Input must be 2D**
    """
    assert len(input.shape) == 2
    K, N = input.shape
    M = output.shape[-1]
    bias_flops = K * M if bias else 0
    macs = M * N * K + bias_flops
    flops = 2*macs
    print(f"MACs: {macs} | FLOPs: {flops}")


def test_mac_compiled(use_bias: bool):
    """
    Source: https://docs.pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html

    Can be executed with:
         TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 TORCHINDUCTOR_BENCHMARK_KERNEL=1

    (Source: https://docs.pytorch.org/docs/stable/torch.compiler_inductor_profiling.html)
    """
    print(
        f'Testing MAC for a << Compiled >> Linear layer with bias={use_bias}')
    device = "mps"

    N, K = 5, 10
    A = torch.randn(N, K).to(device)
    model = torch.compile(Model(K).to(device))

    with torch.profiler.profile() as prof:
        model(A)

    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="cpu_time_total", row_limit=100))
    prof.export_chrome_trace("trace-compiled.json")


if __name__ == "__main__":
    torch.manual_seed(1137)
    test_fma(True) if BIAS == "1" else test_fma(False)
