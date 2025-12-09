import torch

from torch.utils.flop_counter import FlopCounterMode


class Model(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = False) -> None:
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(
            in_features=input_dim, out_features=output_dim, bias=use_bias)

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1).contiguous()
        x = self.linear(x)
        return x


if __name__ == "__main__":
    """
    for cifar10
    in_features = in ; in = 3072 (3 x 32 x 32)

    1 FLOP = 2x MAC (Fused Multiply Add Accumulate FMA)
    NVIDIA: https://docs.nvidia.com/deeplearning/performance/pdf/Optimizing-Linear-Fully-Connected-Layers-User-Guide.pdf
    -- forward propagation --
    A (input matrix)
    A has dims: N, K
    N: batch size
    K: number of inputs
    """

    # source for convention of notations
    # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-mem
    """
    Mapping of inputs, outputs, and batch size to GEMM parameters M, N, K.
    Computation Phase	               M	             N	            K
    Forward Propagation	        Number of outputs	Batch size	Number of inputs

    Source: https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html
    """
    batch_size = 32
    input_dim = 64
    output_dim = 10
    A = torch.randn((batch_size, input_dim),
                    requires_grad=True)
    B = torch.randn((input_dim, output_dim), requires_grad=True)

    use_backward = False
    print('Configs')
    print(f'Using backward pass: {use_backward}')
    print(f'bs = {batch_size} (denoted by M)')
    print(f'input = {input_dim} (denoted by K)')
    print(f'output = {output_dim} (denoted by N)')

    # O = A@B
    M, K = A.shape
    K, N = B.shape
    O = A@B
    M, N = O.shape
    flops_fp = 2*M * N * K
    flops = flops_fp + 2 * flops_fp if use_backward else flops_fp
    print(f'FLOP: {flops}')
    print(f'A ( M, K ) -> {A.shape}')
    print(f'B ( K, N ) -> {B.shape}')
    print(f'O ( M, N ) -> {O.shape}')

    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        if use_backward:
            A.matmul(B).sum().backward()
        else:
            A.matmul(B)
    flops_builtin = flop_counter.get_total_flops()
    assert flops_builtin == flops, ValueError("Inconsistent FLOP estimation")

    print(f'{"="*80}\nSimulating GEMM with `model(x)`')
    model = Model(input_dim=A.shape[1], output_dim=B.shape[1], use_bias=False)
    print(model)
    flop_counter = FlopCounterMode(display=True, depth=1)
    A.requires_grad_(False)
    with flop_counter:
        model(A)
        if use_backward:
            model(A).sum().backward()
    flops_builtin_model = flop_counter.get_total_flops()
    assert flops_builtin_model == flops, ValueError(
        "Inconsistent FLOP estimation")

    print(f'{"="*80}\nFinal results')
    print(f'FLOP: {flops}')
    print(f'FLOP (builtin): {flops_builtin}')
    print(f'FLOP model (builtin): {flops_builtin_model}')
