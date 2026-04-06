import torch

torch._logging.set_logs(graph_code=True)


def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


opt_foo1 = torch.compile(foo)
print(opt_foo1(torch.randn(3, 3), torch.randn(3, 3)))


@torch.compile
def opt_foo2(x, y):
    a = torch.sin(x).requires_grad_()
    b = torch.cos(y).requires_grad_()
    dab = (a+b).sum().backward(retain_graph=True)
    return a+b, dab


device = "mps"
print(opt_foo2(torch.randn(3, 3, device=device), torch.randn(3, 3, device=device)))
