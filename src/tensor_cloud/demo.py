import torch


# torch.manual_seed(1137)
# device = "mps"

T_float = torch.randn(10, 10, dtype=torch.bfloat16).to(device)

T_float.data

# model = torch.nn.Sequential(torch.nn.Linear(10, 10)).to(device)


# torch.save(T_float, "tfloat.safetensor")
# torch.save(model, "model.pth")
# torch.load("tfloat.safetensor")
torch.load("model.pth")
