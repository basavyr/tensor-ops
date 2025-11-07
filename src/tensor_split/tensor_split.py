import torch


import os
blocks = int(os.getenv("BLOCKS",1))

device = torch.device("cuda:0")
seed = 1137
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

t = torch.randn(128, 3, 28, 28)

ts = t.tensor_split(blocks,dim=1)

t = t.to(device)
print(t.shape)
print("Tensor split op:")

for _ts in ts:
    print(_ts.shape)
