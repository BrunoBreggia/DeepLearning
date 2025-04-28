import torch
import numpy as np

## STORAGE DEVICE

device = torch.device("cpu")
if torch.cuda.is_available():  # checks if there is GPU available
    device = torch.device("cuda")

x = torch.zeros(2,2, device=device)  # creates tensor, alternatives: ones, randn, empty, tensor
y = torch.zeros(2,2)  # by default it is stored at cpu
y = y.to(device)  # returns tensor stored in specified device

z = x + y  # can operate on tensors stored in same device
z = z.to("cpu").numpy()  # to get numpay array from tensor, it must be stored in CPU
print(z)
print(type(z))

z = torch.from_numpy(z)  # and stores it in the cpu

# When creating tensors, we have parameters: size, dtype, device, requieres_grad (default False)
z = torch.zeros(2,2, dtype=torch.float32, device=device, requires_grad=True)  # float32 is default
