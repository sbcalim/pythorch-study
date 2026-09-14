import torch
import numpy as np
from triton.language import dtype

data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
x_data = torch.Tensor(data)

np_data = np.array(data)
x_np_data = torch.Tensor(np_data)

x_ones = torch.ones_like(x_np_data)
x_rand = torch.rand_like(x_np_data, dtype=torch.float64)

shape = (4,6)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

tensor = torch.rand(4,5)

if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

tensor = torch.ones(4, 4)
tensor[:,1] = 0

t1 = torch.cat([tensor, tensor, tensor], dim=1)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

agg = tensor.sum()
agg_item = agg.item()
# print(f"Item: {agg_item}, type: {type(agg_item)}")

# print(tensor)
tensor.add_(5)
# print(tensor)

t = torch.ones(5)
n = t.numpy()
t.add_(1)
# print(n)
# print(t)

n = np.ones(7)
t = torch.from_numpy(n)
np.add(n, 4, out=n)
# print(n)
# print(t)