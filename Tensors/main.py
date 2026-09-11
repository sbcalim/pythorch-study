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
    print(torch.accelerator.current_accelerator())
    print(torch.accelerator.device_count())
    tensor_cuda = tensor.cuda()
    tensor = tensor.to(torch.accelerator.current_accelerator())
