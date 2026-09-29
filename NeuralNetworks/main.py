from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # For a conv with no padding, output size = input_size - kernel_size + 1 = 32 - 5 + 1 = 28.
        c1 = F.relu(self.conv1(x))
        # print(c1.size())
        s2 = F.max_pool2d(c1, (2,2))
        # print(s2.size())
        c3 = F.relu(self.conv2(s2))
        # print(c3.size())
        s4 = F.max_pool2d(c3, 2)
        s4 = torch.flatten(s4, 1)
        # print(s4.size())
        f5 = F.relu(self.fc1(s4))
        # print(f5.size())
        f6 = F.relu(self.fc2(f5))
        # print(f6.size())
        output = self.fc3(f6)
        # print(output.size())
        return output

net = Net()
print(net)

params = list(net.parameters())
# print(len(params))
# print(params[0].size())
# print(params[1].size())
# print(params[2].size())
# print(params[3].size())
# print(params[4].size())
# print(params[5].size())
# print(params[6].size())
# print(params[7].size())
# print(params[8].size())
# print(params[9].size())

inp = torch.rand(1,1,32,32)
out = net(inp)
# print(out)

net.zero_grad()
# out.backward(torch.randn_like(out))

target = torch.randn_like(out)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
print()

learning_rate = 0.01
for i in range(1000):
    for f in net.parameters():
        with torch.no_grad():
            f -= f.grad * learning_rate

    import torch.optim as optim
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    optimizer.zero_grad()
    output = net(inp)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(loss) # Last few are the same