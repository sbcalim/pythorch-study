import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cpu"
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()

print(f"Device: {device}")

# This is unrelated from the actual topic. I just wanted to experiment with Flatten
# btw. Default Flatten parameters seem to be (1, input.dim()-1)
input = torch.randn(5, 3, 4, 8, 2)
# print(input.dim())
m = nn.Flatten(1,2)
output = m(input)
# print(f"Output: {output.size()}")

# Neural network class inherits the nn.Module.
# Layers are defined in __init__ function
# Input is handled in forward function
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
