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

import time


# torch.accelerator.synchronize()

# time_before = round(time.time() * 1000)
model = NeuralNetwork().to(device)
# time_after = round(time.time() * 1000)
# print(f"Time taken: {time_after - time_before} milliseconds")

print(model)

# This is interesting. Without synchronizing the accelerator, first .to() takes forever.
# t1 = round(time.time() * 1000)
# nn_obj = NeuralNetwork()
# t2 = round(time.time() * 1000)
# model2 = nn_obj.to(device)
# t3 = round(time.time() * 1000)
# print(f"Object created in {t2-t1} milliseconds")
# print(f"Object moved in {t3-t2} milliseconds")

# The softmax function turns a list of raw numbers into a set of probabilities that always add up to 1.
X = torch.rand(1,28,28,device=device)
logits = model(X)
# print(logits)
pred_probab = nn.Softmax(dim=1)(logits)
# print (pred_probab)
y_pred = pred_probab.argmax(1)
print(f"Predicted: {y_pred}")

# Illustrate layers in our dataset. There will be three 28x28 image
input_image = torch.rand(3,28,28)
print(f"Input image: {input_image.size()}")

# nn.Flatten will convert a 28x28 array into one dimension 784 pixel  value
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(f"Flat image size: {flat_image.size()}")

# nn.Linear is linear layer that applies linear transformation
layer1 = nn.Linear(in_features=28*28,out_features=20)
hidden1 = layer1(flat_image)
print(f"Hidden layer size: {hidden1.size()}")

# nn.ReLu to reset negative values between linear layers
print(f"Before ReLu: {hidden1}\n\n")
hidden1 = torch.relu(hidden1)
print(f"After ReLu: {hidden1}\n\n")

# nn.Sequential is a container of an ordered multi layer neural network.
seq_models = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(in_features=20,out_features=10),
)
input_image = torch.rand(3,28,28)
logits = seq_models(input_image)
# print(logits)

# nn.Softmax... Last logic contains raw values [-infinity, infinity]. Softmax scale them to values between 0 and 1.
# Those values are the probability of each element and total of it adds up to 1.
# dim parameter indicates the dimension
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


# See the model parameters:
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")