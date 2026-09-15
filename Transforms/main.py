# transform: Modify the features
# target_transform: modify the labels

# torchvision.transforms module has a bunch of common transforms ready to use

# FashionMNIST feature image format is PIL, label types are integer.
# Features are needed to be normalized tensors, thus we need torchvision.transforms.v2
# Labels are needed to be one hot encoded tensors, thus we need torch.nn.functional.one_hot

import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import v2

ds = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=v2.Lambda(lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()),
)

# Compose functions: ToImage converts PIL or numpy array => torchvision.tv_tensors.Image
# ToDtype: Converts torchvision.tv_tensors.Image to float. When scale is true, all values put in range 0-1

# Lambda: one_hot here is to used to turn the int label into a one-hot encoded tensor with size 10.
# (10 is the size of labels)
# It is then converted to float to match the expected type.