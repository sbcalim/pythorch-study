import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandals",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    # print(f"sample_idx: {sample_idx}")
    img,label = training_data[int(sample_idx)]
    # print(f"label: {labels_map[label]}\n")
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis('off')
    # print(type(img))
    plt.imshow(img.squeeze(), cmap='cividis')
# plt.show()

from torch.utils.data import DataLoader
# After iterating over all batches, data is shuffled
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
# print(f"Train features: {train_features.size()}")
# print(f"Train labels: {train_labels.size()}")

img = train_features[0].squeeze()
# print(f"img shape: {img.size()}")
label = train_labels[0]

plt.clf()
plt.imshow(img.squeeze(), cmap='gray')
plt.axis('off')
plt.show()
# print(f"Label: {labels_map[int(label)]}")
