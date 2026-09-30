import torch
import torchvision
from torchvision.transforms import v2

device = torch.accelerator.current_accelerator().type
print(device)

# import time
# start = time.time()

# Torchvision datasets are PILImage of range [0-1]. We have to transform them to Tensor [-1,1]
transform = v2.Compose([v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

batch_size = 4
trainSet = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=0)

testSet = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# print(len(trainLoader))
# dataIterator = iter(trainLoader)
# images, labels = next(dataIterator)
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
# imshow(torchvision.utils.make_grid(images))

# Convolutional network definition
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# net  = Net()

# Loss function definition: Cross-Entropy loss and SGD with momentum
# import torch.optim as optim
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# Training the network
# for epoch in range(2):
#     running_loss = 0
#     for i, data in enumerate(trainLoader, 0):
#         inputs, labels = data
#         optimizer.zero_grad()
#
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#         if i % 2000 == 1999:  # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0
#
# print('Finished Training')
#
PATH = './cifar_net.pt'
# torch.save(net.state_dict(), PATH)

# end = time.time()
# length = end - start
# print("It took", length, "seconds!")
#
# testDataIterator = iter(testLoader)
# images, labels = next(testDataIterator)

# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
# imshow(torchvision.utils.make_grid(images))

net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

# outputs = net(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

correct = 0
total = 0

with torch.no_grad():
    for data in testLoader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

correct_pred = {classname : 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testLoader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')