import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

batchSize = 64

trainDataLoader = DataLoader(training_data, batch_size=batchSize)
testDataLoader = DataLoader(test_data, batch_size=batchSize)

for X, y in testDataLoader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

lossFunction = nn.modules.loss.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataLoader, model, lossFunction, optimizer):
    size = len(dataLoader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataLoader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = lossFunction(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataLoader, moodel, lossFunction):
    size = len(dataLoader.dataset)
    numBatches = len(dataLoader)
    model.eval()
    testLoss, correct = 0, 0

    with torch.no_grad():
        for X,y in dataLoader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            testLoss += lossFunction(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    testLoss /= numBatches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {testLoss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(trainDataLoader, model, lossFunction, optimizer)
    test(testDataLoader, model, lossFunction)

print("Done")