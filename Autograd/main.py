import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights)
data = torch.randn(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data)
# print(prediction)

loss = torch.nn.MSELoss()(prediction, labels)
# This is where the autograd magic happens. labels are prepared ford backward propagation.
loss.backward()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Apply gradient descent
optimizer.step()

# What goes in the background? How does Autograd collect gradients?:
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3 * a ** 3 - b ** 2

# Let’s assume a and b to be parameters of an NN, and Q to be the error.
# Gradients would be parameters' derivatives as 9*a**2 and -2*b

# Because gradient is a vector, we have to pass that as an argument to .backward function.
Q.backward(torch.tensor([1., 1.]))

# Now we have the gradients of a and b
print(a.grad)
print(b.grad)
