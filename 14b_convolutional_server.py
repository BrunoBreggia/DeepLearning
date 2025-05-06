import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from datetime import datetime

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# CIFAR dataset
# data stored as PIL images of ranage [0,1]
# need to transform to tensors of normalized range [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])
# get datasets
train_dataset = torchvision.datasets.CIFAR10(root="./data",
                                             train=True,
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root="./data",
                                            train=False,
                                            transform=transform,
                                            download=True)
# create dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
fecha = datetime.now().strftime("%Y-%m-%d")
print(f"{fecha} Comienza el entrenamiento", flush=True)
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (raw_images, raw_labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32]
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = raw_images.to(device)
        labels = raw_labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%2000 == 0:
            hora = datetime.now().strftime("%H:%M:%S")
            print(f"{hora} epoch: {epoch + 1}/{num_epochs}, step: {i+1}/{n_total_steps}, loss: {loss.item():.4f}",
                  flush=True)

print("Finished training", flush=True)

# testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(10)]
    n_class_samples = [0 for _ in range(10)]

    for raw_images, raw_labels in test_loader:
        images = raw_images.to(device)
        labels = raw_labels.to(device)
        outputs = model(images)
        # torch.max return (max_value, index)
        _, predicted = torch.max(outputs, dim=1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label==pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network: {acc:.2f} %", flush=True)

    for i in range(len(classes)):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f"Accuracy of class {classes[i]}: {acc:.2f} %", flush=True)

ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"{ahora} Finalización del proceso", flush=True)
