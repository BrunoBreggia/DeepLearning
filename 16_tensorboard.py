# Use as base code the feed forward net from exercise 13

import torch
from torch import nn
import torch.utils.data.dataloader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

import sys
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/feedForward2")  # change the name of target folder and run a new simulation!
# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper paramters
input_size = 28 * 28
hidden_size = 100
n_classes = 10
n_epochs = 1
batch_size = 64
learning_rate = 0.01

# download MNIST datasets
train_dataset = torchvision.datasets.MNIST(root="./data",
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root="./data",
                                          train=False,
                                          transform=transforms.ToTensor())
# create dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap="gray")
# plt.show()

############## LOG INFO TO TENSORBOARD ##############
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist/images', img_grid)
writer.close()
# sys.exit()
#####################################################

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet(input_size, hidden_size, n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

############## LOG INFO TO TENSORBOARD ##############
writer.add_graph(model, samples.reshape(-1, 28 * 28))
writer.close()
# sys.exit()
#####################################################

# training loop
running_loss = 0.0
running_correct = 0
n_total_steps = len(train_loader)
for epoc in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # need to reshape the input images from 28x28 to 784
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predictions = torch.max(outputs.data, 1)
        running_correct += (predictions == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f"epoch: {i + 1}/{n_epochs}, step: {i}/{n_total_steps}, loss: {loss.item():.4f}")

            ############## LOG INFO TO TENSORBOARD ##############
            writer.add_scalar('training loss', running_loss/100.0, epoc*n_total_steps + i)
            writer.add_scalar('accuracy', running_correct/100.0, epoc*n_total_steps + i)
            #####################################################

            running_loss = 0.0
            running_correct = 0.0

# testing
labels_tb = []
preds = []

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        # reshape
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs.data, 1)  # returs (value, index) of the maximum value
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        labels_tb.append(predictions)
        # class_predictions = [F.softmax(outputs, dim=0) for output in outputs]
        preds.append(F.softmax(outputs, dim=0))

    labels_tb = torch.cat(labels_tb)
    # preds = torch.cat([torch.stack(batch) for batch in preds])
    preds = torch.cat(preds)

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy = {acc:.4f}")

    ############## LOG INFO TO TENSORBOARD ##############
    classes = range(10)
    for i in classes:
        labels_i = (labels_tb==i)
        preds_i = preds[:,i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
    #####################################################
