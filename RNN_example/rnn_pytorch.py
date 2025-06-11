# base code: feedforward from script 13
import torch
from torch import nn
import torch.utils.data.dataloader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper paramters
# input_size = 28 * 28
input_size = 28
sequence_lenght = 28

hidden_size = 128
n_classes = 10
n_epochs = 2
batch_size = 100
learning_rate = 0.001
n_layers = 2  # two rnn layers in cascade

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
plt.show()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # x -> (batch_size, seq, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, (h0, c0))
        # out: batch_size, seq_length, hidden_size
        # out: (N, 28, 128)
        out = out[:, -1, :]
        # out: (N, 128)
        out = self.fc(out)
        return out

model = RNN(input_size, hidden_size, n_layers, n_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoc in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 28, 28]
        images = images.reshape(-1, sequence_lenght, input_size).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"epoch: {epoc + 1}/{n_epochs}, step: {i+1}/{n_total_steps}, loss: {loss.item():.4f}")

# testing
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        # reshape
        images = images.reshape(-1, sequence_lenght, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)  # returs (value, index) of the maximum value
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy = {acc:.4f}")
