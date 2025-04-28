"""
The training process would be extremely slow if all data were to be
passed forward individually, with parameter updates following each pass.
Hence, we will partition the training dataset into smaller chunks called batches.
In one epoch, all the training dataset has to be processed, by looping over all
batches. So we end up with a double iteration process.
"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# Define our Dataset class: must have init, getitem, len
class WineDataset(Dataset):
    def __init__(self):
        # Load the data to memory
        xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

dataset = WineDataset()
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)  #, num_workers=1)
## dataloader is iterable (has an associated iterator)

# data_iter = iter(dataloader)
# data = next(data_iter)
# features, labels = data
# print(features, labels)

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader): # each time this loop starts,a new dataloader iterator is created
        # forward, backward, update
        if (i+1)%5 == 0:
            print(f"epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}")

