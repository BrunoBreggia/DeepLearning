import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare data
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20.0, random_state=1)
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)  # Reshape y to be a column vector
print(x.shape, y.shape)  # (100, 1) (100, 1)

n_samples, n_features = x.shape

# 1) Design model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# 2) Define loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # Backward pass
    loss.backward()

    # Weight update
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

[weights, biases] = model.parameters()
print(weights)
print(biases)

# Plot
predicted = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro', label='Original data')
plt.plot(x_numpy, predicted, 'b', label='Fitted line')
plt.title('Linear Regression Result')
plt.legend()
plt.grid()
plt.show()
