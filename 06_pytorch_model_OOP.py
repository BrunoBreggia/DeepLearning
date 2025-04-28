import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)  # one sample per row, one feature per column
Y = 2*X

n_samples, n_features = X.shape  # 4, 1
input_size = n_features
output_size = n_features

class LinearRegression(nn.Module):  # has to inherit from nn.Module
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.linear_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear_layer(x)

model = LinearRegression(input_size, output_size)

X_test = torch.tensor([5], dtype=torch.float32)
print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")

# Training
learning_rate = 0.01
iterations = 2000

# Loss function
loss = nn.MSELoss()  # instantiate a loss function from nn
# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # instantiate an optimizer from torch.optim


for epoch in range(iterations):
    # prediction: forward pass with x samples
    Y_pred = model(X)
    # loss: compare with expected values
    l = loss(Y, Y_pred)
    # gradients: according to the matrix operations implemented
    l.backward()
    # update weights
    optimizer.step()
    # zero out gradient
    optimizer.zero_grad()

    if (epoch+1) % 100 == 0:
        [weights, biases] = model.parameters()
        print(f"Epoch {epoch+1}: w = {weights[0][0].item():.3f}, loss = {l:.8f}")


print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")
# print(f"Expected value: f(5) = {2*5:.3f}")

