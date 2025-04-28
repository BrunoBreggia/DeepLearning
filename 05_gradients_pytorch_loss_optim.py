import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = 2*X

weight = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return x*weight

# Loss function
loss = nn.MSELoss()  # instantiate a loss function from nn
# Optimizer
optimizer = torch.optim.SGD([weight], lr=0.01)  # instantiate an optimizer from torch.optim

# no gradient function needed thanks to pytorch's autograd

print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training
learning_rate = 0.01
iterations = 100

for epoch in range(iterations):
    # prediction: forward pass with x samples
    Y_pred = forward(X)
    # loss: compare with expected values
    l = loss(Y, Y_pred)
    # gradients: according to the matrix operations implemented
    l.backward()
    # update weights
    optimizer.step()
    # zero out gradient
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}: w = {weight:.3f}, loss = {l:.8f}")


print(f"Prediction after training: f(5) = {forward(5):.3f}")
# print(f"Expected value: f(5) = {2*5:.3f}")


