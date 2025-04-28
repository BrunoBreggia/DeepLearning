import torch

X = torch.tensor([1, 2, 3, 4])
Y = 2*X

weight = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return x*weight

def loss(y_pred, y):
    return ((y_pred - y)**2).mean()

# no gradient function needed thanks to pytorch's autograd

print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training
learning_rate = 0.01
iterations = 100

for epoch in range(iterations):
    # prediction: forward pass with x samples
    Y_pred = forward(X)
    # loss: compare with expected values
    l = loss(Y_pred, Y)
    # gradients: according to the matrix operations implemented
    l.backward()
    # update weights
    with torch.no_grad():
        weight -= learning_rate * weight.grad

    # zero out gradient
    weight.grad.zero_()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}: w = {weight:.3f}, loss = {l:.8f}")


print(f"Prediction after training: f(5) = {forward(5):.3f}")
# print(f"Expected value: f(5) = {2*5:.3f}")

