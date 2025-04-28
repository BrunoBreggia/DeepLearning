import numpy as np

X = np.array([1, 2, 3, 4])
Y = X*2

weight = 0.00  # initial value of single weight
# bias = 0.00

def forward(x):
    return x*weight
    # return x*weight+bias

def loss(y_pred, y):
    return ((y_pred - y)**2).mean()

def gradient_w(x, y, y_pred):
    return np.dot(2*x, y_pred - y).mean()

# def gradient_b(x, y, y_pred):
#     return (y_pred - y).mean()*2

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
    dw = gradient_w(X, Y, Y_pred)
    # db = gradient_b(X, Y, Y_pred)
    # update weights
    weight -= learning_rate * dw
    # bias -= learning_rate * db

    if epoch % 1 == 0:
        print(f"Epoch {epoch+1}: w = {weight:.3f}, loss = {l:.8f}")
        # print(f"Epoch {epoch + 1}: w = {weight:.3f}, b = {bias:.3f}, loss = {l:.8f}")


print(f"Prediction after training: f(5) = {forward(5):.3f}")
# print(f"Expected value: f(5) = {2*5:.3f}")
