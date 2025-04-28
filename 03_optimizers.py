import torch

weights = torch.randn(4, requires_grad=True)
print(weights)

optimizer = torch.optim.SGD([weights], lr=0.01)  # SGD is Stochastic Gradient Descent
# Update according to the rule: weights -= grad(weights)*lr

calculation = (weights*5).sum()  # operate on weights
print(calculation)
calculation.backward()  # update gradient

optimizer.step()
optimizer.zero_grad()  # reset the gradient from the optimizer itself
print(weights)

