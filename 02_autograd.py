import torch

# GRADIENT CALCULATION
x = torch.randn(3, requires_grad=True)  # enables method grad_fn in future tensors
print(x)

y = x + 2
z = 2*y*y
w = z.mean()  # -> w is a scalar
print(w)  # has grad_fn method that points bakwards to last operation over the tensor

w.backward()  # calculates dw/dx (the gradient) -> only works on scalar tensors!
print(x.grad)  # gradient is now available

# IF WE HAVE A VECTOR FUNCTION
print()
x = torch.randn(3, requires_grad=True)
print(x)
y = x + 2
z = 2*y*y
v = torch.tensor([2, 3, 2]) # -> acts as derivative of scalar function respect to 'z' (evaluated at 'x')
z.backward(v)  # this is equivalent to (z dot v).backward()
print(x.grad)

# IF NO GRADIENT CALCULATION IS NEEDED...
print()
# OPTION 1
with torch.no_grad():  # disables gradient calculation inside it's scope
    a = x + 2
    print(a)  # has no grad_fn attribute
# OPTION 2
a = x.detach()  # returns untracked version of x
b = a + 2
print(b)
# OPTION 3
x.requires_grad_(False)  # changes in place the tracking flag
a = x + 2
print(a)

## CLEAN THE GRADIENT (after each cycle, when performing iterations)
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()  # update gradient
    print(weights.grad)
    weights.grad.zero_()  # zero out the gradient (in place, obviously)

