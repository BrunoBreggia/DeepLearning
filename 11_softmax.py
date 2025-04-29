"""
SOFTMAX allows us to convert vector of outputs into probabilities.
CROSS ENTROPY is used as a loss function
"""

import torch
import torch.nn as nn
import numpy as np

# Softmax with numpy
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0 ,0.1])
outputs = softmax(x)
print(f"Softmax with numpy: {outputs}")

# Softmax with pytorch
x = torch.tensor([2.0, 1.0 ,0.1])
outputs = torch.softmax(x, dim=0)
print(f"Softmax with pytorch: {outputs}")

# Cross entropy in numpy
def cross_entropy(y_predicted, y_actual):
    loss = -np.sum(y_actual*np.log(y_predicted))
    return loss / float(y_predicted.shape[0])  # -> optional normalization

# y must be one-hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1,0,0])

Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
L1 = cross_entropy(Y_pred_good, Y)
L2 = cross_entropy(Y_pred_bad, Y)
print(f"Loss 1 with numpy {L1:.4f}")
print(f"Loss 2 with numpy {L2:.4f}")

# Cross entropy in pytorch
loss = nn.CrossEntropyLoss()  # internally applies softmax

# y must be class encoded -> not one-hot!
# if class 0: [0]
# if class 1: [1]
# if class 2: [2]
Y = torch.tensor([2, 0, 1])

# Dimension de salida de la red: n_samples x n_features = 1 x 3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], 
                            [2.0, 1.0, 0.1], 
                            [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.0, 1.0, 0.1],
                           [0.1, 1.0, 2.1],
                           [0.1, 0.3, 0.6]])
L1 = loss(Y_pred_good, Y)
L2 = loss(Y_pred_bad, Y)
print(f"Loss 1 with pytorch {L1.item():.4f}")
print(f"Loss 2 with pytorch {L2.item():.4f}")

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)  # -> indicates the class we choose ([0] [1] [2])
print(predictions2)

