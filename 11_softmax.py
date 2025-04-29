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
def cross_entropy(y_actual, y_predicted):
    loss = -np.sum(y_actual*np.log(y_predicted))
    return loss  # / float(predicted.shape[0])  # -> optional normalization

# y must be one-hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1,0,0])

Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
L1 = cross_entropy(Y, Y_pred_good)
L2 = cross_entropy(Y, Y_pred_bad)
print(f"Loss 1 with numpy {L1:.4f}")
print(f"Loss 2 with numpy {L2:.4f}")


