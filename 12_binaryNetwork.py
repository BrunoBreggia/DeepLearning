""""
Objective of activation functions: add non-linearity to the network.
Makes it capable of representing functions that are not necessarily 
a linear combination of the inputs.
"""
import torch
import torch.nn as nn

class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        # out = torch.relu(out)  # alternative
        out = self.linear2(out)
        y_pred = self.sigmoid(out)
        # y_pred = torch.sigmoid(out)  # alternative
        return y_pred

model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

"""
Other activations functions:
* Step function (binary, non continuous)
* Sigmoid (binary, continuous)
* Hyperbolic tangent (binary, continuouos)

* Relu (unbounded, not smooth)
* Leaky relu (unbounded, not smooth)

* Softmax (normalization for many outputs)
"""

