import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Prepare the data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_sample, n_features = X.shape
print(n_sample, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# print(X_train, X_test)
# print()

# scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(X_train, X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
# print(y_train.shape, y_test.shape)

y_train = y_train.view(y_train.shape[0], 1)  # Reshape y to be a column vector
y_test = y_test.view(y_test.shape[0], 1)  # Reshape y to be a column vector

# print(y_train.shape, y_test.shape)

# 1) model
# f = w*x + b, sigmoid at the end
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)  # one output (one neuron)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegression(n_features)

# 2) Loss and optimizer
learning_rate = 0.005
criterion = nn.BCELoss()  # Binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Stochastic Gradient Descent

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    y_predicted = model(X_train)
    # loss
    loss = criterion(y_predicted, y_train)
    # backward pass
    loss.backward()
    # updated
    optimizer.step()
    # zero gradients
    optimizer.zero_grad()

    if (epoch+1)%10 == 0:
        print(f"epoch: {epoch+1}, loss{loss.item():.4f}")

# 4) testing
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')

