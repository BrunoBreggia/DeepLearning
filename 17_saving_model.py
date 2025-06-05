from typing import Callable

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)
# train the model...

## METHODS TO SAVE THE MODEL
# 1) Lazy method:
FILE = "models/17_model.pth"
torch.save(model, FILE) # uses pickle, serialization (can be used for any object, like tensors, dicts)
# now load the trained model
loaded_model = torch.load(FILE)
loaded_model.eval()
for param in loaded_model.parameters():
    print(param)

# 2) The preferred method
FILE = "models/17_state_dict.pth"
torch.save(model.state_dict(), FILE) # save only state dictionary
# nos load the trained model
loaded_model = Model(n_input_features=6)  # instantiate the model
loaded_model.load_state_dict(torch.load(FILE))  # pass it the state dictionary
loaded_model.eval()
for param in loaded_model.parameters():
    print(param)

# 3) Optimizers also can be saved
model = Model(n_input_features=6)
learning_rate = 0.001
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

# inside the training loop...
checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_dict": optimizer.state_dict()
}
torch.save(checkpoint, "models/checkpoint.pth")
# now load the checkpoint
loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]

model = Model(n_input_features=6)
model.load_state_dict(loaded_checkpoint["model_state"])

optimizer = torch.optim.SGD(params=model.parameters(), lr=0)
optimizer.load_state_dict(loaded_checkpoint["optim_dict"])

# 4) Save on GPU, load on CPU
device = torch.device("cuda")  # work on gpu
model.to(device)
FILE = "models/17_state_dict_gpu.pth"
torch.save(model.state_dict(), FILE)
# load the model on CPU
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE, map_location="cpu"))

# 5) Save on GPU, load on GPU
device = torch.device("cuda")  # work on gpu
model.to(device)
FILE = "models/17_state_dict_gpu.pth"
torch.save(model.state_dict(), FILE)
# load the model on GPU
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.to("cuda")  # this step is necessary, but you can ALSO add map_location="cuda" in torch.load()

# 6)Save on cpu, load on GPU
FILE = "models/17_state_dict_gpu.pth"
torch.save(model.state_dict(), FILE)
# load the model on CPU
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE, map_location="cuda:0"))
loaded_model.to("cuda")
