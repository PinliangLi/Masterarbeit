import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import math
from sklearn.metrics import mean_absolute_error

all_results = np.load('./temp/all_results.npy', allow_pickle=True)
all_results = np.delete(all_results, 0, 0)

target = torch.tensor(all_results[:, -1]).float()
train_data_X = torch.tensor(all_results[:, :-1]).float()

print(train_data_X, target)

train_tensor = data_utils.TensorDataset(train_data_X, target)
num_train_data = math.floor(0.9*train_data_X.size(dim=0))
num_valid_data = math.floor(0.1*train_data_X.size(dim=0))
train_data = torch.utils.data.Subset(train_tensor, range(num_train_data))
valid_data = torch.utils.data.Subset(train_tensor, range(num_train_data, num_train_data + num_valid_data))

train_loader = data_utils.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
valid_loader = data_utils.DataLoader(dataset=valid_data, batch_size=1, shuffle=True)


class Net(nn.Module):
    def __init__(self, size):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(size, 120)
        self.fc2 = nn.Linear(120, 240)
        self.fc3 = nn.Linear(240, 120)
        self.fc4 = nn.Linear(120, 40)
        self.fc5 = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return torch.sigmoid(x)


model = Net(train_data_X.size(dim=1))

optimizer = optim.SGD(model.parameters(), lr=0.001)


for epoch in range(10):
    loss = 0
    model.train()
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x)
        print(output)
        loss = F.l1_loss(output, y)
        loss.backward()
        optimizer.step()

model.eval()

with torch.no_grad():
    for x, y in valid_loader:
        #print(x)
        output = model(x)
        #print(output)
        # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        print(output)
        # import pdb; pdb.set_trace()

