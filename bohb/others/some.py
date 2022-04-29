import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
from sklearn.metrics import mean_absolute_error

train_data = pd.read_csv('../../data/home-data-for-ml-course/train.csv')
train_data = train_data.dropna(axis=1)
train_data = train_data.drop(['Id'], axis=1)
print(train_data)

for column in train_data:
    if train_data[column].dtype == 'object':
        translate_dict = {}
        class_num = 1
        for class_name, _ in train_data[column].value_counts().items():
            translate_dict[class_name] = class_num
            class_num += 1
        print(translate_dict)
        train_data[column] = train_data[column].replace(translate_dict)

print(train_data)

train, valid = train_test_split(train_data, train_size=0.9)

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
        return F.sigmoid(x)




train_target = torch.tensor(train['SalePrice'].values.astype(np.float32))
train_target = torch.reshape(train_target, (-1, 1))
train = torch.tensor(train.drop('SalePrice', axis=1).values.astype(np.float32))
train_tensor = data_utils.TensorDataset(train, train_target)
train_loader = data_utils.DataLoader(dataset=train_tensor, batch_size=64, shuffle=True)


valid_target = torch.tensor(valid['SalePrice'].values.astype(np.float32))
valid_target = torch.reshape(valid_target, (-1, 1))
valid = torch.tensor(valid.drop('SalePrice', axis=1).values.astype(np.float32))
valid_tensor = data_utils.TensorDataset(valid, valid_target)
valid_loader = data_utils.DataLoader(dataset=valid_tensor, batch_size=64, shuffle=True)


model = Net(train.size(dim=1))

optimizer = optim.SGD(model.parameters(), lr=0.0001)


for epoch in range(10):
    loss = 0
    model.train()
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x)
        #print(output)
        loss = F.l1_loss(output, y)
        print(loss)
        loss.backward()
        optimizer.step()

model.eval()
pred = []
correct = 0
with torch.no_grad():
    for x, y in valid_loader:
        #print(x)
        output = model(x)
        #print(output)
        # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
        pred.append(output)
        # import pdb; pdb.set_trace()
pred = np.array(pred)
print(pred.reshape((-1, 1)))
print(mean_absolute_error(valid_target, pred))