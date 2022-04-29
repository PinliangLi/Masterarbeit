import numpy as np
from pathlib import Path

from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.metrics import balanced_accuracy
from autosklearn.constants import BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, MULTILABEL_CLASSIFICATION

try:
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.nn.functional as F
except:
    raise ImportError("For this example you need to install pytorch.")

try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")

train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor())

valid_data = torch.utils.data.Subset(train_dataset, range(8192, 8192 + 1024))
validation_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=64,
                                                shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)


class MNISTConvNet(torch.nn.Module):
    def __init__(self, num_conv_layers, num_filters_1, num_filters_2, num_filters_3, dropout_rate, num_fc_units,
                 kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(1, num_filters_1, kernel_size=kernel_size)
        self.conv2 = None
        self.conv3 = None

        output_size = (28 - kernel_size + 1) // 2
        num_output_filters = num_filters_1

        if num_conv_layers > 1:
            self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=kernel_size)
            num_output_filters = num_filters_2
            output_size = (output_size - kernel_size + 1) // 2

        if num_conv_layers > 2:
            self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=kernel_size)
            num_output_filters = num_filters_3
            output_size = (output_size - kernel_size + 1) // 2

        self.dropout = nn.Dropout(p=dropout_rate)

        self.conv_output_size = num_output_filters * output_size * output_size

        self.fc1 = nn.Linear(self.conv_output_size, num_fc_units)
        self.fc2 = nn.Linear(num_fc_units, 10)

    def forward(self, x):

        # switched order of pooling and relu compared to the original example
        # to make it identical to the keras worker
        # seems to also give better accuracies
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)

        if not self.conv2 is None:
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        if not self.conv3 is None:
            x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = self.dropout(x)

        x = x.view(-1, self.conv_output_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))


def evaluate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        index = 0
        for x, y in data_loader:
            if index == 0:
                output = model(x)
                # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1]
                return_y_hat = pred.view_as(y)
                return_y = y
                index = 1
            output = model(x)
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            return_y_hat = torch.cat((return_y_hat, pred.view_as(y)), 0)
            return_y = torch.cat((return_y, y), 0)
    # import pdb; pdb.set_trace()
    return return_y_hat.cpu().detach().numpy(), return_y.cpu().detach().numpy()


constraint_results = np.load('./temp/constraint_results.npy', allow_pickle=True)
constraint_results = np.delete(constraint_results, 0, 0)

configs = np.unique(constraint_results[:, 1])
results = list([])
for i in configs:
    n = list([])
    for j in constraint_results:
        if j[1] == i:
            n.append(j)

    p = sorted(n, key=lambda x: x[7], reverse=True)[0]
    results.append(p)

results = np.array(results)
results = results[results[:, 2].argsort()[::-1]]

print(results)

left_space = 1
configs = []
for i in results:
    left_space = left_space - i[5]
    if left_space < 0:
        break
    else:
        configs.append([i[1], i[6]])

models = []
validation_predictions = []
test_predictions = []
for i in configs:
    model = MNISTConvNet(num_conv_layers=i[1]['num_conv_layers'],
                         num_filters_1=i[1]['num_filters_1'],
                         num_filters_2=i[1]['num_filters_2'] if 'num_filters_2' in i[1] else None,
                         num_filters_3=i[1]['num_filters_3'] if 'num_filters_3' in i[1] else None,
                         dropout_rate=i[1]['dropout_rate'],
                         num_fc_units=i[1]['num_fc_units'],
                         kernel_size=3
                         )
    optimizer = torch.optim.Adam(model.parameters(), lr=i[1]['lr'])
    print(i[0])
    print(i[1])
    model_parameters = Path('./temp/' + str(i[0]) + '.pth')
    model.load_state_dict(torch.load(str(model_parameters)))
    models.append(model)

    validation_pred, validation_y = evaluate_accuracy(model, validation_loader)
    validation_predictions.append(validation_pred)
    test_pred, test_y = evaluate_accuracy(model, test_loader)
    test_predictions.append(test_pred)

#print(validation_predictions[1].dtype)
ensemble_selection = EnsembleSelection(ensemble_size=10,
                             task_type=MULTICLASS_CLASSIFICATION,
                             random_state=0,
                             metric=balanced_accuracy)

ensemble_selection.fit(validation_predictions, validation_y, identifiers=None)
y_hat_ensemble = ensemble_selection.predict(np.array(validation_predictions))
y_hat_test = ensemble_selection.predict(np.array(test_predictions))

for i in range(len(models)):
    print(balanced_accuracy(test_y, test_predictions[i]))
print('\n\n')
print(balanced_accuracy(test_y, y_hat_test))