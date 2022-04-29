import numpy as np
import time
from sklearn.model_selection import train_test_split

import numpy as np

import torchvision
import torchvision.transforms as transforms
import timeit

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import json
from emukit.core.continuous_parameter import ContinuousParameter
from emukit.examples.gp_bayesian_optimization.unknown_constraint_bayesian_optimization_2 \
    import UnknownConstraintGPBayesianOptimization2
from emukit.core.loop import UserFunctionResult

if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print("cuda:1")
else:
    device = torch.device("cpu")
    print("cpu")

class MNISTConvNet(torch.nn.Module):
    def __init__(self, num_conv_layers, num_filters_1, num_filters_2, num_filters_3, dropout_rate, num_fc_units,
                 kernel_size):
        super(MNISTConvNet, self).__init__()

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
        for x, y in data_loader:
            output = model(x)
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
    # import pdb; pdb.set_trace()
    accuracy = float(correct) / float(len(data_loader.sampler))
    return (accuracy)

def compute_size(model):
    re1 = model.number_of_parameters()
    return (re1 * 32) / (8 * (10**6))

def evaluate(params):
    # x = params['X']
    # y = params['Y']
    #
    # print 'Evaluating at (%f, %f)' % (x, y)
    #
    # if x < 0 or x > 5.0 or y > 5.0:
    #     return np.nan
    # # Feasible region: x in [0,5] and y in [0,5]
    #
    # obj = float(np.square(y - (5.1/(4*np.square(math.pi)))*np.square(x) + (5/math.pi)*x- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x) + 10)
    #
    # con1 = float(y-x)   # y >= x
    #
    # con2 = float(10.0-y)  # y <= 10
    #
    # return {
    #     "branin"       : obj,
    #     "y_at_least_x" : con1,
    #     "y_at_most_10" : con2
    # }

    # True minimum is at 2.945, 2.945, with a value of 0.8447

    print('Evaluating at ')
    config = {'num_conv_layers': int(params[0]), 'num_filters_1': int(params[1]),
              'num_filters_2': int(params[2]), 'num_filters_3': int(params[3]),
              'num_fc_units': int(params[4]), 'lr': params[5],
              'dropout_rate': params[6], 'num_training_data': int(params[7])}
    print(config)
    batch_size = 64
    if config['num_conv_layers'] == 1:
        config['num_filters_2'] = 0
        config['num_filters_3'] = 0
    elif config['num_conv_layers'] == 2:
        config['num_filters_3'] = 0

    model = MNISTConvNet(num_conv_layers=config['num_conv_layers'],
                         num_filters_1=config['num_filters_1'],
                         num_filters_2=config['num_filters_2'] if config['num_filters_2'] != 0 else None,
                         num_filters_3=config['num_filters_3'] if config['num_filters_3'] != 0 else None,
                         dropout_rate=config['dropout_rate'],
                         num_fc_units=config['num_fc_units'],
                         kernel_size=3
                         )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor())

    train_data, valid_data = train_test_split(train_dataset, train_size=config['num_training_data'], test_size=1024,
                                              stratify=train_dataset.targets)

    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(config['num_training_data']))
    # validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(config['num_training_data'], config['num_training_data'] + 1024))

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    validation_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=1024, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    model.to(device)

    start = timeit.default_timer()

    for epoch in range(9):
        loss = 0
        model.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()

    stop = timeit.default_timer()

    training_time = stop - start

    model_size = compute_size(model)

    train_accuracy = evaluate_accuracy(model, train_loader)
    validation_accuracy = evaluate_accuracy(model, validation_loader)
    test_accuracy = evaluate_accuracy(model, test_loader)

    # constraint = float(20 - training_time)
    time_stamp = time.time()
    # results = np.load('./temp/time_stamp.npy', allow_pickle=True)
    #
    # print(results[0, 2])
    # time_stamp = time_stamp - results[0, 2]
    # print(time_stamp)
    # result = [[config, 1 - validation_accuracy, time_stamp, training_time]]
    # results = np.append(results, result)
    # np.save('./time_stamp', results)

    print('train_accuracy: %s validation_accracy: %s test_accuracy: %s traning_time: %s model_size: %s' % (
        train_accuracy, validation_accuracy, test_accuracy, training_time, model_size))
    # validation_accuracy = 0.001
    return 1 - validation_accuracy, test_accuracy, training_time, time_stamp, model_size


def init():
    init_num_conv_layers = np.random.randint(low=1, high=3, size=(5, 1))
    init_num_filters_1 = np.random.randint(low=4, high=64, size=(5, 1))
    init_num_filters_2 = np.random.randint(low=4, high=64, size=(5, 1))
    init_num_filters_3 = np.random.randint(low=4, high=64, size=(5, 1))
    init_num_fc_units = np.random.randint(low=8, high=256, size=(5, 1))
    init_lr = np.random.uniform(low=1e-6, high=1e-1, size=(5, 1))
    init_dropout_rate = np.random.uniform(low=0.0, high=0.9, size=(5, 1))
    init_num_training_data = np.random.randint(low=1024, high=8192, size=(5, 1))

    init_config = np.hstack(
        (init_num_conv_layers, init_num_filters_1, init_num_filters_2, init_num_filters_3, init_num_fc_units, init_lr,
         init_dropout_rate, init_num_training_data))
    init_loss = []
    init_c_training_time = []
    init_c_model_size = []
    for i in init_config:
        loss, test_accuracy, training_time, time_stamp, model_size = evaluate(i)
        init_loss.append(loss)
        # init_constraints.append([training_time, model_size])
        init_c_training_time.append(training_time)
        init_c_model_size.append(model_size)

    init_loss = np.array(init_loss).reshape((5, 1))
    init_c_training_time = np.array(init_c_training_time).reshape((5, 1))
    init_c_model_size = np.array(init_c_model_size).reshape((5, 1))
    return init_config, init_loss, init_c_training_time, init_c_model_size


def create_config_space():
    num_conv_layers = ContinuousParameter('num_conv_layers', 1, 3)
    num_filters_1 = ContinuousParameter('num_filters_1', 4, 64)
    num_filters_2 = ContinuousParameter('num_filters_2', 4, 64)
    num_filters_3 = ContinuousParameter('num_filters_3', 4, 64)
    num_fc_units = ContinuousParameter('num_fc_units', 8, 256)
    lr = ContinuousParameter('lr', 1e-6, 1e-1)
    dropout_rate = ContinuousParameter('dropout_rate', 0.0, 0.9)
    max_num_training_data = np.log(8192)
    min_num_training_data = np.log(1024)
    num_training_data = ContinuousParameter('num_training_data', min_num_training_data, max_num_training_data)
    config_space = [num_conv_layers, num_filters_1, num_filters_2, num_filters_3, num_fc_units, lr, dropout_rate, num_training_data]

    return config_space


if __name__ == '__main__':
    constraint_results = np.array([np.inf, 0, 0, 0, 0, []], dtype=object)
    record_results = np.array([np.inf, 0, 0, 0, 0, []], dtype=object)

    n_iterations = 100

    init_config, init_loss, init_c_training_time, init_c_model_size = init()

    init_c_training_time = init_c_training_time - 1.5
    init_c_model_size = init_c_model_size - 0.5
    config_space = create_config_space()
    bo = UnknownConstraintGPBayesianOptimization2(variables_list=config_space, X=init_config, Y=init_loss,
                                                 Yc1=init_c_training_time, Yc2=init_c_model_size, batch_size=1)
    results = None
    best_result = np.inf
    start_time = time.time()
    for _ in range(n_iterations + 1):
        X_new = bo.get_next_points(results)
        new_config = [round(X_new[0][0]), round(X_new[0][1]), round(X_new[0][2]), round(X_new[0][3]),
                      round(X_new[0][4]), X_new[0][5], X_new[0][6], round(X_new[0][7])]
        new_config[7] = round(np.power(np.e, X_new[0][7]))
        print("training instances:", new_config[7])
        _loss, test_accuracy, training_time, time_stamp, model_size = evaluate(new_config)
        record_time = time_stamp - start_time
        loss = np.array(_loss).reshape((1, 1))
        c_training_time = np.array(training_time).reshape((1, 1))
        c_model_size = np.array(model_size).reshape((1, 1))
        c_training_time = c_training_time - 1.5
        c_model_size = c_model_size - 0.5
        print(c_training_time, c_model_size)
        if (training_time < 1.5) and (model_size < 0.5):
            constraint_results = np.vstack((constraint_results, np.array([_loss, test_accuracy, training_time, record_time, model_size, new_config], dtype=object)))
            current_best_results = constraint_results[constraint_results[:, 0].argsort()][0]
            current_best_results[3] = record_time
            record_results = np.vstack((record_results, current_best_results))

            if best_result > loss:
                best_config = new_config
                best_result = loss
                print("best result: ")
                print("------------------------------------------------------------")
                print("best config: {} best result: {} traning_time: {} model size: {}".format(best_config, best_result, training_time, model_size))
        results = [UserFunctionResult(X_new[0], loss[0], Y_constraint_1=c_training_time[0], Y_constraint_2=c_model_size[0])]

    np.save('./running_results_origin_cnn_MNIST_1.5_0.5_log_1.npy', record_results)
