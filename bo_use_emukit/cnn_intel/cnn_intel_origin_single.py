import numpy as np
import time
from sklearn.model_selection import train_test_split

import numpy as np

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
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
from emukit.examples.gp_bayesian_optimization.unknown_constraint_bayesian_optimization \
    import UnknownConstraintGPBayesianOptimization

from emukit.core.loop import UserFunctionResult

if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print("cuda:1")
else:
    device = torch.device("cpu")
    print("cpu")

class MNISTConvNet(torch.nn.Module):
    def __init__(self, num_conv_layers, num_filters_1, num_filters_2, num_filters_3, num_filters_4, num_filters_5, num_filters_6, dropout_rate, num_fc_units,
                 kernel_size):
        super().__init__()

        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.conv4 = None
        self.conv5 = None
        self.conv6 = None

        if num_conv_layers >= 1:
            self.conv1 = nn.Conv2d(3, num_filters_1, kernel_size=kernel_size, padding=1)
            output_size = 32 // 2
            num_output_filters = num_filters_1

        if num_conv_layers >= 2:
            self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=kernel_size, padding=1)
            num_output_filters = num_filters_2
            output_size = 32 // 2

        if num_conv_layers >= 3:
            self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, kernel_size=kernel_size, padding=1)
            num_output_filters = num_filters_3
            output_size = (32 // 2) // 2

        if num_conv_layers >= 4:
            self.conv4 = nn.Conv2d(num_filters_3, num_filters_4, kernel_size=kernel_size, padding=1)
            num_output_filters = num_filters_4
            output_size = (32 // 2) // 2

        if num_conv_layers >= 5:
            self.conv5 = nn.Conv2d(num_filters_4, num_filters_5, kernel_size=kernel_size, padding=1)
            num_output_filters = num_filters_5
            output_size = ((32 // 2) // 2) // 2

        if num_conv_layers >= 6:
            self.conv6 = nn.Conv2d(num_filters_5, num_filters_6, kernel_size=kernel_size, padding=1)
            num_output_filters = num_filters_6
            output_size = ((32 // 2) // 2) // 2


        self.dropout = nn.Dropout(p=dropout_rate)

        self.conv_output_size = num_output_filters * output_size * output_size

        self.fc1 = nn.Linear(self.conv_output_size, num_fc_units)
        self.fc2 = nn.Linear(num_fc_units, 6)

    def forward(self, x):

        # switched order of pooling and relu compared to the original example
        # to make it identical to the keras worker
        # seems to also give better accuracies
        if self.conv2 is None:
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)

            x = self.dropout(x)

            x = x.view(-1, self.conv_output_size)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)


        if self.conv3 is None:

            x = F.relu(self.conv1(x))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)

            x = self.dropout(x)

            x = x.view(-1, self.conv_output_size)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)


        if self.conv4 is None:

            x = F.relu(self.conv1(x))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = F.max_pool2d(F.relu(self.conv3(x)), 2)

            x = self.dropout(x)

            x = x.view(-1, self.conv_output_size)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)


        if self.conv5 is None:

            x = F.relu(self.conv1(x))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(F.relu(self.conv4(x)), 2)

            x = self.dropout(x)

            x = x.view(-1, self.conv_output_size)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

        if self.conv6 is None:

            x = F.relu(self.conv1(x))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(F.relu(self.conv4(x)), 2)
            x = F.max_pool2d(F.relu(self.conv5(x)), 2)

            x = self.dropout(x)

            x = x.view(-1, self.conv_output_size)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

        if not self.conv6 is None:

            x = F.relu(self.conv1(x))
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(F.relu(self.conv4(x)), 2)
            x = F.relu(self.conv5(x))
            x = F.max_pool2d(F.relu(self.conv6(x)), 2)

            x = self.dropout(x)

            x = x.view(-1, self.conv_output_size)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

        raise RuntimeError

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
              'num_filters_4': int(params[4]), 'num_filters_5': int(params[5]),
              'num_filters_6': int(params[6]),
              'num_fc_units': int(params[7]), 'lr': params[8],
              'dropout_rate': params[9], 'num_training_data': int(params[10])}
    print(config)
    batch_size = 1024
    if config['num_conv_layers'] == 1:
        config['num_filters_2'] = 0
        config['num_filters_3'] = 0
        config['num_filters_4'] = 0
        config['num_filters_5'] = 0
        config['num_filters_6'] = 0
    elif config['num_conv_layers'] == 2:
        config['num_filters_3'] = 0
        config['num_filters_4'] = 0
        config['num_filters_5'] = 0
        config['num_filters_6'] = 0
    elif config['num_conv_layers'] == 3:
        config['num_filters_4'] = 0
        config['num_filters_5'] = 0
        config['num_filters_6'] = 0
    elif config['num_conv_layers'] == 4:
        config['num_filters_5'] = 0
        config['num_filters_6'] = 0
    elif config['num_conv_layers'] == 5:
        config['num_filters_6'] = 0




    model = MNISTConvNet(num_conv_layers=config['num_conv_layers'],
                         num_filters_1=config['num_filters_1'],
                         num_filters_2=config['num_filters_2'] if config['num_filters_2'] != 0 else None,
                         num_filters_3=config['num_filters_3'] if config['num_filters_3'] != 0 else None,
                         num_filters_4=config['num_filters_4'] if config['num_filters_4'] != 0 else None,
                         num_filters_5=config['num_filters_5'] if config['num_filters_5'] != 0 else None,
                         num_filters_6=config['num_filters_6'] if config['num_filters_6'] != 0 else None,
                         dropout_rate=config['dropout_rate'],
                         num_fc_units=config['num_fc_units'],
                         kernel_size=3
                         )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    train_dir = '../data/intel_image/seg_train/seg_train'
    test_dir = '../data/intel_image/seg_test/seg_test'

    transf = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor(),
                                 ])  # could be augmentation

    train_dataset = datasets.ImageFolder(train_dir, transform=transf)
    test_dataset = datasets.ImageFolder(test_dir, transform=transf)

    train_data, valid_data = train_test_split(train_dataset, train_size=config['num_training_data'], test_size=1000,
                                              stratify=train_dataset.targets)

    # train_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(config['num_training_data']))
    # validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(range(config['num_training_data'], config['num_training_data'] + 1024))

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    validation_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=1024, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

    model.to(device)

    start = timeit.default_timer()

    for epoch in range(100):
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
    init_num_conv_layers = np.random.randint(low=1, high=6, size=(5, 1))
    init_num_filters_1 = np.random.randint(low=4, high=512, size=(5, 1))
    init_num_filters_2 = np.random.randint(low=4, high=512, size=(5, 1))
    init_num_filters_3 = np.random.randint(low=4, high=512, size=(5, 1))
    init_num_filters_4 = np.random.randint(low=4, high=512, size=(5, 1))
    init_num_filters_5 = np.random.randint(low=4, high=512, size=(5, 1))
    init_num_filters_6 = np.random.randint(low=4, high=512, size=(5, 1))
    init_num_fc_units = np.random.randint(low=8, high=1024, size=(5, 1))
    init_lr = np.random.uniform(low=1e-6, high=1e-1, size=(5, 1))
    init_dropout_rate = np.random.uniform(low=0.0, high=0.9, size=(5, 1))
    init_num_training_data = np.random.randint(low=1500, high=12000, size=(5, 1))

    init_config = np.hstack(
        (init_num_conv_layers, init_num_filters_1, init_num_filters_2, init_num_filters_3, init_num_filters_4, init_num_filters_5, init_num_filters_6, init_num_fc_units, init_lr,
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
    num_conv_layers = ContinuousParameter('num_conv_layers', 1, 6)
    num_filters_1 = ContinuousParameter('num_filters_1', 4, 512)
    num_filters_2 = ContinuousParameter('num_filters_2', 4, 512)
    num_filters_3 = ContinuousParameter('num_filters_3', 4, 512)
    num_filters_4 = ContinuousParameter('num_filters_4', 4, 512)
    num_filters_5 = ContinuousParameter('num_filters_5', 4, 512)
    num_filters_6 = ContinuousParameter('num_filters_6', 4, 512)
    num_fc_units = ContinuousParameter('num_fc_units', 8, 1024)
    lr = ContinuousParameter('lr', 1e-6, 1e-1)
    dropout_rate = ContinuousParameter('dropout_rate', 0.0, 0.9)
    max_num_training_data = np.log(12000)
    min_num_training_data = np.log(1500)
    num_training_data = ContinuousParameter('num_training_data', min_num_training_data, max_num_training_data)
    config_space = [num_conv_layers, num_filters_1, num_filters_2, num_filters_3, num_filters_4, num_filters_5, num_filters_6, num_fc_units, lr, dropout_rate, num_training_data]

    return config_space


if __name__ == '__main__':
    constraint_results = np.array([np.inf, 0, 0, 0, 0, []], dtype=object)
    record_results = np.array([np.inf, 0, 0, 0, 0, []], dtype=object)

    n_iterations = 200

    start_time = time.time()

    init_config, init_loss, init_c_training_time, init_c_model_size = init()

    init_c_training_time = init_c_training_time - 500
    init_c_model_size = init_c_model_size - 20
    config_space = create_config_space()
    bo = UnknownConstraintGPBayesianOptimization(variables_list=config_space, X=init_config, Y=init_loss,
                                                 Yc=init_c_training_time, batch_size=1)
    results = None
    best_result = np.inf

    for _ in range(n_iterations + 1):
        X_new = bo.get_next_points(results)
        new_config = [round(X_new[0][0]), round(X_new[0][1]), round(X_new[0][2]), round(X_new[0][3]), round(X_new[0][4]), round(X_new[0][5]), round(X_new[0][6]),
                      round(X_new[0][7]), X_new[0][8], X_new[0][9], round(X_new[0][10])]
        new_config[10] = round(np.power(np.e, X_new[0][10]))
        _loss, test_accuracy, training_time, time_stamp, model_size = evaluate(new_config)
        record_time = time_stamp - start_time
        loss = np.array(_loss).reshape((1, 1))
        c_training_time = np.array(training_time).reshape((1, 1))
        c_model_size = np.array(model_size).reshape((1, 1))
        c_training_time = c_training_time - 500
        c_model_size = c_model_size - 20

        iteraion_num = _
        print("iteration num: ", iteraion_num)

        print(c_training_time, c_model_size)
        #if (training_time < 125) and (model_size < 20):
        if (training_time < 500):
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
        results = [UserFunctionResult(X_new[0], loss[0], Y_constraint=c_training_time[0])]
        if record_time > 10000:
            print("---------------   time out   ------------------")
            break
    np.save("./iteration_num.npy", iteraion_num)
    np.save('./running_results_bo_cnn6_cifar_125_20_3.npy', record_results)
