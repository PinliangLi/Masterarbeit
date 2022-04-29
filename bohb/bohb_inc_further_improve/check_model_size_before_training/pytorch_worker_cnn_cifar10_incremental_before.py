import time
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data.dataloader import default_collate
except:
    raise ImportError("For this example you need to install pytorch.")

try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")

import math
import numpy as np
import timeit
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker
from hpbandster.optimizers.bohb import BOHB

import logging

logging.basicConfig(level=logging.DEBUG)

print('CNN on CIFAR10')

if torch.cuda.is_available():
    device = torch.device("cuda:6")
    print("cuda:6")
else:
    torch.device("cpu")
    print("cpu")


class PyTorchWorker(Worker):
    def __init__(self, N_train=40000, N_valid=5000, num_epoch=100, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = 1024
        self.num_epoch = num_epoch

        # Load the MNIST Data here
        self.train_dataset = torchvision.datasets.CIFAR10(root='../../../data', train=True,
                                                          transform=transforms.ToTensor(),
                                                          download=True)
        self.test_dataset = torchvision.datasets.CIFAR10(root='../../../data', train=False,
                                                         transform=transforms.ToTensor())

        self.all_train_data = torch.utils.data.Subset(self.train_dataset, range(N_train))
        self.all_valid_data = torch.utils.data.Subset(self.train_dataset, range(N_train, N_train + N_valid))

        self.N_train = N_train
        self.N_valid = N_valid
        self.check_budget = {}

    def compute(self, config_id, config, budget, working_directory, *args, **kwargs):
        """
            Simple example for a compute function using a feed forward network.
            It is trained on the MNIST dataset.
            The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
            """

        # device = torch.device('cpu')

        model = MNISTConvNet(num_conv_layers=config['num_conv_layers'],
                             num_filters_1=config['num_filters_1'],
                             num_filters_2=config['num_filters_2'] if 'num_filters_2' in config else None,
                             num_filters_3=config['num_filters_3'] if 'num_filters_3' in config else None,
                             num_filters_4=config['num_filters_4'] if 'num_filters_4' in config else None,
                             num_filters_5=config['num_filters_5'] if 'num_filters_5' in config else None,
                             num_filters_6=config['num_filters_6'] if 'num_filters_6' in config else None,
                             kernel_size=3,
                             dropout_rate=config['dropout_rate'],
                             num_fc_units=config['num_fc_units'],
                             )

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        # criterion = torch.nn.CrossEntropyLoss()
        # if config['optimizer'] == 'Adam':
        #     optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        # else:
        #     optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])

        budget_data_num = math.floor((budget / 10) * self.N_train)
        # num_valid_data = math.floor((budget / 10) * self.N_valid)

        time_stamp = np.load('./temp/time_stamp.npy')
        if time.time() - time_stamp > 10000:
            print("time out")
            loss = np.inf
            return ({
                'loss': loss,  # remember: HpBandSter always minimizes!
                'info': {'test accuracy': 0,
                         'train accuracy': 0,
                         'validation accuracy': 0,
                         'model size': 0,
                         'training_time': 0,
                         'record_time': 0,
                         }

            })

        status_file = Path('./temp/status_file.pkl')

        if not status_file.is_file():
            already_used_data_num = 0
            status_training_time = 0
            train_data, valid_data = train_test_split(self.train_dataset,
                                                      train_size=(budget_data_num - already_used_data_num),
                                                      test_size=self.N_valid, stratify=self.train_dataset.targets)
            already_used_data_num = budget_data_num
            status_file = {
                config_id: {'already_used_data_num': already_used_data_num,
                            'training_time': status_training_time}}
        else:
            status_file = self.load_status_file('status_file')
            if config_id not in status_file:
                already_used_data_num = 0
                status_training_time = 0
                train_data, valid_data = train_test_split(self.train_dataset,
                                                          train_size=(budget_data_num - already_used_data_num),
                                                          test_size=self.N_valid, stratify=self.train_dataset.targets)
                already_used_data_num = budget_data_num
                status_file[config_id] = {'already_used_data_num': already_used_data_num,
                                          'training_time': status_training_time}
            else:
                already_used_data_num = status_file[config_id]['already_used_data_num']
                status_training_time = status_file[config_id]['training_time']
                train_data, valid_data = train_test_split(self.train_dataset,
                                                          train_size=(budget_data_num - already_used_data_num),
                                                          test_size=self.N_valid, stratify=self.train_dataset.targets)
                already_used_data_num = budget_data_num
                status_file[config_id]['already_used_data_num'] = already_used_data_num

        # train_data = torch.utils.data.Subset(self.train_dataset, range(already_used_data_num, current_iter_data_num))
        # valid_data = torch.utils.data.Subset(self.train_dataset, range(self.N_train, self.N_train + self.N_valid))

        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=self.batch_size,
                                                   shuffle=True, collate_fn=lambda x: tuple(
                x_.to(device) for x_ in default_collate(x)))
        validation_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=self.batch_size,
                                                        shuffle=True, collate_fn=lambda x: tuple(
                x_.to(device) for x_ in default_collate(x)))

        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=1024, shuffle=False,
                                                  collate_fn=lambda x: tuple(
                                                      x_.to(device) for x_ in default_collate(x)))

        model_parameters = Path('./temp/' + str(config_id) + '.pth')

        model.to(device)

        megabytes_of_model = self.compute_size(model)
        print("parameters size of model: %s MB" % megabytes_of_model)
        if megabytes_of_model > 10:
            print("model size bigger than 50MB")
            loss = np.inf
            return ({
                'loss': loss,  # remember: HpBandSter always minimizes!
                'info': {'test accuracy': 0,
                         'train accuracy': 0,
                         'validation accuracy': 0,
                         'model size': megabytes_of_model,
                         'training_time': 0,
                         'record_time': 0,
                         }

            })

        if not model_parameters.is_file():

            for epoch in range(self.num_epoch):

                iter_training_start = timeit.default_timer()

                loss = 0
                model.train()
                for i, (x, y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(x)
                    loss = F.nll_loss(output, y)
                    loss.backward()
                    optimizer.step()

                iter_training_stop = timeit.default_timer()
                iter_training_time = iter_training_stop - iter_training_start
                status_training_time = status_training_time + iter_training_time

                if status_training_time > np.inf:
                    print("training time is %s S so stop" % status_training_time)
                    loss = np.inf
                    return ({
                        'loss': loss,  # remember: HpBandSter always minimizes!
                        'info': {'test accuracy': 0,
                                 'train accuracy': 0,
                                 'validation accuracy': 0,
                                 'model size': megabytes_of_model,
                                 'training_time': status_training_time,
                                 'record_time': 0,
                                 }

                    })

            status_file[config_id]['training_time'] = status_training_time
            torch.save(model.state_dict(), str(model_parameters))

        else:

            model.load_state_dict(torch.load(str(model_parameters)))

            for epoch in range(self.num_epoch):

                iter_training_start = timeit.default_timer()

                loss = 0
                model.train()
                for i, (x, y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(x)
                    loss = F.nll_loss(output, y)
                    loss.backward()
                    optimizer.step()

                iter_training_stop = timeit.default_timer()
                iter_training_time = iter_training_stop - iter_training_start
                status_training_time = status_training_time + iter_training_time

                if status_training_time > np.inf:
                    print("training time is %s S so stop" % status_training_time)
                    loss = np.inf
                    return ({
                        'loss': loss,  # remember: HpBandSter always minimizes!
                        'info': {'test accuracy': 0,
                                 'train accuracy': 0,
                                 'validation accuracy': 0,
                                 'model size': megabytes_of_model,
                                 'training_time': status_training_time,
                                 'record_time': 0,
                                 }

                    })

            status_file[config_id]['training_time'] = status_training_time
            torch.save(model.state_dict(), str(model_parameters))

        print(iter_training_time)
        print(status_file)
        self.save_status_file(status_file, 'status_file')

        train_accuracy = self.evaluate_accuracy(model, train_loader)
        validation_accuracy = self.evaluate_accuracy(model, validation_loader)
        test_accuracy = self.evaluate_accuracy(model, test_loader)

        record_time = time.time() - time_stamp

        loss = 1 - validation_accuracy
        result = np.array(
            [loss, config_id, test_accuracy, status_training_time, record_time, megabytes_of_model, config,
             budget_data_num], dtype=object)
        # print(result)

        constraint_results = np.load('./temp/constraint_results.npy', allow_pickle=True)
        constraint_results = np.vstack((constraint_results, result))
        current_best_result = constraint_results[constraint_results[:, 0].argsort()][0]
        current_best_result[4] = record_time

        running_results = np.load('./running_results.npy', allow_pickle=True)
        running_results = np.vstack((running_results, current_best_result))

        print("current best results:", current_best_result)
        # print(constraint_results[constraint_results[:, 0].argsort()])
        np.save('./temp/constraint_results', constraint_results)
        np.save('./running_results', running_results)

        return ({
            'loss': loss,  # remember: HpBandSter always minimizes!
            'info': {'test accuracy': test_accuracy,
                     'train accuracy': train_accuracy,
                     'validation accuracy': validation_accuracy,
                     'model size': megabytes_of_model,
                     'training_time': status_training_time,
                     'record_time': record_time,
                     }

        })

    @staticmethod
    def save_status_file(obj, name):
        with open('./temp/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_status_file(name):
        with open('./temp/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def compute_size(self, model):
        re1 = model.number_of_parameters()
        return (re1 * 32) / (8 * (10 ** 6))

    def evaluate_accuracy(self, model, data_loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in data_loader:
                output = model(x)
                # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum().item()
        # import pdb; pdb.set_trace()
        accuracy = correct / len(data_loader.sampler)
        return (accuracy)

    @staticmethod
    def get_configspace():
        """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        # optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])
        #
        # sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9,
        #                                               log=False)

        # cs.add_hyperparameters([lr, optimizer, sgd_momentum])

        cs.add_hyperparameters([lr])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        # cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        # cs.add_condition(cond)

        num_conv_layers = CSH.UniformIntegerHyperparameter('num_conv_layers', lower=1, upper=6, default_value=3)

        num_filters_1 = CSH.UniformIntegerHyperparameter('num_filters_1', lower=4, upper=512, default_value=128,
                                                         log=True)
        num_filters_2 = CSH.UniformIntegerHyperparameter('num_filters_2', lower=4, upper=512, default_value=128,
                                                         log=True)
        num_filters_3 = CSH.UniformIntegerHyperparameter('num_filters_3', lower=4, upper=512, default_value=128,
                                                         log=True)
        num_filters_4 = CSH.UniformIntegerHyperparameter('num_filters_4', lower=4, upper=512, default_value=128,
                                                         log=True)
        num_filters_5 = CSH.UniformIntegerHyperparameter('num_filters_5', lower=4, upper=512, default_value=128,
                                                         log=True)
        num_filters_6 = CSH.UniformIntegerHyperparameter('num_filters_6', lower=4, upper=512, default_value=128,
                                                         log=True)

        cs.add_hyperparameters(
            [num_conv_layers, num_filters_1, num_filters_2, num_filters_3, num_filters_4, num_filters_5, num_filters_6])

        # You can also use inequality conditions:
        cond = CS.GreaterThanCondition(num_filters_2, num_conv_layers, 1)
        cs.add_condition(cond)

        cond = CS.GreaterThanCondition(num_filters_3, num_conv_layers, 2)
        cs.add_condition(cond)

        cond = CS.GreaterThanCondition(num_filters_4, num_conv_layers, 3)
        cs.add_condition(cond)

        cond = CS.GreaterThanCondition(num_filters_5, num_conv_layers, 4)
        cs.add_condition(cond)

        cond = CS.GreaterThanCondition(num_filters_6, num_conv_layers, 5)
        cs.add_condition(cond)

        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5,
                                                      log=False)
        num_fc_units = CSH.UniformIntegerHyperparameter('num_fc_units', lower=8, upper=1024, default_value=512,
                                                        log=True)

        cs.add_hyperparameters([dropout_rate, num_fc_units])

        return cs


class MNISTConvNet(torch.nn.Module):
    def __init__(self, num_conv_layers, num_filters_1, num_filters_2, num_filters_3, num_filters_4, num_filters_5,
                 num_filters_6, dropout_rate, num_fc_units,
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
        self.fc2 = nn.Linear(num_fc_units, 10)

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


if __name__ == "__main__":
    worker = PyTorchWorker(run_id='0')
    cs = worker.get_configspace()

    # config = cs.sample_configuration().get_dictionary()
    config = cs.get_default_configuration()
    start = timeit.default_timer()
    print(config)
    res = worker.compute(config=config, num_epoch=9, budget=10, working_directory='.')
    print(res)
    stop = timeit.default_timer()
    print("time:", (stop - start))
