import timeit
import math
import pickle
import sys
import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from autosklearn.metrics import balanced_accuracy


class Random_forest_worker(Worker):

    def __init__(self, *args, N_train=50000, N_valid=10000, **kwargs):
        super().__init__(*args, **kwargs)
        load_train_data = pd.read_csv('./aps_failure_training_set.csv')
        train_x = load_train_data.iloc[:, 1:].replace('na', -1)
        train_x = train_x.astype(float)
        self.train_x = train_x.to_numpy()
        self.train_y = load_train_data['class'].to_numpy()

        test_data = pd.read_csv('./aps_failure_test_set.csv')
        self.test_y = test_data['class'].to_numpy()
        test_x = test_data.iloc[:, 1:].replace('na', -1)
        test_x = test_x.astype(float)
        self.test_x = test_x.to_numpy()

        self.test_y[self.test_y == 'neg'] = 0
        self.test_y[self.test_y == 'pos'] = 1

        self.N_train = N_train
        self.N_valid = N_valid


    def compute(self, config_id, config, budget, **kwargs):

        num_train_data = math.floor((budget / 10) * self.N_train)

        _train_data = np.hstack((self.train_x, self.train_y.reshape((-1, 1))))
        train_data, valid_data = train_test_split(_train_data, train_size=num_train_data, test_size=self.N_valid,
                                                  stratify=self.train_y)

        train_data_x = train_data[:, :-1]
        train_data_y = train_data[:, -1].flatten()

        valid_data_x = valid_data[:, :-1]
        valid_data_y = valid_data[:, -1].flatten()

        time_stamp = np.load('./temp/time_stamp.npy')
        model = RandomForestClassifier(n_estimators=config['n_estimators'],
                                       min_samples_split=config['min_samples_split'], criterion=config['criterion'])

        start = timeit.default_timer()

        #model = RandomForestClassifier(n_estimators=config['n_estimators'], min_samples_leaf=config['min_samples_leaf'], min_samples_split=config['min_samples_split'], criterion=config['criterion'])
        model.fit(train_data_x, train_data_y)

        stop = timeit.default_timer()

        training_time = stop - start

        p = pickle.dumps(model)
        model_size = sys.getsizeof(p) / (1024 ** 2)

        pred_valid = model.predict(valid_data_x)
        pred_test = model.predict(self.test_x)

        pred_valid[pred_valid == 'neg'] = 0
        pred_valid[pred_valid == 'pos'] = 1

        pred_test[pred_test == 'neg'] = 0
        pred_test[pred_test == 'pos'] = 1

        valid_data_y[valid_data_y == 'neg'] = 0
        valid_data_y[valid_data_y == 'pos'] = 1

        self.test_y[self.test_y == 'neg'] = 0
        self.test_y[self.test_y == 'pos'] = 1

        valid_data_y = valid_data_y.astype(float)

        test_y = self.test_y.astype(float)

        pred_valid = pred_valid.astype(float)
        pred_test = pred_test.astype(float)

        tn1, fp1, fn1, tp1 = confusion_matrix(valid_data_y, pred_valid).ravel()
        tn2, fp2, fn2, tp2 = confusion_matrix(test_y, pred_test).ravel()

        balanced_acc_valid = ((tp1 / (tp1 + fn1)) + (tn1 / (fp1 + tn1))) / 2
        balanced_acc_test = ((tp2 / (tp2 + fn2)) + (tn2 / (fp2 + tn2))) / 2


        print('num of train pos :', len(train_data_y[train_data_y == 'pos']))
        print('num of test pos :', len(pred_test[pred_test == 1]))
        print('num of test neg :', len(pred_test[pred_test == 0]))
        print('num of train :', num_train_data)

        print('-------------------------------------------')
        print('Config :', config)
        print('model size (MB):', model_size)
        print('training time :', training_time)
        print('valid balanced accuracy :', balanced_acc_valid)
        print('test balanced accuracy :', balanced_acc_test)
        comparison = pred_test == self.test_y
        test_accuracy = comparison[comparison == True].shape[0] / comparison.shape[0]
        # print(pred)
        print('test_accuracy :', test_accuracy)
        print('-------------------------------------------')

        record_time = time.time() - time_stamp

        if training_time > 100: #or model_size > 10:
            loss = np.inf
        else:
            loss = 1 - balanced_acc_valid
            result = np.array([loss, config_id, balanced_acc_test, training_time, record_time, model_size, config,
                               num_train_data], dtype=object)

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
                     'balanced accuracy': balanced_acc_test,
                     'model size': model_size,
                     'training_time': training_time,
                     'record_time': record_time,
                     }

        })

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        n_estimators = CSH.UniformIntegerHyperparameter('n_estimators', lower=1, upper=1000, default_value=100)
        min_samples_split = CSH.UniformIntegerHyperparameter('min_samples_split', lower=2, upper=10, default_value=5)
        #min_samples_leaf = CSH.UniformFloatHyperparameter('min_samples_leaf', lower=0.001, upper=0.5, default_value=0.1)
        criterion = CSH.CategoricalHyperparameter('criterion', ['gini', 'entropy'])

        #cs.add_hyperparameters([n_estimators, min_samples_leaf, min_samples_split, criterion])
        cs.add_hyperparameters([n_estimators, min_samples_split, criterion])


        return cs

if __name__ == "__main__":
    worker = Random_forest_worker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)