import numpy as np
import time
import timeit
import sys

import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import pickle
from emukit.core.continuous_parameter import ContinuousParameter
from emukit.examples.gp_bayesian_optimization.unknown_constraint_bayesian_optimization \
    import UnknownConstraintGPBayesianOptimization
from emukit.core.loop import UserFunctionResult


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

    load_train_data = pd.read_csv('./aps_failure_training_set.csv')
    train_x = load_train_data.iloc[:, 1:].replace('na', -1)
    train_x = train_x.astype(float)
    train_x = train_x.to_numpy()
    train_y = load_train_data['class'].to_numpy()

    test_data = pd.read_csv('./aps_failure_test_set.csv')
    test_y = test_data['class'].to_numpy()
    test_x = test_data.iloc[:, 1:].replace('na', -1)
    test_x = test_x.astype(float)
    test_x = test_x.to_numpy()


    print('Evaluating at ')
    config = {'n_estimators': int(params[0]), 'min_samples_split': int(params[1]),
              'criterion': int(params[2]), 'num_training_data': int(params[3])}
    print(config)
    if config['criterion'] < 1:
        config['criterion'] = 'gini'
    else:
        config['criterion'] = 'entropy'

    _train_data = np.hstack((train_x, train_y.reshape((-1, 1))))
    train_data, valid_data = train_test_split(_train_data, train_size=config['num_training_data'], test_size=0.1, stratify=train_y)

    train_data_x = train_data[:, :-1]
    train_data_y = train_data[:, -1].flatten()

    valid_data_x = valid_data[:, :-1]
    valid_data_y = valid_data[:, -1].flatten()


    model = RandomForestClassifier(n_estimators=config['n_estimators'], min_samples_split=config['min_samples_split'], criterion=config['criterion'])


    start = timeit.default_timer()

    model.fit(train_data_x, train_data_y)

    stop = timeit.default_timer()

    training_time = stop - start

    pred_valid = model.predict(valid_data_x)
    pred_test = model.predict(test_x)

    pred_valid[pred_valid == 'neg'] = 0
    pred_valid[pred_valid == 'pos'] = 1

    pred_test[pred_test == 'neg'] = 0
    pred_test[pred_test == 'pos'] = 1

    valid_data_y[valid_data_y == 'neg'] = 0
    valid_data_y[valid_data_y == 'pos'] = 1


    test_y[test_y == 'neg'] = 0
    test_y[test_y == 'pos'] = 1



    valid_data_y = valid_data_y.astype(float)

    test_y = test_y.astype(float)

    pred_valid =pred_valid.astype(float)
    pred_test = pred_test.astype(float)

    tn1, fp1, fn1, tp1 = confusion_matrix(valid_data_y, pred_valid).ravel()
    tn2, fp2, fn2, tp2 = confusion_matrix(test_y, pred_test).ravel()

    balanced_acc_valid = ((tp1 / (tp1 + fn1)) + (tn1 / (fp1 + tn1))) / 2
    balanced_acc_test = ((tp2 / (tp2 + fn2)) + (tn2 / (fp2 + tn2))) / 2

    p = pickle.dumps(model)
    model_size = sys.getsizeof(p) / (1024 ** 2)

    comparison = pred_test == test_y
    test_accuracy = comparison[comparison == True].shape[0] / comparison.shape[0]


    constraint_value = 0
    satisfied_constraints = 0

    if training_time < 100:
        constraint_value = constraint_value + (training_time - 100)
        satisfied_constraints = satisfied_constraints + 1

    if model_size < np.inf:
        #constraint_value = constraint_value + (model_size - 10)
        satisfied_constraints = satisfied_constraints + 1

    if satisfied_constraints != 2:
        constraint_value = 1

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

    print('valid balanced accuracy: %s test balanced accuracy: %s test accuracy: %s test_traning_time: %s model_size: %s' % (
        balanced_acc_valid, balanced_acc_test, test_accuracy, training_time, model_size))
    # validation_accuracy = 0.001
    return 1 - balanced_acc_valid, balanced_acc_test, constraint_value, time_stamp, training_time, model_size


def init():
    init_n_estimators = np.random.randint(low=1, high=1000, size=(5, 1))
    init_min_samples_split = np.random.randint(low=2, high=10, size=(5, 1))
    init_criterion = np.random.randint(low=0, high=2, size=(5, 1))
    init_num_training_data = np.random.randint(low=6250, high=50000, size=(5, 1))

    init_config = np.hstack(
        (init_n_estimators, init_min_samples_split, init_criterion, init_num_training_data))
    init_loss = []
    init_c_value = []
    for i in init_config:
        loss, test_accuracy, constraint_value, time_stamp, _t, _m = evaluate(i)
        init_loss.append(loss)
        # init_constraints.append([training_time, model_size])
        init_c_value.append(constraint_value)

    init_loss = np.array(init_loss).reshape((5, 1))
    init_c_value = np.array(init_c_value).reshape((5, 1))
    return init_config, init_loss, init_c_value


def create_config_space():
    n_estimators = ContinuousParameter('n_estimators', 1, 1000)
    min_samples_split = ContinuousParameter('min_samples_split', 2, 10)
    criterion = ContinuousParameter('criterion', 0, 2)
    max_num_training_data = np.log(50000)
    min_num_training_data = np.log(6250)
    num_training_data = ContinuousParameter('num_training_data', min_num_training_data, max_num_training_data)
    #num_training_data = ContinuousParameter('num_training_data', 6250, 50000)
    config_space = [n_estimators, min_samples_split, criterion, num_training_data]

    return config_space


if __name__ == '__main__':
    constraint_results = np.array([np.inf, 0, 0, 0, 0, []], dtype=object)
    record_results = np.array([np.inf, 0, 0, 0, 0, []], dtype=object)

    n_iterations = 100

    init_config, init_loss, init_c_value = init()

    config_space = create_config_space()
    bo = UnknownConstraintGPBayesianOptimization(variables_list=config_space, X=init_config, Y=init_loss,
                                                 Yc=init_c_value, batch_size=1)
    results = None
    best_result = np.inf
    start_time = time.time()
    for _ in range(n_iterations + 1):
        X_new = bo.get_next_points(results)
        new_config = [round(X_new[0][0]), round(X_new[0][1]), round(X_new[0][2]), round(X_new[0][3])]
        new_config[3] = round(np.power(np.e, X_new[0][3]))
        _loss, balanced_acc, c_value, time_stamp, training_time, model_size = evaluate(new_config)
        record_time = time_stamp - start_time
        loss = np.array(_loss).reshape((1, 1))
        c_value = np.array(c_value).reshape((1, 1))

        print("constraint value :", c_value)
        if c_value < 0:
            constraint_results = np.vstack((constraint_results, np.array([_loss, balanced_acc, training_time, record_time, model_size, new_config], dtype=object)))
            current_best_results = constraint_results[constraint_results[:, 0].argsort()][0]
            current_best_results[3] = record_time
            record_results = np.vstack((record_results, current_best_results))
            print('record time:', record_time)

            if best_result > loss:
                best_config = new_config
                best_result = loss
                print("best result: ")
                print("------------------------------------------------------------")
                print("best config: {} best result: {} traning_time: {} model size: {} record time {}".format(best_config, best_result, training_time, model_size, record_time))
        results = [UserFunctionResult(X_new[0], loss[0], Y_constraint=c_value[0])]


    np.save('./running_results_new.npy', record_results)
