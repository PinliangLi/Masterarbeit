import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import math
from sklearn.metrics import mean_absolute_error
import pickle
from autosklearn.metrics import balanced_accuracy

all_results_1 = np.load('./all_results_500_50_1.npy', allow_pickle=True)
all_results_1 = np.delete(all_results_1, 0, 0)

all_results_2 = np.load('./all_results_500_50_2.npy', allow_pickle=True)
all_results_2 = np.delete(all_results_1, 0, 0)

all_results_3 = np.load('./all_results_250_50.npy', allow_pickle=True)
all_results_3 = np.delete(all_results_1, 0, 0)
all_results_3[:, -3] = 250

all_results_4 = np.load('./all_results_250_50_1.npy', allow_pickle=True)
all_results_4 = np.delete(all_results_1, 0, 0)
all_results_4[:, -3] = 250

all_results_5 = np.load('./all_results_250_50_2.npy', allow_pickle=True)
all_results_5 = np.delete(all_results_1, 0, 0)
all_results_5[:, -3] = 250


all_results_c = np.concatenate((all_results_1, all_results_2, all_results_3, all_results_4, all_results_5))

target = all_results_c[:, -1]
train_data_X = all_results_c[:, 6:-1]

# print(train_data_X, target)
# print(len(target))

train_X, val_X, train_y, val_y = train_test_split(train_data_X, target, random_state = 0)
forest_model = RandomForestRegressor(n_estimators=1000)
forest_model.fit(train_X, train_y)
filename = 'finalized_model'
pickle.dump(forest_model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
melb_preds = loaded_model.predict(val_X)
melb_preds[melb_preds >= 0.5] = 1
melb_preds[melb_preds < 0.5] = 0
comparison = melb_preds == val_y
accuracy = comparison[comparison == True].shape[0] / comparison.shape[0]
print(melb_preds)
print(val_y)
print(comparison)
print(accuracy)
# #print(balanced_accuracy(val_y, melb_preds))