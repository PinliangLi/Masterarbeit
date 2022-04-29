import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from autosklearn.metrics import balanced_accuracy

train_data = pd.read_csv('./aps_failure_training_set.csv')
train_y = train_data['class'].to_numpy()
train_x = train_data.iloc[:, 1:].replace('na', -1)
print(len(train_x))
train_x = train_x.astype(float)
train_x = train_x.to_numpy()

test_data = pd.read_csv('./aps_failure_test_set.csv')
test_y = test_data['class'].to_numpy()
test_x = test_data.iloc[:, 1:].replace('na', -1)
test_x = test_x.astype(float)
test_x = test_x.to_numpy()

num = 100

train_x = train_x[:num, :]
train_y = train_y[0:num]

print(train_x, train_y)
print(test_x, test_y)

print(len(train_x))
print(len(test_x))

model = RandomForestClassifier(n_estimators=10)
model.fit(train_x, train_y)
pred = model.predict(test_x)
pred[pred == 'neg'] = 0
pred[pred == 'pos'] = 1
test_y[test_y == 'neg'] = 0
test_y[test_y == 'pos'] = 1

comparison = pred == test_y
accuracy = comparison[comparison == True].shape[0] / comparison.shape[0]
print(pred)
print(accuracy)

print(balanced_accuracy(test_y.astype(float), pred.astype(float)))