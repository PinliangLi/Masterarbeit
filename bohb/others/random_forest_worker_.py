import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

train_data = pd.read_csv('../../data/home-data-for-ml-course/train.csv')
train_data = train_data.dropna(axis=1)
train_data_X = train_data.drop(['Id', 'SalePrice'], axis=1)
train_data_y = train_data.SalePrice
print(train_data)
print(train_data_X)

for column in train_data_X:
    if train_data_X[column].dtype == 'object':
        translate_dict = {}
        class_num = 1
        for class_name, _ in train_data_X[column].value_counts().items():
            translate_dict[class_name] = class_num
            class_num += 1
        print(translate_dict)
        train_data_X[column] = train_data_X[column].replace(translate_dict)

print(train_data_X)

train_X, val_X, train_y, val_y = train_test_split(train_data_X, train_data_y, random_state = 0)
forest_model = RandomForestRegressor(n_estimators=1000, criterion="mse")
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
