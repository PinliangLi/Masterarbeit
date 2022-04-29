from autosklearn.ensembles.ensemble_selection import EnsembleSelection
from autosklearn.metrics import balanced_accuracy
from autosklearn.constants import BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn import datasets


X, y = datasets.load_breast_cancer(return_X_y=True)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

models = []
models.append(KNeighborsClassifier().fit(X_train, y_train))
models.append(DecisionTreeClassifier().fit(X_train, y_train))
models.append(RandomForestClassifier().fit(X_train, y_train))

validation_predictions = []
test_predictions = []
for i in range(len(models)):
    validation_predictions.append(models[i].predict(X_valid))
    #print(models[i].predict(X_valid).dtype)
    test_predictions.append(models[i].predict(X_test))

ensemble_selection = EnsembleSelection(ensemble_size=10,
                             task_type=BINARY_CLASSIFICATION,
                             random_state=0,
                             metric=balanced_accuracy)

print(validation_predictions, y_valid.dtype)
ensemble_selection.fit(validation_predictions, y_valid, identifiers=None)
y_hat_ensemble = ensemble_selection.predict(np.array(validation_predictions))
y_hat_test = ensemble_selection.predict(np.array(test_predictions))

for i in range(len(models)):
    print(balanced_accuracy(y_test, test_predictions[i]))
print('\n\n')
print(balanced_accuracy(y_test, y_hat_test))
