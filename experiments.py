import utils
from config import datasets, algorithms
import numpy as np
from iforest import IForest
from sklearn.metrics import roc_auc_score
from dataset import get_train_test

for dataset in datasets:
    for algorithm in algorithms:

        utils.reset_random_state()

        X_train, X_test, y_train, y_test = get_train_test(dataset)

        iForest = algorithm()
        iForest.fit(X_train)

        y_train_pred = iForest.predict(X_train)
        y_test_pred = iForest.predict(X_test)

        train_auc = roc_auc_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_pred)

        print(iForest.name, dataset["name"], train_auc, test_auc)
