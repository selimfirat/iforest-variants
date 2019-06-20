from config import datasets
import numpy as np
import time
from iforest import IForest
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import IsolationForest

np.random.seed(1)

scores = []
for dataset in datasets:
    X = np.loadtxt("data/" + dataset["file"], delimiter=",")
    y = np.array([0] * (len(X) - dataset["num_anomalies"]) + [1] * dataset["num_anomalies"])

    X, y = shuffle(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    iForest = IForest(t=100, psi=32)

    y_train_pred = iForest.train(X_train)

    y_test_pred = iForest.test(X_test)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    print(dataset["name"], train_auc, test_auc)


    iForest = IsolationForest(max_samples=32, n_estimators=100)
    iForest.fit(X_train)

    y_train_pred = -iForest.score_samples(X_train)

    y_test_pred = -iForest.predict(X_test)

    train_auc = roc_auc_score(y_train, y_train_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)

    print(dataset["name"], train_auc, test_auc)
