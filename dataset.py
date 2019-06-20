import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def get_train_test(dataset):
    X = np.loadtxt("data/" + dataset["file"], delimiter=",")
    y = np.array([0] * (len(X) - dataset["num_anomalies"]) + [1] * dataset["num_anomalies"])

    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test