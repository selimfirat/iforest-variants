import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def label_first_N_anomaly(anomaly_scores, N): # N anomalies

    mask = np.argsort(-anomaly_scores)[:N]

    y = np.zeros(anomaly_scores.shape[0])

    y[mask] = 1

    return y


def reset_random_state():

    np.random.seed(1)


def calculate_stats(y_train, y_train_pred, y_test, y_test_pred, y_val=None, y_val_pred=None):

    stats = {}
    stats["train_auc"] = roc_auc_score(y_train, y_train_pred)
    stats["train_ap"] = average_precision_score(y_train, y_train_pred)

    stats["test_auc"] = roc_auc_score(y_test, y_test_pred)
    stats["test_ap"] = average_precision_score(y_test, y_test_pred)


    if y_val != None and y_val_pred != None:
        stats["val_auc"] = roc_auc_score(y_val, y_val_pred)
        stats["val_ap"] = average_precision_score(y_val, y_val_pred)

    return stats


def get_train_test(dataset):
    X = np.loadtxt("data/" + dataset["file"], delimiter=",")
    y = np.array([0] * (len(X) - dataset["num_anomalies"]) + [1] * dataset["num_anomalies"])

    X, y = shuffle(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test
