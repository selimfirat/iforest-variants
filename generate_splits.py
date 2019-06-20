from sklearn.utils import shuffle
import config, utils
from sklearn.model_selection import StratifiedKFold
import pickle

def generate_splits():

    datasets = {}

    skf = StratifiedKFold(n_splits=config.num_cv_splits)

    for dataset in config.datasets:
        utils.reset_random_state(1)

        X, y = utils.get_data(dataset)
        X, y = shuffle(X, y)

        splits_idxs = skf.split(X, y)

        splits = []

        for train, test in splits_idxs:
            X_train = X[train, :]
            y_train = y[train]

            X_test = X[test, :]
            y_test = y[test]

            split = (X_train, y_train, X_test, y_test)

            splits.append(split)

        datasets[dataset["name"]] = splits

    with open("data/splits.pkl", "wb+") as f:
        pickle.dump(datasets, f)

if __name__ == "__main__":
    generate_splits()