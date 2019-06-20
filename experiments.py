import utils, config

for dataset in config.datasets:
    for algorithm in config.algorithms:

        utils.reset_random_state()

        X_train, X_test, y_train, y_test = utils.get_train_test(dataset)

        algo = algorithm()

        algo.fit(X_train)

        y_train_pred = algo.predict(X_train)
        y_test_pred = algo.predict(X_test)

        stats = utils.calculate_stats(y_train, y_train_pred, y_test, y_test_pred)

        print(dataset["name"], algo.name, stats["train_auc"], stats["train_ap"], stats["test_auc"], stats["test_ap"])
