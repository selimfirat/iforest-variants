import utils, config

scores = {}
for dataset in config.datasets:

    scores[dataset["name"]] = {}

    for algorithm in config.algorithms:

        statslist = []

        splits = utils.get_splits(dataset)

        for X_train, y_train, X_test, y_test in splits:

            for random_state in config.random_states:

                utils.reset_random_state(random_state)

                algo = algorithm()

                algo.fit(X_train)

                y_train_pred = algo.predict(X_train)
                y_test_pred = algo.predict(X_test)

                stats = utils.calculate_stats(y_train, y_train_pred, y_test, y_test_pred)

                statslist.append(stats)

        scores[dataset["name"]][algorithm.name] = {}

        for k in statslist[0].keys():
            scores[dataset["name"]][algorithm.name][k] = 1.0*sum(s[k] for s in statslist)/len(statslist)

        print(dataset["name"], algorithm.name, scores[dataset["name"]][algorithm.name]["train_auc"], scores[dataset["name"]][algorithm.name]["train_ap"], scores[dataset["name"]][algorithm.name]["test_auc"], scores[dataset["name"]][algorithm.name]["test_ap"])

print(scores)