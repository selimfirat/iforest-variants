from iforest import IForest
from iforest_pyod import IForestPyOD
from iforest_sklearn import IForestSklearn

datasets = [
    {
        "name": "madelon",
        "file": "madelon_sampled.txt",
        "num_anomalies": 130,
        "active": True
    },
    {
        "name": "ionosphere",
        "file": "ionosphere_sampled.txt",
        "num_anomalies": 17,
        "active": True
    },
    {
        "name": "telescope",
        "file": "magic-telescope_sampled.txt",
        "num_anomalies": 951,
        "active": True
    },
    {
        "name": "indians",
        "file": "pima-indians_sampled.txt",
        "num_anomalies": 38,
        "active": True
    }
]

algorithms = [IForest, IForestSklearn, IForestPyOD, ]
