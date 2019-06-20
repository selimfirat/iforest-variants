from pyod.models.iforest import IForest
from base_algorithm import BaseAlgorithm


class IForestPyOD(BaseAlgorithm):
    name = "iForest_pyod"

    def __init__(self, t=100, psi=256):

        self.iforest = IForest(max_samples=psi, n_estimators=t)

    def fit(self, X):

        self.iforest.fit(X)

    def predict(self, X):

        return self.iforest.decision_function(X)