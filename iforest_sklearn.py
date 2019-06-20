from sklearn.ensemble import IsolationForest
from base_algorithm import BaseAlgorithm


class IForestSklearn(BaseAlgorithm):
    name = "iForest_sklearn"

    def __init__(self, t=100, psi=256):

        self.iforest = IsolationForest(max_samples=psi, n_estimators=t, behaviour="new", contamination="auto")

    def fit(self, X):

        self.iforest.fit(X)

    def predict(self, X):

        return -self.iforest.score_samples(X)