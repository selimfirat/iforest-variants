import eif
from base_algorithm import BaseAlgorithm


class IForestExtended(BaseAlgorithm):
    name = "iForest_extended"

    def __init__(self, t=100, psi=128):

        self.t = t
        self.psi = psi
        self.iforest = None

    def fit(self, X):

        self.iforest = eif.iForest(X, ntrees=self.t, sample_size=self.psi, ExtensionLevel=X.shape[1]-1)


    def predict(self, X):

        return self.iforest.compute_paths(X_in=X)