from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @property
    def name(self):
        raise NotImplementedError
