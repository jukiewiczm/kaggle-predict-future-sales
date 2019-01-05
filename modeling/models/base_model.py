from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self, train, y_train, valid_set_tuple=None):
        pass

    @abstractmethod
    def transform(self, test):
        pass
