"""
Interface that all encapsulated models should implement
"""
from abc import ABC, abstractmethod

class InterfaceModel(ABC):
    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def train(self, train_data, eval_data):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def diagnose(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass
