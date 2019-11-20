from enum import Enum
from typing import Union

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import torch
from torch import Tensor

TorchOrNumpy = Union[np.ndarray, Tensor]


class BlackBoxDetector:
    class Type(Enum):
        DecisionTree = DecisionTreeClassifier()
        LogisticRegression = LogisticRegression(solver='lbfgs', max_iter=int(1e6))
        MultiLayerPerceptron = MLPClassifier()
        RandomForest = RandomForestClassifier(n_estimators=100)
        SVM = SVC(gamma="auto")

        @classmethod
        def names(cls):
            return [c.name for c in cls]

        @classmethod
        def get_classifier_from_name(cls, name):
            for c in BlackBoxDetector.Type:
                if c.name == name:
                    return c
            raise ValueError("Unknown enum \"%s\" for class \"%s\"", name, cls.name)

    def __init__(self, learner_type: 'BlackBoxDetector.Type'):
        self.type = learner_type
        self._model = sklearn.clone(self.type.value)
        self.training = True

    def fit(self, train_data: TorchOrNumpy, train_labels: TorchOrNumpy):
        if isinstance(train_data, Tensor):
            train_data = train_data.cpu().numpy()

        if isinstance(train_labels, Tensor):
            train_labels = train_labels.cpu().numpy()

        self._model.fit(train_data, train_labels)
        self.training = False

    def predict(self, test_data: TorchOrNumpy) -> Tensor:
        if self.training:
            raise ValueError("Detector does not appear to be trained but trying to predict")

        if torch.cuda.is_available():
            test_data = test_data.cpu()

        if isinstance(test_data, Tensor):
            test_data = test_data.numpy()

        predictions = torch.from_numpy(self._model.predict(test_data)).float()

        return predictions.cuda() if torch.cuda.is_available() else predictions
