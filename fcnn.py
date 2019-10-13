from typing import Tuple

import numpy as np


class FCNN:
    """Fully-connected neural network.
    """

    def __init__(self) -> None:
        self.l1 = FCLayer(shape=(3, 4))
        self.l2 = FCLayer(shape=(4, 1))

    def __forward_pass(self, X) -> None:
        self.l1.get_output(X)
        self.l2.get_output(self.l1.output)

    def __backward_pass(self, X, y) -> None:
        self.l2_delta = (y - self.l2.output) * (self.l2.output * (1 - self.l2.output))
        self.l1_delta = self.l2_delta.dot(self.l2.weights.T) * (self.l1.output * (1 - self.l1.output))

        self.l2.update_weights(self.l1.output, self.l2_delta)
        self.l1.update_weights(X, self.l1_delta)

    def fit(self, X: np.array, y: np.array, n_epochs: int = 1000) -> 'FCNN':
        """Trains this little FC network.

        :param X: input data
        :param y: output to approximate
        :param n_epochs: number of epochs
        :return: fitted model
        """
        for j in range(n_epochs):
            self.__forward_pass(X)
            self.__backward_pass(X, y)

        return self

    def predict(self, X: np.array) -> np.array:
        """Predicts y given X.

        :param X: input data
        :return: predictions
        """
        self.__forward_pass(X)
        return self.l2.output


class FCLayer:
    """One fully-connected layer.
    """
    def __init__(self, shape: Tuple[int, int]):
        self.weights = 2 * np.random.random(shape) - 1
        self.output = None

    def update_weights(self, input_array, delta) -> np.array:
        self.weights += input_array.T.dot(delta)
        return self.weights

    def get_output(self, input_array: np.array) -> np.array:
        self.output = nonlin(np.dot(input_array, self.weights))
        return self.output


def nonlin(x: np.array) -> float:
    """Sygmoid for the layer's output.

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def sygmoid_derivative(x: np.array) -> np.array:
    """Derivative of sygmoid function.

    :param x:
    :return:
    """
    return x * (1 - x)
