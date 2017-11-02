import numpy as np
from math import exp
from numpy.ctypeslib import ndarray
from typing import Union


def identity_matrix(n: int) -> ndarray:
    _identity_matrix = np.identity(n, dtype=int)
    _identity_matrix[_identity_matrix == 0] = -1
    return _identity_matrix


def bipolar_sigmoid(x: Union[float, int]) -> float:
    return 2 * (1 + exp(-1 * x)) ** -1 - 1


def differentiated_bipolar_sigmoid(x: Union[float, int]) -> float:
    func = bipolar_sigmoid(x)
    return 0.5 * (1 + func) * (1 - func)


class MLP:
    def __init__(self, hidden_units: int = 30):
        self.hidden_units = hidden_units

    def train(self, data: ndarray, epochs: int = 100) -> None:
        pass
