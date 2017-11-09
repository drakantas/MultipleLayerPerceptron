import numpy as np
from time import sleep
from sys import stdout
from math import exp, log10
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

        self._trained = False

        # The purpose of the cache is to store the latest trained data array, amount of epochs
        # trained, and some other useful data.
        self._cache = dict()

    def train(self, data: ndarray, epochs: int = 100) -> None:
        counter = 0

        self._cache['epochs'] = epochs
        self._cache['data'] = data

        while True:
            self._epoch_progress(counter)
            counter += 1
            sleep(1)
            if counter == epochs:
                if not self._trained:
                    self._trained = True  # The network has been trained
                break

    def _epoch_progress(self, epoch: int):
        assert self._cache
        _epoch = epoch + 1

        if _epoch != self._cache['epochs']:
            msg = '\rTraining epoch {} out of {}...'.format(_epoch, self._cache['epochs'])
        else:
            msg = '\rTrained {} epochs.{whitespaces}\n'.format(_epoch, whitespaces=' ' * (11 + int(log10(_epoch))))

        stdout.write(msg)
        stdout.flush()
