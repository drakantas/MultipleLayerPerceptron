import numpy as np
from sys import stdout
from math import exp, log10
from numpy.ctypeslib import ndarray
from typing import Union, Callable

__all__ = ('MLP',)


def identity_matrix(n: int) -> ndarray:
    _identity_matrix = np.identity(n, dtype=int)
    _identity_matrix[_identity_matrix == 0] = -1
    return _identity_matrix


def bipolar_sigmoid(x: Union[float, int]) -> float:
    return 2 * (1 + exp(-1 * x)) ** -1 - 1


def differentiated_bipolar_sigmoid(x: Union[float, int]) -> float:
    func = bipolar_sigmoid(x)
    return 0.5 * (1 + func) * (1 - func)


def _2d_unnester(data: ndarray) -> ndarray:
    def _generator(data_: ndarray):
        for row, _ in enumerate(data_):
            for column, __ in enumerate(data_[row]):
                yield data_[row][column]

    return np.array(list(_generator(data)))


class MLP:
    def __init__(self, input_unit_shape: tuple = (9, 7), categories: int = 7, hidden_units: int = 30,
                 unnester: Callable[[ndarray], ndarray] = _2d_unnester):
        # Amount of units within the hidden layer
        self._hidden_units = hidden_units

        # The array of weights is a tuple in which the first index refers to the input-to-hidden weights array and
        # the second and last index refers to the hidden-to-output weights array. Its shape should therefore be (2,n)
        # Step 0 - Initialization of random uniform weights.
        self._weights = (np.random.rand(self._hidden_units, np.prod(input_unit_shape)),
                         np.random.rand(categories, self._hidden_units))

        self._trained = False

        # Callback that should return a 2d array of the data that will be trained.
        self._unnester = unnester

        # The purpose of the cache is to store the latest trained data array, amount of epochs
        # trained, and some other useful data.
        self._cache = dict()

    def train(self, data: ndarray, epochs: int = 100) -> None:
        counter = 0

        self._cache['epochs'] = epochs
        self._cache['data'] = data

        # Step 1 - Epoch iteration
        while True:
            self._epoch_progress(counter)

            # Step 2 - Iterate over each training pair [data - target]
            for pair in data:
                _data = self._unnester(pair[0])

                # Check that the amount of values within each data entry equals the input unit dimension
                assert len(_data) == self._weights[0].shape[1]

                # List of signals to broadcast to the output layer
                output_signals = list()

                # Steps 3,4 - We're going to handle the input unit broadcasting, following calculations and hidden unit
                # broadcasting together. Because as we know, each hidden unit should receive the whole array of input
                # units and this way seems easier.
                for hidden_unit in enumerate(self._weights[0]):
                    # hidden_unit is a Tuple of (index, weight_array)

                    z_in = 1 + sum([_data[i] * hidden_unit[1][i] for i, _ in enumerate(_data)])

                    output_signals.append(bipolar_sigmoid(z_in))  # Calculate signal and add it to the list

                # print(output_signals)

            counter += 1

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
            w_spaces = int(log10(_epoch)) - 14

            if w_spaces < 0:
                w_spaces = 0

            msg = '\rTraining has finished. {} epochs were run.{w_spaces}\n'.format(_epoch, w_spaces=' ' * w_spaces)

        stdout.write(msg)
        stdout.flush()
