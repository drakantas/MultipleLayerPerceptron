import numpy as np
from sys import stdout
from pathlib import Path
from math import exp, log10
from json import loads, dumps
from typing import Union, Callable
from numpy.ctypeslib import ndarray

__all__ = ('MLP', 'identity_matrix')


def identity_matrix(n: int) -> ndarray:
    _identity_matrix = np.identity(n, dtype=int)
    _identity_matrix[_identity_matrix == 0] = -1
    return _identity_matrix


def bipolar_sigmoid(x: Union[float, int]) -> float:
    return 2 * (1 + exp(-1 * x)) ** -1 - 1


def d_bipolar_sigmoid(x: Union[float, int]) -> float:
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
                 unnester: Callable[[ndarray], ndarray] = _2d_unnester, data_path: Path = None):
        # Amount of units within the hidden layer
        self._hidden_units = hidden_units

        self._input_unit_shape = input_unit_shape

        self._input_units_amount = np.prod(input_unit_shape)

        # The array of weights is a tuple in which the first index refers to the input-to-hidden weights array and
        # the second and last index refers to the hidden-to-output weights array. Its shape should therefore be (2,n)
        # Step 0 - Initialization of random uniform weights.
        self._weights = (np.random.rand(self._hidden_units, self._input_units_amount),
                         np.random.rand(categories, self._hidden_units))

        # Biases
        self._bias = np.ones(2)

        self.trained = False

        # Callback that should return a 2d array of the data that will be trained.
        self._unnester = unnester

        # The purpose of the cache is to store the latest trained data array, amount of epochs
        # trained, and some other useful data.
        self._cache = {
            # Amount of epochs run in the last train() call
            'epochs': None,

            # Data run in the last train() call
            'data': None,

            # Net inputs
            'z_in': np.zeros(hidden_units),

            # Hidden layer inputs
            'y_in': np.zeros(categories),

            # Signals generated from broadcasting the input units to the hidden layer
            'hidden_signals': np.zeros(hidden_units),

            # Signals generated from broadcasting the hidden units to the output layer
            'output_signals': np.zeros(categories),

            # Deltas
            'deltas': np.zeros(categories),

            # Weight w correction terms
            'w_ct': np.zeros((categories, self._hidden_units)),

            # Weight v correction terms
            'v_ct': np.zeros((self._hidden_units, self._input_units_amount)),

            # Bias correction term of the w bunch
            'bw_ct': np.zeros(categories),

            # Bias correction term of the v bunch
            'bv_ct': np.zeros(self._hidden_units)
        }

        # Starter learning rate, decreases by .0005 after each epoch until it reaches 0.05
        self._learning_rate = 0.6

        # Path to directory that contains all sort of datasets used by the app
        self._data_path = data_path

    def train(self, data: ndarray, epochs: int = 100) -> None:
        counter = 0

        self._cache['epochs'] = epochs
        self._cache['data'] = data

        # Step 1 - Epoch iteration
        while True:
            self._epoch_progress(counter)

            # 1st stage - Feedforward computation
            # Step 2 - Iterate over each training pair [data - target]
            for pair in data:
                _data = self._unnester(pair[0])

                # Check that the amount of values within each data entry equals the input unit dimension
                assert len(_data) == self._weights[0].shape[1]

                # Steps 3,4 - We're going to handle the input unit broadcasting, following calculations and hidden unit
                # broadcasting together. Because as we know, each hidden unit should receive the whole array of input
                # units and this way seems easier.
                for hu, v in enumerate(self._weights[0]):
                    self._cache['z_in'][hu] = self._bias[0] + sum([_data[i] * v[i] for i, _ in enumerate(_data)])

                    self._cache['hidden_signals'][hu] = bipolar_sigmoid(self._cache['z_in'][hu])

                # Step 5 - Receive output signals broadcasted from the hidden layer and compute the
                # output signals.
                for ou, w in enumerate(self._weights[1]):
                    self._cache['y_in'][ou] = self._bias[1] + sum(self._cache['hidden_signals'] * w)

                    self._cache['output_signals'][ou] = bipolar_sigmoid(self._cache['y_in'][ou])

                # 2nd stage - Backpropagation of error
                # Step 6
                self._cache['deltas'] = (pair[1] - self._cache['output_signals']) * np.vectorize(d_bipolar_sigmoid)(
                    self._cache['y_in'])

                for i, _ in enumerate(self._cache['w_ct']):
                    delta = self._cache['deltas'][i]

                    self._cache['w_ct'][i] = self._learning_rate * delta * self._cache['hidden_signals']
                    self._cache['bw_ct'][i] = self._learning_rate * delta

                # Step 7
                for i in range(0, self._hidden_units):
                    d_in = sum(self._cache['deltas'] * self._weights[1][:, i])

                    d_signal = d_in * d_bipolar_sigmoid(d_in)

                    self._cache['v_ct'][i] = self._learning_rate * d_signal * _data
                    self._cache['bv_ct'][i] = self._learning_rate * d_signal

                # 3rd stage - Update weights
                # Step 8
                for cat, w_ct in enumerate(self._cache['w_ct']):
                    self._weights[1][cat] += w_ct
                    self._bias[1] += self._cache['bw_ct'][cat]

                for hi, v_ct in enumerate(self._cache['v_ct']):
                    self._weights[0][hi] += v_ct
                    self._bias[0] += self._cache['bv_ct'][hi]

            counter += 1

            if self._learning_rate > 0.05:
                self._learning_rate -= 0.0005

            if counter == epochs:
                if not self.trained:
                    self.trained = True  # The network has been trained
                break

    def guess(self, data: ndarray) -> str:
        assert data.shape == self._input_unit_shape
        assert self.trained is True

        character = _2d_unnester(data)
        z_in = list()
        z = list()
        y_in = list()
        y = list()
        deltas = list()
        targets = identity_matrix(7)

        for hu_i, v in enumerate(self._weights[0]):
            z_in.append(self._bias[0] + sum([character[i] * v[i] for i, _ in enumerate(character)]))
            z.append(bipolar_sigmoid(z_in[-1]))

        for ou_i, w in enumerate(self._weights[1]):
            y_in.append(self._bias[1] + sum(z * w))
            y.append(y_in[-1])

        for target in targets:
            deltas.append(target - y)

        # El delta menor significa el menor error
        # Seleccionar los deltas menores de los arreglos y de estos deltas coger el mayor,
        # Ese es el delta que buscamos para identificar el caracter.

        def min_delta(delta):
            _min_delta = None

            for i, d in enumerate(delta):
                if _min_delta is None:
                    _min_delta = [i, d]
                    continue

                if d < _min_delta[1]:
                    _min_delta[0] = i
                    _min_delta[1] = d

            return _min_delta

        max_delta = None

        for i, delta in enumerate(deltas):
            if max_delta is None:
                max_delta = [i, *min_delta(delta)]
                continue

            _min_d = min_delta(delta)

            if _min_d[1] > max_delta[2]:
                max_delta[0] = i
                max_delta[1] = _min_d[0]
                max_delta[2] = _min_d[1]

        return chr(97 + max_delta[1])

    def export_net(self, name: str):
        v = [[str(_v) for _v in r] for r in self._weights[0]]
        w = [[str(_v) for _v in r] for r in self._weights[1]]

        path = self._data_path.joinpath(Path('{}-{}epochs.json'.format(name, self._cache['epochs'])))

        path.write_text(dumps({
            'weights': {
                'v': v,
                'w': w
            },
            'biases': {
                'v': str(self._bias[0]),
                'w': str(self._bias[1])
            },
            'learning_rate': str(self._learning_rate)
        }, sort_keys=False, indent=4))

    def import_net(self, name: str):
        path = self._data_path.joinpath(Path('{}epochs.json'.format(name)))

        net = loads(path.read_text())

        self._bias[0] = net['biases']['v']
        self._bias[1] = net['biases']['w']

        self._learning_rate = net['learning_rate']

        for i, weights in enumerate(self._weights):
            for j, row in enumerate(weights):
                for k, column in enumerate(row):
                    if i == 0:
                        _weight = 'v'
                    else:
                        _weight = 'w'

                    self._weights[i][j][k] = net['weights'][_weight][j][k]

        self.trained = True

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
