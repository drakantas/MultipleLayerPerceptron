import numpy as np
from datetime import datetime,  timedelta

from mlp import MLP


def now() -> datetime:
    return datetime.utcnow() - timedelta(hours=5)  # GMT - 5


data = np.array([
    np.array([
        [
            np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1]),
            np.array([-1, -1, 1, 1, 1, 1, 1, -1, -1]),
            np.array([-1, -1, 1, -1, -1, -1, 1, -1, -1]),
            np.array([-1, -1, 1, -1, -1, -1, 1, -1, -1]),
            np.array([-1, -1, 1, 1, 1, 1, 1, -1, -1]),
            np.array([-1, -1, 1, -1, -1, -1, 1, -1, -1]),
            np.array([-1, -1, 1, -1, -1, -1, 1, -1, -1])
        ],
        [1, -1, -1, -1, -1, -1, -1]
    ]),
    np.array([
        [
            np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1]),
            np.array([-1, -1, 1, 1, 1, 1, 1, -1, -1]),
            np.array([-1, -1, 1, -1, -1, -1, 1, -1, -1]),
            np.array([-1, -1, 1, -1, -1, -1, 1, -1, -1]),
            np.array([-1, -1, 1, 1, 1, 1, 1, -1, -1]),
            np.array([-1, -1, 1, -1, -1, -1, 1, -1, -1]),
            np.array([-1, -1, 1, -1, -1, -1, 1, -1, -1])
        ],
        [1, -1, -1, -1, -1, -1, -1]
    ])
])


mlp = MLP()

if __name__ == '__main__':
    print('----------\n'
          'Multiple Layer Perceptron Backpropagation\n'
          '{0:%d-%m-%Y %H:%M}\n'
          '----------'.format(now()))
    mlp.train(data, 15)
