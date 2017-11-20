import numpy as np
from pathlib import Path
from datetime import datetime,  timedelta

from net import MLP, __version__
from util.dataset_parser import Parser


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
            np.array([-1, -1, -1, -1, -1, -1, 1, -1, -1]),
            np.array([-1, -1, 1, 1, 1, 1, 1, -1, -1]),
            np.array([-1, -1, 1, -1, 1, -1, 1, -1, -1]),
            np.array([-1, -1, 1, -1, -1, -1, 1, -1, -1]),
            np.array([1, -1, 1, 1, 1, 1, 1, -1, -1]),
            np.array([-1, -1, 1, -1, -1, 1, 1, -1, -1]),
            np.array([-1, -1, 1, 1, -1, -1, 1, -1, -1])
        ],
        [1, -1, -1, -1, -1, -1, -1]
    ])
])


mlp = MLP()

if __name__ == '__main__':
    print('----------\n'
          'Multiple Layer Perceptron Backpropagation v{1}\n'
          '{0:%d-%m-%Y %H:%M}\n'
          '----------'.format(now(), __version__))
    parser = Parser()
    parser.load(Path('./dataset/Clustered.xlsx'), 'Hoja1')
    #mlp.train(data, 1000)
