import numpy as np
from json import load
from pathlib import Path
from datetime import datetime,  timedelta

from net import __version__, MLP, identity_matrix
from util.dataset_parser import Parser


config = {
    'dataset_path': Path('./dataset/fonts.json'),
    'net': {
        'char_shape': (9, 9),
        'cats': 7,
        'hidden_units': 14
    }
}


def now() -> datetime:
    return datetime.utcnow() - timedelta(hours=5)  # GMT - 5


def parse_dataset_to_json():
    parser = Parser()
    parser.load(Path('./dataset/Clustered.xlsx'), 'Hoja1', out_path=config['dataset_path'])


def get_dataset():
    with config['dataset_path'].open() as f:
        dataset = load(f)

    def get_pairs():
        for font in dataset:
            characters = np.array([np.array([np.array(r) for r in c]) for c in font])
            targets = identity_matrix(7)

            for i, _ in enumerate(targets):
                yield characters[i], targets[i]

    return np.array(list(get_pairs()))


def run_cli():
    mlp = MLP(input_unit_shape=config['net']['char_shape'], categories=config['net']['cats'],
              hidden_units=config['net']['hidden_units'])

    print('----------\n'
          'Multiple Layer Perceptron Backpropagation v{1}\n'
          '{0:%d-%m-%Y %H:%M}\n'
          '----------'.format(now(), __version__))

    if not mlp.trained:
        mlp.train(get_dataset(), epochs=100000)


def run_web_app():
    app = App()


class App:
    def __init__(self):
        self.net = MLP(input_unit_shape=config['net']['char_shape'], categories=config['net']['cats'],
                       hidden_units=config['net']['hidden_units'])


if __name__ == '__main__':
    run_cli()
