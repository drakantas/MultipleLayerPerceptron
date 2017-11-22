import asyncio
import numpy as np
from json import load
from time import sleep
from pathlib import Path
from aiohttp.web import run_app
from datetime import datetime,  timedelta

from web_app.app import app, init_app
from util.dataset_parser import Parser
from net import __version__, MLP, identity_matrix


config = {
    'app_path': Path('./web_app'),
    'data_path': Path('./data/'),
    'dataset_path': Path('./data/fonts.json'),
    'net': {
        'char_shape': (9, 9),
        'cats': 7,
        'hidden_units': 14
    },
    'web': {
        'dsn': '',
        'host': '127.0.0.1',
        'port': 80
    }
}


def now() -> datetime:
    return datetime.utcnow() - timedelta(hours=5)  # GMT - 5


def parse_dataset_to_json():
    parser = Parser()
    parser.load(Path('./data/Clustered.xlsx'), 'Hoja1', out_path=config['dataset_path'])


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


def print_character(character: np.ndarray):
    for row in character:
        for i, val in enumerate(row):
            if val == 1:
                print('█', end='' if i != 8 else '\n')
            elif val == -1:
                print('-', end='' if i != 8 else '\n')
            else:
                print(' ', end='' if i != 8 else '\n')


def run_cli():
    mlp = MLP(input_unit_shape=config['net']['char_shape'], categories=config['net']['cats'],
              hidden_units=config['net']['hidden_units'], data_path=config['data_path'])

    print('----------\n'
          'Multiple Layer Perceptron Backpropagation v{1}\n'
          '{0:%d-%m-%Y %H:%M}\n'
          '----------'.format(now(), __version__))

    if not mlp.trained:
        mlp.import_net('7chars-100000')

    characters = get_dataset()[:, 0]  # Todos los caracteres del dataset

    for character in characters:
        print_character(character)
        print('----------\n',
              'Se identificó el caracter: {0}\n'.format(mlp.guess(character).upper()))

        sleep(2)  # 2 segundos hasta analizar siguiente caracter


def run_web_app():
    # Inicializar red neuronal
    net = MLP(input_unit_shape=config['net']['char_shape'], categories=config['net']['cats'],
              hidden_units=config['net']['hidden_units'], data_path=config['data_path'])

    # Cargar data preentrenada
    net.import_net('7chars-100000')

    loop = asyncio.get_event_loop()
    loop.run_until_complete(init_app(config['web'], config['app_path'], net))

    run_app(app, host=config['web']['host'], port=config['web']['port'])


if __name__ == '__main__':
    run_cli()
