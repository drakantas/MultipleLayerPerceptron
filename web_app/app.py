import re
import numpy as np
from typing import Union
from pathlib import Path
from asyncpg import create_pool
from jinja2 import FileSystemLoader
from aiohttp.web import Application, Request
from aiohttp_jinja2 import setup as setup_jinja, template

__all__ = ('app', 'init_app')


matrix_cell_pattern = r'cell-([0-8])-([0-8])'

app = Application()


@template('character.html')
async def home_handler(request: Request):
    return {}


@template('character.html')
async def upload_character_handler(request: Request):
    data = await request.post()

    def get_matrix(data_: dict) -> Union[np.ndarray, bool, None]:
        if not data_:
            return None

        matrix = np.array([np.array([-1 for _c in range(0, 9)]) for _ in range(0, 9)])
        pattern = re.compile(matrix_cell_pattern)

        for k, v in data_.items():
            cell = pattern.fullmatch(k)

            if not cell:
                continue

            if v != 'on':
                return False

            matrix[int(cell.group(1))][int(cell.group(2))] = 1

        return matrix

    data = get_matrix(data)

    if data is None:
        return {'error': 'No puedes dejar la matriz en blanco.'}
    elif data is False:
        return {'error': 'Has enviado data incorrecta.'}

    return {'character': data,
            'guess': request.app.net.guess(data)}


async def setup_db(app_: Application, dsn: str):
    setattr(app_, 'db', await create_pool(dsn=dsn))


async def setup_neural_network(app_: Application, net):
    setattr(app_, 'net', net)


async def init_app(config: dict, app_path: Path, net):
    # await setup_db(app, config['dsn'])
    await setup_neural_network(app, net)

    setup_jinja(app, loader=FileSystemLoader(str(app_path.joinpath('./views').resolve())))

    static_path = app_path.joinpath(Path('./static'))
    app.router.add_static('/static', static_path)

    app.router.add_get('/', home_handler)
    app.router.add_post('/identify-character', upload_character_handler)
