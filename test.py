from mlp import MLP
from datetime import datetime,  timedelta


def now() -> datetime:
    return datetime.utcnow() - timedelta(hours=5)  # GMT - 5


mlp = MLP()

if __name__ == '__main__':
    print('----------\n'
          'Multiple Layer Perceptron Backpropagation\n'
          '{0:%d-%m-%Y %H:%M}\n'
          '----------'.format(now()))
    mlp.train([], 15)
    print('mmm')
