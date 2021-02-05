import argparse
import inspect
from data import OfflineData


def main(input_dir='./data',
         batch_length=20,
         batch_size=10):

    data = OfflineData(input_dir)

    for batch in data.iterate(batch_length, batch_size):
        
        print(batch['image'].shape)


if __name__ == '__main__':
    # Use main() kwargs as config
    parser = argparse.ArgumentParser()
    argspec = inspect.getfullargspec(main)
    for key, value in zip(argspec.args, argspec.defaults):
        parser.add_argument(f'--{key}', type=type(value), default=value)
    config = parser.parse_args()
    main(**vars(config))
