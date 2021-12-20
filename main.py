import sys
import numpy as np

from utils import train_test_split, pre_processing
from optimizers import SgdOptimizer, SgdMomentumOptimizer, AdamOptimizer
from modules import FCN


def main():
    max_row = 55000
    train_x = pre_processing(np.loadtxt(sys.argv[1], max_rows=max_row))
    train_y = np.loadtxt(sys.argv[2], dtype=int, max_rows=max_row)
    test_x = pre_processing(np.loadtxt(sys.argv[3]))
    train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y)

    # configurations for all the optimizers: sgd, sgd_momentum, adam
    optimizer_config = {'lr': 5e-3, 'lr_decay': 0.99, 'momentum': 0.7, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8}

    fcn = FCN(hidden_sizes_list=[256, 256, 256], optimizer=AdamOptimizer(optimizer_config),
              batch_size=4096, num_epochs=350, max_running_time=20)
    fcn.train(train_x, train_y, validation_x, validation_y)

    test_y = fcn.predict(test_x)

    # write results

    f = open('test_y', "w")
    for pred in test_y:
        f.write(f"{pred}\n")
    f.close()


if __name__ == '__main__':
    main()
