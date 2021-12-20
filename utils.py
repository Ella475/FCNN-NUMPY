import numpy as np


def train_test_split(X, Y, test_size=0.2):
    num_samples = Y.shape[0]
    random_indices = np.random.choice(num_samples, size=int(num_samples * test_size), replace=False)
    other_indices = np.array([i for i in range(num_samples) if i not in random_indices])

    return X[other_indices], X[random_indices], Y[other_indices], Y[random_indices]


def shuffle(train_x, train_y):
    p = np.random.permutation(len(train_y))
    return train_x[p], train_y[p]


def pre_processing(x):
    x[x > 0] = 1  # if the pixel is not black, it should be white
    x[x <= 0] = -1
    return x