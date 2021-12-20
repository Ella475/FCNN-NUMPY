import numpy as np
from abc import ABC, abstractmethod
import time
from optimizers import Optimizer, SgdOptimizer

from utils import shuffle


class Module(ABC):
    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args):
        raise NotImplementedError


class Relu(Module):
    def __init__(self):
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'] = x
        return x * (x > 0)

    def backward(self, dout) -> np.ndarray:
        return dout * (self.cache['x'] > 0)


class Linear(Module):
    def __init__(self, input_dim: int, output_dim: int):
        self.w = {'value': np.sqrt(2. / input_dim) * np.random.randn(input_dim, output_dim)}
        self.b = {'value': np.zeros(output_dim, dtype=float)}
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'] = x
        return x @ self.w['value'] + self.b['value']

    def backward(self, dout) -> np.ndarray:
        dx = dout @ self.w['value'].T
        self.w['grad'] = self.cache['x'].T @ dout
        self.b['grad'] = dout.sum(axis=0)
        return dx

    def get_params(self):
        return {'w': self.w, 'b': self.b}

    def set_params(self, params):
        self.w = params['w']
        self.b = params['b']


class LinearRelu(Module):
    def __init__(self, input_dim: int, output_dim: int):
        self.linear = Linear(input_dim, output_dim)
        self.relu = Relu()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.relu.forward(self.linear.forward(x))

    def backward(self, dout) -> np.ndarray:
        return self.linear.backward(self.relu.backward(dout))

    def get_params(self):
        return self.linear.get_params()

    def set_params(self, params):
        self.linear.set_params(params)


class NN(Module):
    @abstractmethod
    def train(self, *argv):
        raise NotImplementedError

    @abstractmethod
    def predict(self, *argv):
        raise NotImplementedError


class FCN(NN):
    def __init__(self, hidden_sizes_list: list, input_dim: int = 28 * 28, num_classes: int = 10,
                 optimizer: Optimizer = SgdOptimizer, batch_size: int = 100, num_epochs: int = 20,
                 max_running_time: float = 20):

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_layers = 1 + len(hidden_sizes_list)
        self.layers = []
        self.best_val_acc = 0
        self.max_running_time = max_running_time * 60
        self.best_model_param = []

        for i in range(self.num_layers - 1):
            output_dim = hidden_sizes_list[i]
            layer = LinearRelu(input_dim, output_dim)
            self.layers.append(layer)
            input_dim = output_dim

        layer = Linear(input_dim, num_classes)
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, dout):
        self.layers.reverse()
        for layer in self.layers:
            dout = layer.backward(dout)
        self.layers.reverse()

    def update(self):
        for layer in self.layers:
            layer.set_params(self.optimizer.update(layer.get_params()))

    def loss(self, scores, y):
        loss, dloss = softmax_loss(scores, y)
        for layer in self.layers:
            params = layer.get_params()

        return loss, dloss

    def predict(self, test_x):
        scores = self.forward(test_x)
        return np.argmax(scores, axis=1)

    def check_accuracy(self, x, labels):
        preds = self.predict(x)
        return 100 * float(sum(preds == labels)) / float(len(labels))

    def update_lr(self):
        config = self.optimizer.get_config()
        config['lr'] *= config['lr_decay']
        self.optimizer.set_config(config)

    def train(self, train_x, train_y, validation_x, validation_y):
        start_training_time = time.perf_counter()

        num_train = train_x.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)

        for i in range(self.num_epochs):
            start_epoch_time = time.perf_counter()
            train_x, train_y = shuffle(train_x, train_y)
            for j in range(iterations_per_epoch):
                batch = train_x[j:(j + self.batch_size)]
                batch_labels = train_y[j:(j + self.batch_size)]
                loss, dloss = self.loss(self.forward(batch), batch_labels)
                self.backward(dloss)
                self.update()

            self.update_lr()
            cur_val_accuracy = self.check_accuracy(validation_x, validation_y)

            if cur_val_accuracy > self.best_val_acc:
                self.best_val_acc = cur_val_accuracy
                self.best_model_param = [layer.get_params() for layer in self.layers]

            end_epoch_time = time.perf_counter()

            if end_epoch_time - start_training_time > self.max_running_time:
                print('Training time is up!')
                break

            print('(Epoch %d / %d; time: %.2f) train_acc: %.3f; val_acc: %.3f; loss: %f;' % (
                i, self.num_epochs, end_epoch_time - start_epoch_time, self.check_accuracy(train_x, train_y),
                cur_val_accuracy, loss))

        # At the end of training swap the best params into the model
        for layer, params in zip(self.layers, self.best_model_param):
            layer.set_params(params)


def softmax_loss(x, y):
    x = (x.T - x.max(axis=1).T).T
    log_probs = (x.T - np.log(np.exp(x).sum(axis=1)).T).T
    probs = np.exp(log_probs)
    loss = (-1.0 / x.shape[0]) * log_probs[np.arange(x.shape[0]), y].sum()
    dx = probs.copy()
    dx[np.arange(x.shape[0]), y] -= 1
    dx /= x.shape[0]
    return loss, dx
