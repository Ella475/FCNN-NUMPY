from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def update(self, params: dict) -> dict:
        raise NotImplementedError

    def get_config(self):
        return self.config

    def set_config(self, config):
        self.config = config


class SgdOptimizer(Optimizer):
    def update(self, params):
        lr = self.config['lr']

        for p in params.keys():
            grad = params[p]['grad']

            params[p]['value'] -= lr * grad

        return params


class SgdMomentumOptimizer(Optimizer):
    def update(self, params):
        momentum = self.config['momentum']
        lr = self.config['lr']

        for p in params.keys():
            if 'optimizer_config' not in params[p]:
                params[p]['optimizer_config'] = {}
            if 'v' not in params[p]['optimizer_config']:
                params[p]['optimizer_config']['v'] = np.zeros_like(params[p]['value'])

            grad = params[p]['grad']
            v = params[p]['optimizer_config']['v']

            v = momentum * v - lr * grad  # integrate velocity
            params[p]['value'] += v

            params[p]['optimizer_config']['v'] = v

        return params


class AdamOptimizer(Optimizer):
    def update(self, params):
        lr = self.config['lr']
        beta1 = self.config['beta1']
        beta2 = self.config['beta1']
        eps = self.config['eps']

        for p in params.keys():
            if not 'optimizer_config' in params[p]:
                params[p]['optimizer_config'] = {}
            if not 'v' in params[p]['optimizer_config']:
                params[p]['optimizer_config']['v'] = np.zeros_like(params[p]['value'])
            if not 'm' in params[p]['optimizer_config']:
                params[p]['optimizer_config']['m'] = np.zeros_like(params[p]['value'])
            if not 't' in params[p]['optimizer_config']:
                params[p]['optimizer_config']['t'] = 0

            grad = params[p]['grad']
            t = params[p]['optimizer_config']['t']
            m = params[p]['optimizer_config']['m']
            v = params[p]['optimizer_config']['v']

            t += 1
            m = beta1 * m + (1 - beta1) * grad
            mt = m / (1 - beta1 ** t)
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            vt = v / (1 - beta2 ** t)
            params[p]['value'] -= lr * mt / (np.sqrt(vt) + eps)

            params[p]['optimizer_config']['t'] = t
            params[p]['optimizer_config']['m'] = m
            params[p]['optimizer_config']['v'] = v

        return params
