import numpy as np
import math


# class Oracle:
#     '''Provides an interface for evaluating a function and its derivative at arbitrary point'''
# 
#     def value(self, x: np.ndarray) -> float:
#         raise NotImplementedError()
# 
#     def gradient(self, x: np.ndarray) -> np.ndarray:
#         return np.array(np.cos(x))

class AdaGrad:
    eta: float
    epsilon: float

    def __init__(self, *, eta=0.1, epsilon=1e-8):
        self.eta = eta
        self.epsilon = epsilon

    def optimize(self, oracle, x0, *,
                 max_iter=100, eps=1e-5):
        x = x0
        G = np.zeros(x0.shape)
        for _ in range(max_iter):
            grad = oracle.gradient(x)
            G += np.square(grad)
            v = (self.eta / np.sqrt(G + self.epsilon)) * grad
            if np.linalg.norm(grad) < eps:
                break
            x -= v
        return x