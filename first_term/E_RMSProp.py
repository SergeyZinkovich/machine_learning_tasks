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

class RMSProp:
    eta: float
    gamma: float
    epsilon: float

    def __init__(self, *, eta=0.1, gamma=0.9, epsilon=1e-8):
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon

    def optimize(self, oracle, x0, *,
                 max_iter=100, eps=1e-5):
        x = np.copy(x0)
        E = np.zeros(x0.shape)
        for _ in range(max_iter):
            grad = oracle.gradient(x)
            E = self.gamma * E + (1 - self.gamma) * np.square(grad)
            v = grad * self.eta / np.sqrt(E + self.epsilon)
            if np.linalg.norm(grad) < eps:
                break
            x -= v
        return x