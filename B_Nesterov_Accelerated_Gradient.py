import numpy as np
import math



class NesterovAG:
    eta: float
    alpha: float

    def __init__(self, *, alpha=0.9, eta=0.1):
        self.alpha = alpha
        self.eta = eta

    def optimize(self, oracle, x0, *,
                 max_iter=100, eps=1e-5):
        x = x0
        v = 0
        for _ in range(max_iter):
            grad = oracle.gradient(x+v)
            v = self.alpha * v - self.eta * grad
            if np.linalg.norm(grad) < eps:
                break
            x += v
        return x