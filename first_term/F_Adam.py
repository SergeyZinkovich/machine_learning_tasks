import numpy as np

# class Oracle:
#     '''Provides an interface for evaluating a function and its derivative at arbitrary point'''
#
#     def value(self, x: np.ndarray) -> float:
#         raise NotImplementedError()
#
#     def gradient(self, x: np.ndarray) -> np.ndarray:
#         return np.array(np.cos(x))

class Adam:
    eta: float
    beta1: float
    beta2: float
    epsilon: float

    def __init__(self, *, eta: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.beta2 = beta2
        self.beta1 = beta1
        self.eta = eta
        self.epsilon = epsilon

    def optimize(self, oracle, x0, *,
                 max_iter=100, eps=1e-5):
        x = np.copy(x0)
        m = np.zeros(x0.shape)
        v = np.zeros(x0.shape)
        for t in range(max_iter):
            grad = oracle.gradient(x)
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * np.square(grad)
            m_1 = m / (1 - np.power(self.beta1, t+1))
            v_1 = v / (1 - np.power(self.beta2, t+1))
            if np.linalg.norm(grad) < eps:
                break
            x -= self.eta * m_1 / (np.sqrt(v_1) + self.epsilon)
        return x