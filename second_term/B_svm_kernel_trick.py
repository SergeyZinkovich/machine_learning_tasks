import numpy as np


def transform_squares(X: np.ndarray) -> np.ndarray:
    a = np.max([np.abs(X[:, 0]), np.abs(X[:, 1])], axis=0)
    X[:, 1] = a
    return X


def transform_circles(X: np.ndarray) -> np.ndarray:
    a = X[:, 0] ** 2 + X[:, 1] ** 2
    a[np.logical_and(20 < a, a < 60)] = -a[np.logical_and(20 < a, a < 60)]
    X[:, 1] = a
    return X


def transform_moons(X: np.ndarray) -> np.ndarray:
    a = 0.5 * np.sin(3 * X[:, 0] + 1.7) + 0.25 - X[:, 1]
    X[:, 1] = a
    return X


def transform_spirals(X: np.ndarray) -> np.ndarray:
    r = np.linspace(0.010, 0.120, 1000)
    t = 27 * np.pi * r
    y = r * 100 * np.sin(t) + 0.2
    x = r * 100 * np.cos(t) + 0.35
    points = np.array(list(zip(x, y)))
    a = []
    for i in X:
        dist = min(np.linalg.norm(points - i, ord=2, axis=1))
        a.append(dist)
    X[:, 1] = a
    return X