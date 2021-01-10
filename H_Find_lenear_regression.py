import numpy as np


def fit_linear_regression(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y