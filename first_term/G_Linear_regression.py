import numpy as np


def linear_func(theta, x):
    return x @ theta


def linear_func_all(theta, X):
    return linear_func(theta, X)


def mean_squared_error(theta, X, y):
    y_pred = linear_func_all(theta, X)
    return np.mean(np.square(y - y_pred))


def grad_mean_squared_error(theta, X, y):
    return -2 * ((y - linear_func_all(theta, X)) * X.T).mean(axis=1)