import numpy as np


def loo_score(predict, X, y, k):
    err = 0
    for i in range(0, len(X)):
        y_out = predict(np.concatenate([X[:i], X[i+1:]]), np.concatenate([y[:i], y[i+1:]]), X[i], k)
        err += int(y_out == y[i])
        #err += (y[i] - y_out) ** 2
    return err