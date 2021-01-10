import numpy as np


def knn_predict_simple(X, y, x, k):  # array of pairs -- class and number of votes of neighbors
    dist = np.sqrt(np.sum((X - x) ** 2, axis=len(X.shape) - 1))
    ids = np.argsort(dist)

    ans = {}
    for i in range(k):
        a = y[ids[i]]
        if ans.get(a):
            ans[a] += 1
        else:
            ans[a] = 1
    return list(ans.items())
