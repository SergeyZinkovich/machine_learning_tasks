import numpy as np


def gini(y):
    unique, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p**2)


def entropy(y):
    unique, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log(p))


def split(X, y, feature_index, threshold):
    t = X[:, feature_index] <= threshold
    return y[t], y[np.invert(t)]


def tree_split(X, y, criterion):
    best_index = -1
    best_threshold_id = -1
    best_threshold = float('inf')
    best_entropy = float('inf')
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            feature_index = i
            threshold = X[j, i]
            left, right = split(X, y, feature_index, threshold)
            if criterion == 'var':
                entr = np.var(left) * len(left) / len(y) + np.var(right) * len(right) / len(y)
            elif criterion == 'entropy':
                entr = entropy(left) * len(left) / len(y) + entropy(right) * len(right) / len(y)
            else:
                entr = gini(left) * len(left) / len(y) + gini(right) * len(right) / len(y)
            if entr < best_entropy:
                best_entropy = entr
                best_threshold = threshold
                best_index = feature_index
                best_threshold_id = j

    return best_index, best_threshold_id