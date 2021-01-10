import numpy as np

def entropy(y):
        unique, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log(p))