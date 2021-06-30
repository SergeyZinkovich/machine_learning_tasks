import numpy as np


def sus(fitness: np.ndarray, n: int, start: float) -> list:
    """Selects exactly `n` indices of `fitness` using Stochastic universal sampling alpgorithm. 

    Args:
        fitness: one-dimensional array, fitness values of the population, sorted in descending order
        n: number of individuals to keep
        start: minimal cumulative fitness value

    Return:
        Indices of the new population"""

    f = np.sum(fitness)
    p = f / n
    pointers = [start + i * p for i in range(n)]

    return RWS(fitness, pointers)


def RWS(fitness, points):
    keep = []
    for P in points:
        i = 0
        s = fitness[i]
        while s < P:
            i += 1
            s += fitness[i]
        keep.append(i)
    return keep