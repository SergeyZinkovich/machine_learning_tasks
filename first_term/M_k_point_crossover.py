import numpy as np
from typing import Tuple


def single_point_crossover(a: np.ndarray, b: np.ndarray, point: int) -> Tuple[np.ndarray, np.ndarray]:
    """Performs single point crossover of `a` and `b` using `point` as crossover point.
    Chromosomes to the right of the `point` are swapped

    Args:
        a: one-dimensional array, first parent
        b: one-dimensional array, second parent
        point: crossover point

    Return:
        Two np.ndarray objects -- the offspring"""

    new_a = np.concatenate((a[:point+1], b[point+1:]))
    new_b = np.concatenate((b[:point+1], a[point+1:]))
    return new_a, new_b


def two_point_crossover(a: np.ndarray, b: np.ndarray, first: int, second: int) -> Tuple[np.ndarray, np.ndarray]:
    """Performs two point crossover of `a` and `b` using `first` and `second` as crossover points.
    Chromosomes between `first` and `second` are swapped

    Args:
        a: one-dimensional array, first parent
        b: one-dimensional array, second parent
        first: first crossover point
        second: second crossover point

    Return:
        Two np.ndarray objects -- the offspring"""

    new_a = np.concatenate((a[:first+1], b[first+1:second], a[second:]))
    new_b = np.concatenate((b[:first+1], a[first+1:second], b[second:]))
    return new_a, new_b


def k_point_crossover(a: np.ndarray, b: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Performs k point crossover of `a` and `b` using `points` as crossover points.
    Chromosomes between each even pair of points are swapped

    Args:
        a: one-dimensional array, first parent
        b: one-dimensional array, second parent
        points: one-dimensional array, crossover points

    Return:
        Two np.ndarray objects -- the offspring"""

    point = 0

    new_a = np.array([])
    new_b = np.array([])

    for i, p in enumerate(points):
        if i % 2 == 0:
            new_a = np.concatenate((new_a, a[point:p+1]))
            new_b = np.concatenate((new_b, b[point:p+1]))
        else:
            new_a = np.concatenate((new_a, b[point+1:p]))
            new_b = np.concatenate((new_b, a[point+1:p]))

        point = p

        if i == len(points)-1:
            if (i + 1) % 2 == 0:
                new_a = np.concatenate((new_a, a[p:]))
                new_b = np.concatenate((new_b, b[p:]))
            else:
                new_a = np.concatenate((new_a, b[p + 1:]))
                new_b = np.concatenate((new_b, a[p + 1:]))

    return new_a, new_b