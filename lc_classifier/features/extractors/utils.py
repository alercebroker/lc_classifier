import numba
import numpy as np


@numba.jit(nopython=True)
def is_sorted(a):
    for i in range(a.size-1):
        if a[i+1] < a[i]:
            return False
    return True


@numba.jit(nopython=True)
def first_occurrence(a: np.ndarray, value):
    for i, x in enumerate(a):
        if x == value:
            return i
    return -1
