import math
import numpy as np
import operator as op
from functools import reduce


def length(arr):
    return max(arr.shape)


def pol2cart(phi, rho):
    x = rho * math.cos(phi)
    y = rho * math.sin(phi)
    return(x, y)


def nchoosek(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom


def get_bounded_max(arr, up_thresh, low_thresh):
    (pnts_up,) = (arr < up_thresh).nonzero()
    (pnts_low,) = (arr > low_thresh).nonzero()
    pnts = np.intersect1d(pnts_up, pnts_low)
    return pnts


def get_max(arr, threshold, num_pnts):
    out = []
    if len(arr) == 0:
        return np.array(out)
    for i in range(num_pnts):
        [a, b] = arr.max(0), arr.argmax(0)
        if a<threshold:
            return np.array(out)
        else:
            out.append(b)
            arr[b] = 0
    return np.array(out)
