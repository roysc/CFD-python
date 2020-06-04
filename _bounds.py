from collections import namedtuple
from operator import add
from itertools import product, chain
from functools import reduce

import numpy as np
import _util

def _show_slice(s):
    if not isinstance(s, slice):
        return str(s)
    rep = lambda i: (i if i is not None else '')
    ret = f'{rep(s.start)}:{rep(s.stop)}'
    if s.step is not None:
        ret += f':{s.step}'
    return ret

def _show_slices(slices):
    ret = ', '.join(_show_slice(s) for s in slices)
    return ret

def shape_concat(*s):
    tups = ()
    for a in s:
        if isinstance(a, int):
            if a != 1: tups += (a,)
        else:
            tups += tuple(a)
    return tups

def wall_boundary(shape, u, height):
    D = len(shape)
    for d, i in product(range(D), (0, -1)):
        k = [slice(None)] * D
        k[d] = i
        # print('wall[{}] = {}'.format(_show_slices(k), height))
        u[tuple(k)] = height

class WallBoundary:
    def __init__(self, shape, value):
        self.shape = shape
        self.value = value
    def enforce(self, u):
        D = len(self.shape)
        for d, i in product(range(D), (0, -1)):
            k = [slice(None)] * D
            k[d] = i
            u[tuple(k)] = self.value

class SliceWindow:
    def __init__(self, a, k=1, bc=None):
        a_shape = np.shape(a)
        D_shape, k_a = a_shape[:-1], a_shape[-1]
        if k_a != k:
            if k == 1: D_shape = a_shape
            else: raise ValueError(k)
        self.k = k
        self.a = a
        self._shape = D_shape
        # if bc is None:
        #     wall = np.full((self.k,), 1)
        #     bc = WallBoundary(self._shape, wall)
        # self.bcond = bc

    @property
    def D(self):
        return len(self._shape)

    def __getitem__(self, i):
        if i == slice(None):
            return self.a[i]
        raise NotImplementedError

    def __setitem__(self, i, val):
        if i == slice(None):
            self.a[i] = val
            # self.bcond.enforce(self.a)
        else:
            raise NotImplementedError

    def diff_prev(self, i=1):
        return np.array([_util.diffprev(self.a, i, axis=ax) for ax in range(self.D)])

    def diff_next(self, i=1):
        return np.array([_util.diffnext(self.a, i, axis=ax) for ax in range(self.D)])

    def diff_central(self, i=1):
        return self.diff_next(i) - self.diff_prev(i)
