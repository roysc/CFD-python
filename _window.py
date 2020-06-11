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

class SliceWindow:
    def __init__(self, a, k=1, mask=slice(None)):
        a_shape = np.shape(a)
        data_shape, k_a = a_shape[:-1], a_shape[-1]
        if k_a != k:
            if k != 1:
                raise ValueError('unexpected shape', k)
            data_shape = a_shape
        if k == 1: k = None
        self.k = k
        self.a = a
        self._mask = mask
        self._shape = self.a[mask].shape
        self.data_shape = data_shape

        if isinstance(mask, slice):
            self._mask = (mask,) + (slice(None),)*(self.D-1)

        class _NbrIx(namedtuple('_', 'this')):
            def __getitem__(self, *args):
                return self.this.neighbors(*args)
        self.n = _NbrIx(self)

    @property
    def D(self):
        return len(self._shape) - bool(self.k or 0)

    @property
    def array(self): return self.a

    @property
    def bounds(self):
        def or_(v, e):
            if v is not None: return v
            return e
        return tuple((s.start or 0, or_(s.stop, 0)) for s in self._mask)

    def __getitem__(self, i):
        if i != slice(None):
            raise NotImplementedError
        return self.a[self._mask]

    def __setitem__(self, i, val):
        if i == slice(None):
            self.a[self._mask] = val
        else:
            raise NotImplementedError

    # get slices, offset on axis
    def _nbr(self, shift, axis):
        a, b = self.bounds[axis]
        mb = (b % -self.a.shape[axis])
        under, over = 0, 0
        start, end = a + shift, mb + shift

        if start < 0:
            under = start
            start = 0
            end -= under
        if end > 0:
            over = end
            start -= over
            end = 0
        if end == 0: end = None

        if under or over:
            if under and over: raise TypeError('backing array is too small')
            amt = under + over
            full = np.roll(self.a, -amt, axis=axis)
        else:
            full = self.a

        slices = tuple(_util.with_(self._mask, axis, slice(start, end)))
        return full[slices]

    def neighbors(self, shift):
        ret = np.array([self._nbr(shift, ax) for ax in range(self.D)])
        if len(ret) == 1:
            ret = ret[0]
        return ret

    # equiv to util.diffprev(a, i, ax) for ax...
    def diff_prev(self, i=1): return self[:] - self.n[-1]
    def diff_next(self, i=1): return self.n[1] - self[:]

    def diff_central(self, i=1):
        return self.diff_next(i) - self.diff_prev(i)

    def __repr__(self):
        return f'SliceWindow([{self.a.shape} {self.a.dtype}], k={self.k}, mask=[{_show_slices(self._mask)}])'

Window = SliceWindow

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

# Conditions on difference
# e.g. du/dx = 0
class NeumannBoundary:
    pass
