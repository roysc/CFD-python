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

class Periodic:
    def array_length(self, nx):
        return nx
    def diff(self, u):
        return np.roll(np.diff(u, append=u[0]), 1)

class Closed:
    def array_length(self, nx):
        return nx

class _Window(namedtuple('_W', ('a', 'ix'))):
    def __getitem__(self, i):
        return self.a[(self.ix+i) % len(self.a)]
def window(u):
    for ix in range(0, len(u)):
        yield ix, _Window(u, ix)

# Access neighbors as a composite vector
# U[-1] = u[1:-1, ]
# Use a buffer cell for boundary
class SliceWindow:
    data_slice = slice(1, None)      # slice referencing actual data

    def __init__(self, shape, a, k=None, data=slice(1, None)):
        # self.data_slice = data
        a_shape = np.shape(a)
        if k is not None:
            full_shape = shape_concat(shape, k)
            assert full_shape == a_shape
        else:
            if shape == a_shape: # 1-dim
                k = 1
            else:
                assert shape == a_shape[:-1]
                k = a_shape[-1]
        self.k = k              # num. components of data type
        self.D = len(shape)     # dimensions of the data
        self.a = a
        self.slice_shape = self.a[self._key()].shape

    @property
    def array(self):
        return self.a
    def _key(self):
        return (self.data_slice,) * self.D
    def _diff_slice(self, i):
        # return slice(i + (s.start or 0), i or s.stop, s.step)
        return slice(i+1, i or None)

    def diff(self, i):
        sfull = [self.data_slice,] * self.D
        keys = []
        for ax in range(self.D):
            slices = sfull[:]
            slices[ax] = self._diff_slice(i)            # shift along each axis
            keys.append(slices)

        # return a D x S matrix?
        ret = [
            self.a[tuple(sfull)] - self.a[tuple(keys[ax])]
            for ax in range(self.D)
        ]
        return np.array(ret)

    def __repr__(self):
        return f'SliceWindow([{_show_slice(self.data_slice)}] .a=({self.a.shape} {self.a.dtype})'

    def __getitem__(self, i):
        if i == slice(None):
            ret = self.a[self._key()]
            return ret
        raise NotImplementedError

    def __setitem__(self, i, val):
        if i == slice(None):
            self.a[(self.data_slice,) * self.D] = val
        else:
            raise NotImplementedError

def _test():
    u = np.random.randn(4,4)
    s = SliceWindow((4,4), u)
    assert(s.slice_shape == (3,3))
    assert(s[:].shape == s.slice_shape)

    uv = np.random.randn(4,4,2)
    s = SliceWindow((4,4), uv, k=2)
    assert(s.slice_shape == (3,3,2))
    assert(s[:].shape == s.slice_shape)

    s = SliceWindow((4,4), uv)
    assert(s[:].shape == (3,3,2))

    # multiply vec type by diff matrix
    # g = np.array([0,1])
    # print(s.diff(-1).shape)
    # print(np.dot(s.diff(-1), s[:]))
_test()

# u = np.random.randn(4,4)
# s = SliceWindow((4,4), u)

# class _WindowFactory:
#     def __init__(self, W):
#         self.Window = W
#         def __getitem__(self, s):
#             return lambda *a: self.Window(*a,*k, data=s)
# _w = _WindowFactory(SliceWindow)
# s = _w[1:](a) # => SliceWindow()
