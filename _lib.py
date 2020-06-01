import math
from collections import namedtuple

import numpy as np
from matplotlib import pyplot

class _Window(namedtuple('_W', ('a', 'ix'))):
    def __getitem__(self, i):
        return self.a[(self.ix+i) % len(self.a)]
def window(u):
    for ix in range(0, len(u)):
        yield ix, _Window(u, ix)

# boundary conditions
class Periodic:
    def array_length(self, nx):
        return nx
    def diff(self, u):
        return np.roll(np.diff(u, append=u[0]), 1)
class Closed:
    def array_length(self, nx):
        return nx

def plot(xspace, u, n, step):
    for _ in range(n):
        pyplot.plot(xspace, u)
        u = step(u)
    pyplot.show()
    return u

def compare(x, u0, funcs, n=1, error=0.01, show=True):
    err = 0.0
    for fa, fb in zip(funcs[:-1], funcs[1:]):
        ua = fa(u0.copy())
        ub = fb(u0.copy())
        # eq &= np.all(np.abs(ua - ub) < error)
        err += np.sum(np.abs(ua - ub))
        if show:
            pyplot.plot(x, ua)
            pyplot.plot(x, ub)
            pyplot.show()
    return err
