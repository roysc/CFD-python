import math
from collections import namedtuple

import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

class _Window(namedtuple('_W', ('a', 'ix'))):
    def __getitem__(self, i):
        return self.a[(self.ix+i) % len(self.a)]
def window(u):
    for ix in range(0, len(u)):
        yield ix, _Window(u, ix)

class Periodic:
    def array_length(self, nx):
        return nx
    def diff(self, u):
        return np.roll(np.diff(u, append=u[0]), 1)
class Closed:
    def array_length(self, nx):
        return nx

def mkarray(shape):
    def _a(*a):
        if len(a) == 1:
            return np.full(shape, a[0])
        return np.array(a)
    return _a

def diffprev(a, shift=1):
    return np.roll(np.diff(a, append=a[0:shift]), shift)

def linspaces(bounds, nr):
    return [np.linspace(a, b, n) for a, b, n in zip(*bounds, nr)]

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

def steps(u, n, step):
    for _ in range(n):
        u = step(u)
        yield u
        # yield u

def plot_steps(x, steps):
    for u in steps:
        pyplot.plot(x, u)
    pyplot.show()
    return u

def plot(x, u, n, step):
    return plot_steps(x, steps(u, n, step))

def plot_2D(xy, p):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(*xy)
    surf = ax.plot_surface(X, Y, p, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.view_init(30, 225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$');
