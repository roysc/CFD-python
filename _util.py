import math

import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

def ArrayGen(shape):
    def _a(*a):
        if len(a) == 1:
            return np.full(shape, a[0])
        return np.array(a)
    return _a
mkarray = ArrayGen

def with_(a, i, e):
    r = list(a)
    r[i] = e
    return r

class SliceGen:
    def __getitem__(self, i):
        return i
s_ = SliceGen()

def diffprev(a, shift=1, axis=-1):
    """
    Calculate the n-th discrete difference along the given axis with a
    periodic boundary condition

    The first difference is given by ``out[i] = a[i] - a[i-1]`` along
    the given axis, higher differences are calculated by using `diff`
    recursively.
    """
    slices = [slice(None)] * len(np.shape(a))
    tail = a[tuple(with_(slices, axis, slice(0, shift)))]
    return np.roll(np.diff(a, shift, append=tail, axis=axis), shift, axis=axis)

def diffnext(a, shift=1, axis=-1):
    slices = [slice(None)] * len(np.shape(a))
    tail = a[tuple(with_(slices, axis, slice(0, shift)))]
    return np.diff(a, shift, append=tail, axis=axis)

def linspaces(rbounds, nr):
    return [np.linspace(a, b, n) for a, b, n in zip(*rbounds, nr)]

def compare(x, u0, funcs, n=1, error=0.01, show=False):
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

def plot_steps(x, steps):
    for u in steps:
        pyplot.plot(x, u)
    pyplot.show()
    return u

def plot(x, u, n, step):
    return plot_steps(x, steps(u, n, step))

def plot_2D(xy, p):
    fig = pyplot.figure(figsize=(8, 8), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(*xy)
    surf = ax.plot_surface(X, Y, p, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.view_init(30, 225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$');
