import math

import numpy as np
from matplotlib import pyplot

class B:
    class Periodic:
        def array_length(self, nx):
            return nx
        def diff(self, u):
            return np.roll(np.diff(u, append=u[0]), 1)
    class Closed:
        def array_length(self, nx):
            return nx

boundary = B.Periodic()
def v(*a): return a[0]

space_size = v(2,)
num_x = v(100,)
num_t = v(24,)
dt = .01
dx = space_size / num_x
c = 1

def _step_i(u):
    u = u.copy()
    for n in range(num_t):  #loop for values of n from 0 to nt, so it will run nt times
        un = u.copy() ##copy the existing values of u into un
        # un = boundary.copy(u)
        for i in range(0, num_x): ## you can try commenting this line and...
            u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
    return u

def _step_v(u):
    for n in range(num_t):
        diff = boundary.diff(u)
        u = u - c * dt/dx * diff
    return u


def _plot(u):
    pyplot.plot(np.linspace(0, 2, num_x), u[:num_x])

def _run(n, step):
    _u = np.ones(num_x)
    # hat function
    _u[int(.5 / dx):int(1/dx + 1)] = 2
    for _ in range(n):
        # print((_stepv(_u) - _step1(_u) < 0.1))
        _plot(step(_u))
        _u = step(_u)
    return _u

a=_run(8, _step_v)
pyplot.show()
