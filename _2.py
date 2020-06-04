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

# class C:
space_size = v(2,)
num_x = v(100,)
num_t = v(24,)
dt = .01
dx = space_size / num_x


def _step_i(u):
    u = u.copy()
    for n in range(num_t):
        un = u.copy()
        for i in range(0, num_x):
            v = u[i]
            u[i] = un[i] - v * dt / dx * (un[i] - un[i-1])
    return u

def _step_v(u):
    for n in range(num_t):
        diff = boundary.diff(u)
        u = u - u * dt/dx * diff
    return u

def _run(x, u, n, step):
    for _ in range(n):
        pyplot.plot(x, u[:num_x])
        u = step(u)
    pyplot.show()
    return u

def test():
    # assert compare(x, _u, (_step_i, _step_v))
    pass

if __name__ == '__main__':
    x = np.linspace(0, 2, num_x)
    u = np.ones(num_x)
    u[int(.5 / dx):int(1/dx + 1)] = 2 # hat function
    _run(x, u, 32, _step_v)
