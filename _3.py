import math
from collections import namedtuple

import numpy as np
from matplotlib import pyplot

def window(u):
    class _Window(namedtuple('_W', ('a', 'ix'))):
        def __getitem__(self, i):
            return self.a[(self.ix+i) % len(self.a)]
    for ix in range(0, len(u)):
        yield ix, _Window(u, ix)

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
dx = space_size / num_x
num_t = v(24,)
nu = 0.3
sigma = 0.2
dt = sigma * dx**2 / nu

# iterative
def _step_i(u):
    u = u.copy()
    for n in range(num_t):
        un = u.copy()
        # for i, w in window(un):
        #     u[i] = w[0] + nu * dt / dx**2 * (w[1] - 2*w[0] + w[-1])
        for i in range(0, num_x-1):
            u[i] = un[i] + nu * dt / dx**2 * (un[i+1] - 2*un[i] + un[i-1])
    return u

def rolldiff(a, shift=1):
    return np.roll(np.diff(a, append=a[0:shift]), shift)
    
# vectorized
def _step_v(u):
    for n in range(num_t):
        u = u + nu * dt/dx**2 * (rolldiff(np.roll(u, -1)) - rolldiff(u))
    return u

def _run(u, n, step):
    xspace = np.linspace(0, 2, num_x)
    for _ in range(n): 
        pyplot.plot(xspace, u[:num_x])
        u = step(u)
    return u

_u = np.ones(num_x)
# hat function
_u[int(.5 / dx):int(1/dx + 1)] = 2
_run(_u, 32, _step_v)

pyplot.show()
