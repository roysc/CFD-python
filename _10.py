# 10 poisson
import math
import numpy as np
from matplotlib import pyplot

from _lib import mkarray, plot_2D

def step(p, b, dr, nt=100):
    dx, dy = dr
    for _ in range(nt):
        pd = p.copy()
        p[1:-1,1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy**2 +
                        (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx**2 -
                        b[1:-1, 1:-1] * dx**2 * dy**2) /
                        (2 * (dx**2 + dy**2)))
        _set_bounds(p, *nr)
    return p

# boundary conditions
def _set_bounds(p, nx, ny):
    p[0, :] = 0
    p[ny-1, :] = 0
    p[:, 0] = 0
    p[:, nx-1] = 0

dim = 2
_a = mkarray((dim,))
nr = _a(50)
rmin = _a(0)
rmax = _a(2,1)
dr = (rmax - rmin)/(nr - 1)
bounds = [rmin, rmax]

nx, ny = nr

_p = np.zeros(nr)
_b = np.zeros(nr)
_r = [np.linspace(0, s, n) for s, n in zip(bounds[1], nr)]

# source
_b[int(ny/4), int(nx/4)] = 100
_b[int(3*ny/4), int(3*nx/4)] = -100

step(_p, _b, dr)
plot_2D(_r, _p)
pyplot.show()
