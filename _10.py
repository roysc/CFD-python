# 10 poisson
import math
import numpy as np
from matplotlib import pyplot

import _lib
from _lib import mkarray, plot_2D

def set_bounds_old(p):
    p[0, :] = 0
    p[-1, :] = 0
    p[:, 0] = 0
    p[:, -1] = 0

def poisson_old(p, b, dr, nt=100):
    dx, dy = dr
    for _ in range(nt):
        pd = p.copy()
        p[1:-1,1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy**2 +
                        (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx**2 -
                        b[1:-1, 1:-1] * dx**2 * dy**2) /
                        (2 * (dx**2 + dy**2)))
        set_bounds_old(p)
    return p

# boundary conditions
def set_bounds(p):
    p[0, :] = 0
    p[-1, :] = 0
    p[:, 0] = 0
    p[:, -1] = 0

def poisson(p, b, dr, nt=100):
    P = _lib.Window(p, mask=_lib.s_[1:-1, 1:-1])
    for _ in range(nt):
        N = np.transpose(P.neighbors(-1) + P.neighbors(1), (1,2,0))
        P[:] = (np.dot(N, np.flip(dr**2)) - b[1:-1, 1:-1] * np.prod(dr**2)) / (2 * np.dot(dr, dr))
        set_bounds(p)
    return p

def run():
    dim = 2
    _a = mkarray((dim,))

    nr = _a(50)
    rmin = _a(0)
    rmax = _a(1,2)
    dr = (rmax - rmin)/(nr - 1)
    bounds = [rmin, rmax]

    _p = np.zeros(nr)
    _b = np.zeros(nr)
    _r = [np.linspace(0, s, n) for s, n in zip(bounds[1], nr)]

    # source
    nx, ny = nr
    _b[int(nx/4), int(ny/4)] = 100
    _b[int(3*nx/4), int(3*ny/4)] = -100

    poisson(_p, _b, dr)
    plot_2D(_r, _p)
    pyplot.show()
    
if __name__ == '__main__':
    run()
