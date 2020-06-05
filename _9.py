# 9 laplace eq.
import math
import numpy as np
from itertools import product

from matplotlib import pyplot

import _util
from _lib import SliceWindow, s_

# boundary conditions
def set_bounds_old(p, y):
    p[:, 0] = 0  # p = 0 @ x = 0
    p[:, -1] = y  # p = y @ x = 2
    p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
    p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1

# Calculate equilibrium state
def calc_laplace_old(p, dr, r):
    dx, dy = dr
    p[1:-1, 1:-1] = (dy**2 * (p[1:-1, 2:] + p[1:-1, 0:-2]) +
            dx**2 * (p[2:, 1:-1] + p[0:-2, 1:-1])) / \
            (2 * (dx**2 + dy**2))

def set_bounds(p, y):
    p[0, :] = 0  # p = 0 @ x = 0
    p[-1, :] = y  # p = y @ x = 2
    p[:, 0] = p[:, 1]  # dp/dy = 0 @ y = 0
    p[:, -1] = p[:, -2]  # dp/dy = 0 @ y = 1

def calc_laplace(p, dr, r):
    P = SliceWindow(p, mask=s_[1:-1, 1:-1])
    N = np.flip((P.neighbors(1) + P.neighbors(-1)), axis=0)
    N = np.transpose(N, (1,2,0))
    P[:] = np.dot(N, dr**2) / (2 * np.dot(dr, dr))

def laplace_2D(p, dr, r, l1norm_target, params):
    calc, set_bounds = params
    l1norm = 1
    while l1norm > l1norm_target:
        pn = p.copy()
        calc(p, dr, r)
        set_bounds(p, r[1])
        l1norm = np.sum(np.abs(p) - np.abs(pn)) / np.sum(np.abs(pn))
    return p

def analytical_soln(x, y):
    sinh, cos = np.sinh, np.cos
    limit = 32
    sum = 0
    for n in range(1, limit+1):
        sum += sinh(n * math.pi * x) * cos(n * math.pi * y) / \
            (n * math.pi)**2 * sinh(2 * n * math.pi)
    return x / 4 - 4 * sum

def make_solution(nr, r):
    soln = np.zeros(nr)
    X, Y = r
    for i, j in product(*(range(nk) for nk in nr)):
        soln[i,j] = analytical_soln(X[i], Y[j])
    return soln

# constants
dim = 2
_a = _util.ArrayGen((dim,))
c = 1
nr = _a(16)
dr = 2 / (nr - 1)
rbounds = [_a(0), _a(2, 1)]

def _run():
    _r = _util.linspaces(rbounds, nr)
    _p = np.zeros(nr)

    # params = (calc_laplace_old, set_bounds_old)
    params = (calc_laplace, set_bounds)
    _r = list(reversed(_r))
    _p = laplace_2D(_p, dr, _r, 1e-4, params)

    # soln = make_solution(nr, _r)
    _util.plot_2D(_r, _p)
    pyplot.show()

if __name__ == '__main__':
    _run()
