# 9 laplace eq.
import math
import numpy as np

from matplotlib import pyplot

from _lib import mkarray, plot_2D

# boundary conditions
def _set_bounds(p, y):
    p[:, 0] = 0  # p = 0 @ x = 0
    p[:, -1] = y  # p = y @ x = 2
    p[0, :] = p[1, :]  # dp/dy = 0 @ y = 0
    p[-1, :] = p[-2, :]  # dp/dy = 0 @ y = 1

def laplace_2D(p, dr, r, l1norm_target):
    l1norm = 1
    dx, dy = dr
    while l1norm > l1norm_target:
        pn = p.copy()
        p[1:-1, 1:-1] = ((dy**2 * (pn[1:-1, 2:] + pn[1:-1, 0:-2]) +
                         dx**2 * (pn[2:, 1:-1] + pn[0:-2, 1:-1])) /
                        (2 * (dx**2 + dy**2)))
        _set_bounds(p, r[1])
        l1norm = (np.sum(np.abs(p[:]) - np.abs(pn[:])) /
                  np.sum(np.abs(pn[:])))
    return p

# def analytical_soln(x, y):
#     limit = 32
#     sum = 0
#     for n in range(1, limit+1):
#         sum += sinh(n * math.pi * x) * cos(n * math.pi * y) /
#             (n * math.pi)**2 * sinh(2 * n * math.pi)
#     return x / 4 - 4 * sum

# constants
dim = 2
_a = mkarray((dim,))
c = 1
nr = _a(31)
dr = 2 / (nr - 1)
bounds = [_a(0), _a(2, 1)]

_r = [np.linspace(0, s, n) for s, n in zip(bounds[1], nr)]
_p = np.zeros(nr)
_set_bounds(_p, _r[1])
_p = laplace_2D(_p, dr, _r, 1e-4)

plot_2D(_r, _p)
pyplot.show()
