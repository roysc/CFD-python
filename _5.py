# 2D linear convection
import math
import numpy as np

from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots

import _lib
from _lib import plot, compare, wall_boundary

# def v(*a): return np.array(a)
v = _lib.mkarray((2,))

# constants
wall = (1)
rbox = [v(0), v(2)]
num_r = v(11)
num_t = 100
c = 1
dr = rbox[1] / (num_r - 1)
sigma = 0.2
dt = sigma * np.prod(dr)
dx, dy = dr

# iterative
def _step_it(u):
    u = u.copy()
    for n in range(num_t):
        un = u.copy()
        row, col = u.shape
        for j in range(1, row):
            for i in range(1, col):
                u[j, i] = (un[j, i] - (c * dt / dx * (un[j, i] - un[j, i - 1])) -
                                      (c * dt / dy * (un[j, i] - un[j - 1, i])))
                wall_boundary(num_r, u, wall)
    return u

# vectorized
def _step_vec(u):
    U = _lib.SliceWindow(u)
    for n in range(num_t):
        U[:] = U[:] - np.dot(U.diff_prev().T, c * dt / dr)
        wall_boundary(num_r, u, wall)
    return u

def _run():
    r = [np.linspace(0, size, num) for size, num in zip(rbox[1], num_r)]

    _u = np.ones(num_r)
    # set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    _u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
    _u = _step_vec(_u)

    fig = pyplot.figure(figsize=(8,8), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(*r)
    surf = ax.plot_surface(X, Y, _u, cmap=cm.viridis)
    pyplot.show()

if __name__ == '__main__': _run()

def test():pass
