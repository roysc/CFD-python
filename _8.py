# 8 burgers (2d)
import math
import numpy as np

from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

import _lib

dim = 2
_a = _lib.mkarray((dim,))

# constants
wall = _a(1)
sigma = 0.0009
nu = .01
r_space = _a(2)
r_shape = _a(41)
num_t = 120
dr = r_space / (r_shape - 1)
dt = sigma * np.prod(dr) / nu

dx, dy = dr

def step(uv, nt):
    UV = _lib.SliceWindow(uv, dim)
    for n in range(nt + 1):
        UV[:] = UV[:] - \
            np.dot(UV.diff_prev().T * UV[:], dt / dr).T + \
            np.dot(UV.diff_central().T, nu * dt / dr**2).T

        _lib.wall_boundary(r_shape, uv, wall)
    return uv

# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
def set_hat(u, dr, box=np.array([[.5, 1], [.5, 1]])):
    (xa, ya), (xb, yb) = (box[:, 0]/dr), (box[:, 1]/dr + 1)
    u[int(xa):int(xb), int(ya):int(yb)] = _a(2)

def _run():
    _uv = np.ones(_lib.shape_concat(r_shape, dim))
    set_hat(_uv, dr)
    _uv = step(_uv, num_t)

    P = [np.linspace(0, size, num) for size, num in zip(r_space, r_shape)]
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(*P)
    ax.plot_surface(X, Y, _uv[:,:,0], cmap=cm.viridis, rstride=2, cstride=2)
    # ax.plot_surface(X, Y, _uv[:,:,1], cmap=cm.viridis, rstride=2, cstride=2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$');
    pyplot.show()

if __name__ == '__main__': _run()
