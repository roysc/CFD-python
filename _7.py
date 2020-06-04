# 7 diffusion
import math
import numpy as np

from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

import _lib

dim = 2
_a = _lib.mkarray((dim,))

# constants
wall = 1
sigma = 0.25
nu = .05
space_size = _a(2,2)
shape_r = _a(33)
num_t = 15
dr = space_size / (shape_r - 1)
dt = sigma * np.prod(dr) / nu

def diffuse(u, nt):
    U = _lib.SliceWindow(u)
    for n in range(nt + 1):
        U[:] = U[:] + np.dot(U.diff_central().T, nu * dt / dr**2)
        _lib.wall_boundary(shape_r, u, wall)

    P = [np.linspace(0, size, num) for size, num in zip(space_size, shape_r)]
    X, Y = np.meshgrid(*P)

    fig = pyplot.figure(figsize=(8,8), dpi=100)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, u[:], rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=True)
    ax.set_zlim(1, 2.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$');
    return u

# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
def set_hat(u, dr, box=np.array([[.5, 1], [.5, 1]])):
    (xa, ya), (xb, yb) = (box[:, 0]/dr), (box[:, 1]/dr + 1)
    u[int(xa):int(xb), int(ya):int(yb)] = 2

_u = np.ones(shape_r)
set_hat(_u, dr)
diffuse(_u, num_t)

pyplot.show()
