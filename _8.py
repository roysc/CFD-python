# 8 burgers (2d)
import math
import numpy as np

from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

from _lib import Periodic, plot

dim = 2
boundary = Periodic()
def _a(*a):
    if len(a) == 1:
        return np.full((dim,), a[0])
    return np.array(a)

# constants
sigma = 0.0009
nu = .01
space_size = _a(2,2)
num_P = _a(41)
num_t = 120
dP = space_size / (num_P - 1)
dt = sigma * np.prod(dP) / nu

dx, dy = dP

def step(u, v, nt):
    for n in range(nt + 1): ##loop across number of time steps
        un = u.copy()
        vn = v.copy()

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         dt / dx * un[1:-1, 1:-1] *
                         (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                         dt / dy * vn[1:-1, 1:-1] *
                         (un[1:-1, 1:-1] - un[0:-2, 1:-1]) +
                         nu * dt / dx**2 *
                         (un[1:-1,2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                         nu * dt / dy**2 *
                         (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         dt / dx * un[1:-1, 1:-1] *
                         (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                         dt / dy * vn[1:-1, 1:-1] *
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) +
                         nu * dt / dx**2 *
                         (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                         nu * dt / dy**2 *
                         (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))

        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1
    return u, v

# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
def set_hat(u, dr, box=np.array([[.5, 1], [.5, 1]])):
    (xa, ya), (xb, yb) = (box[:, 0]/dr), (box[:, 1]/dr + 1)
    u[int(xa):int(xb), int(ya):int(yb)] = 2

_u = np.ones(num_P)
_v = np.ones(num_P)
set_hat(_u, dP)
set_hat(_v, dP)

_u, _v = step(_u, _v, num_t)

P = [np.linspace(0, size, num) for size, num in zip(space_size, num_P)]
fig = pyplot.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(*P)
ax.plot_surface(X, Y, _u, cmap=cm.viridis, rstride=2, cstride=2)
ax.plot_surface(X, Y, _v, cmap=cm.viridis, rstride=2, cstride=2)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$');
pyplot.show()
