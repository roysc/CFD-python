# 7 diffusion
import math
import numpy as np

from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

from _lib import plot

dim = 2
def _a(*a):
    if len(a) == 1:
        return np.full((dim,), a[0])
    return np.array(a)

# constants
sigma = 0.25
nu = .05
space_size = _a(2,2)
num_P = _a(33)
num_t = 15
dP = space_size / (num_P - 1)
dt = sigma * np.prod(dP) / nu

dx, dy = dP

def diffuse(u, nt):
    for n in range(nt + 1):
        un = u.copy()
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1] +
            nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
            nu * dt / dy**2 * (un[2:,1: -1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
        )
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

    P = [np.linspace(0, size, num) for size, num in zip(space_size, num_P)]
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

_u = np.ones(num_P)
set_hat(_u, dP)
diffuse(_u, num_t)

pyplot.show()
