import math
import numpy as np

from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots

from _lib import Periodic, plot

tau = 2*math.pi

dim = 2
boundary = Periodic()
def _a(*a):
    if len(a) == 1:
        return np.full((dim,), a[0])
    return np.array(a)

# constants
space_size = _a(2,2)
num_P = _a(80)
num_t = 80
c = 1
dP = space_size / (num_P - 1)
sigma = 0.2
dt = sigma * min(*dP)

dx, dy = dP

# vectorized
def _step_v(u, v):
    # u, v = U.copy()
    for n in range(num_t):
        un = u.copy()
        vn = v.copy()
        u[1:, 1:] = (un[1:, 1:] - 
                     (un[1:, 1:] * c * dt / dx * (un[1:, 1:] - un[:-1, 1:])) -
                      vn[1:, 1:] * c * dt / dy * (un[1:, 1:] - un[1:, :-1]))
        v[1:, 1:] = (vn[1:, 1:] -
                     (un[1:, 1:] * c * dt / dx * (vn[1:, 1:] - vn[:-1, 1:])) -
                     vn[1:, 1:] * c * dt / dy * (vn[1:, 1:] - vn[1:, :-1]))

        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

    return u, v

P = [np.linspace(0, size, num) for size, num in zip(space_size,num_P)]
_u = np.ones(num_P)
_v = np.ones(num_P)

# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
def set_hat(u, dr):
    (xa, ya), (xb, yb) = (.5/dr), (1/dr + 1)
    u[int(xa):int(xb), int(ya):int(yb)] = 2
set_hat(_u, dP)
set_hat(_v, dP)

_u, _v = _step_v(_u, _v)

fig = pyplot.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(*P)

surf = ax.plot_surface(X, Y, _u, cmap=cm.viridis, rstride=2, cstride=2)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$');

pyplot.show()

