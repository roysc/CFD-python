# 2D linear convection
import math
import numpy as np

from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots

from _lib import Periodic, plot, compare

tau = 2*math.pi

def v(*a): return np.array(a)
boundary = Periodic()

# constants
space_size = v(2,2)
num_P = v(65,65)
num_t = 100
c = 1
dP = space_size / (num_P - 1)
sigma = 0.2
dt = sigma * min(*dP)

# iterative
def _step_i(u):
    u = u.copy()
    for n in range(num_t):
        un = u.copy()
        row, col = u.shape
        for j in range(1, row):
            for i in range(1, col):
                u[j, i] = (un[j, i] - (c * dt / dx * (un[j, i] - un[j, i - 1])) -
                                      (c * dt / dy * (un[j, i] - un[j - 1, i])))
                u[0, :] = 1
                u[-1, :] = 1
                u[:, 0] = 1
                u[:, -1] = 1
    return u

# vectorized
def _step_v(u):
    for n in range(num_t):
        u[1:, 1:] = (u[1:, 1:] - (c * dt / dx * (u[1:, 1:] - u[1:, :-1])) -
                                  (c * dt / dy * (u[1:, 1:] - u[:-1, 1:])))
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1
    return u

P = [np.linspace(0, size, num) for size, num in zip(space_size,num_P)]
_u = np.ones(num_P)

# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
dx, dy = dP
_u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2

_u = _step_v(_u)

fig = pyplot.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(*P)
surf = ax.plot_surface(X, Y, _u, cmap=cm.viridis)

pyplot.show()
