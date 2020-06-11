import math
import numpy as np

from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots

import _lib

dim = 2
_a = _lib.mkarray((dim,))

# constants
# bounds = _lib.WallBoundary(_a(1))
wall = _a(1)
rbox = _a(2)
shape_r = _a(100) + 1
num_t = 80
c = 1
dr = rbox / (shape_r - 1)
sigma = .2
dt = sigma * dr[0]

# vectorized
def _step_vec(uv, nt):
    UV = _lib.SliceWindow(uv, dim)
    for n in range(nt):
        UV[:] = UV[:] - np.dot(UV.diff_prev().T * UV[:], c * dt / dr).T
        _lib.wall_boundary(shape_r, uv, wall)

    return uv

# set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
def set_hat(uv, dr, box=np.array([[.5, 1], [.5, 1]])):
    (xa, ya), (xb, yb) = (box[:, 0]/dr), (box[:, 1]/dr + 1)
    uv[int(xa):int(xb), int(ya):int(yb)] = _a(2)

def _run():
    ushape = tuple(shape_r) + (dim,)
    _uv = np.ones(ushape)

    box = np.array([[.5,1], [.5, 1]])
    set_hat(_uv, dr, box)

    _uv = _step_vec(_uv, num_t)
    r = [np.linspace(0, size, num) for size, num in zip(rbox, shape_r)]

    fig = pyplot.figure(figsize=(8, 8), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(*r)

    surf = ax.plot_surface(X, Y, _uv[:, :, 0], cmap=cm.viridis, rstride=2, cstride=2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$');

    pyplot.show()

if __name__ == '__main__': _run()

def test():pass
