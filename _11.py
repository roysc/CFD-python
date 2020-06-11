# 11 cavity flow
import math
import numpy as np
from matplotlib import pyplot, cm

import _lib
from _lib import Window, s_

# transpose first to last
def T0(a):
    axes = tuple(range(1, len(np.shape(a)))) + (0,)
    return np.transpose(a, axes)

def diag(a):
    return np.array([a[i][i] for i in range(len(a))])

def build_b(rho, dt, UV, dr):
    m = s_[1:-1, 1:-1]
    b = np.zeros_like(UV.array, shape=UV.data_shape)
    B = Window(b, 1, m)

    D = (UV.n[1] - UV.n[-1]).transpose((3,0,1,2))
    C = T0(diag(D)) / (2 * dr)
    B[:] = rho * (1 / dt * np.sum(C, axis=-1) -
                  2 * np.prod(T0(diag(np.flip(D, axis=0))) / (2 * dr), axis=-1) -
                  np.sum(C * C, axis=-1))
    return B

def poisson_bounds(p):
    # Neumann conditions
    p[0, :] = p[1, :]
    p[-1,:] = p[-2, :]
    p[:, 0] = p[:, 1]
    p[:,-1] = 0

def poisson(P, B, dr, nt=50):
    for _ in range(nt):
        N = np.transpose(P.n[-1] + P.n[1], (1,2,0))
        P[:] = (np.dot(N, np.flip(dr)**2) - B[:] * np.prod(dr**2)) / (2 * np.dot(dr, dr))
        poisson_bounds(P.array)
    return P

def cavity_bounds(uv):
    uv[0, :]  = [0, 0]
    uv[-1, :] = [0, 0]
    uv[:, 0]  = [0, 0]
    uv[:, -1] = [1, 0]    # set velocity on cavity lid equal to 1

def cavity_flow(UV, P, dr, dt, rho, nu):
    UV[:] += (
        - np.sum(UV.diff_prev().T * (UV[:] * dt/dr), axis=-1).T
        - T0(P.n[1] - P.n[-1]) * dt / (2 * rho * dr)
        + nu * np.dot(T0(UV.diff_central()), dt / dr**2)
    )

def solve(UV, P, nt, dt, dr, rho, nu):
    for n in range(nt):
        B = build_b(rho, dt, UV, dr)
        P = poisson(P, B, dr, nt)
        cavity_flow(UV, P, dr, dt, rho, nu)
        cavity_bounds(UV.array)
    return UV, P

# constants
rho = 1
nu = .1
dt = .001

def run(nr, dr):
    m = s_[1:-1, 1:-1]
    uv = np.zeros(tuple(nr) + (dim,))
    p = np.zeros(nr)
    UV = Window(uv, dim, m)
    P = _lib.Window(p, mask=m)

    nt = 100
    return solve(UV, P, nt, dt, dr, rho, nu)

def plot_field(X, Y, p, u, v):
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    pyplot.contour(X, Y, p, alpha=0.5, cmap=cm.viridis)
    pyplot.colorbar()
    pyplot.contour(X, Y, p, cmap=cm.viridis)
    pyplot.streamplot(X, Y, u, v)
    pyplot.xlabel('X')
    pyplot.ylabel('Y')

dim = 2
_a = _lib.mkarray((dim,))
nr = _a(41)
rmin = _a(0)
rmax = _a(2)
dr = (rmax - rmin)/(nr - 1)
r = _lib.linspaces([rmin, rmax], nr)
R = np.meshgrid(*r)

if __name__ == '__main__':
    UV, P = run(nr, dr)
    plot_field(*R, P.array.T, *UV.array.T)
    pyplot.show()
