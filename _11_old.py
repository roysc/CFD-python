# 11 cavity flow
import math
import numpy as np
from matplotlib import pyplot, cm

import _lib

def build_b(rho, dt, uv, dr):
    dx, dy = dr
    u, v = uv
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = rho * \
        (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                   (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
         ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
         2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
              (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
         ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2)

    return b

def pressure_poisson(p, dr, b, nt=50):
    dx, dy = dr
    for q in range(nt):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) *
                          b[1:-1,1:-1])

        p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[-1, :] = 0        # p = 0 at y = 2

    return p

def cavity_flow(uv, p, dr, dt, rho, nu):
    u, v = uv
    dx, dy = dr
    un = u.copy()
    vn = v.copy()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                     vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                     nu * (dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                           dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

    v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                    un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                    vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                    dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                    nu * (dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                          dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

def solve(nt, uv, dt, dr, p, rho, nu):
    u, v = uv
    b = np.zeros_like(u)

    for n in range(nt):
        # breakpoint()
        b = build_b(rho, dt, uv, dr)
        p = pressure_poisson(p, dr, b, nt)
        cavity_flow(uv, p, dr, dt, rho, nu)

        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = 1    # set velocity on cavity lid equal to 1
        v[0, :]  = 0
        v[-1, :] = 0
        v[:, 0]  = 0
        v[:, -1] = 0

    return (u, v), p

# constants
rho = 1
nu = .1
dt = .001

def run(nr, dr):
    u = np.zeros(nr)
    v = np.zeros(nr)
    p = np.zeros(nr)
    nt = 100
    return solve(nt, (u, v), dt, dr, p, rho, nu)

def plot_field(X, Y, p, uv):
    u, v = uv
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    pyplot.contour(X, Y, p, alpha=0.5, cmap=cm.viridis)
    pyplot.colorbar()
    pyplot.contour(X, Y, p, cmap=cm.viridis)
    # pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
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
    uv, p = run(nr, dr)
    plot_field(*R, p, uv)
    pyplot.show()
