# 12 channel flow
import math
import numpy as np
from matplotlib import pyplot, cm

import _lib

def build_b(rho, dt, dr, uv):
    dx, dy = dr
    u, v = uv
    b = np.zeros_like(u)

    b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                      (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                            2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                            ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    # Periodic BC Pressure @ x = 2
    b[1:-1, -1] = (rho * (1 / dt * ((u[1:-1, 0] - u[1:-1,-2]) / (2 * dx) +
                                    (v[2:, -1] - v[0:-2, -1]) / (2 * dy)) -
                          ((u[1:-1, 0] - u[1:-1, -2]) / (2 * dx))**2 -
                          2 * ((u[2:, -1] - u[0:-2, -1]) / (2 * dy) *
                               (v[1:-1, 0] - v[1:-1, -2]) / (2 * dx)) -
                          ((v[2:, -1] - v[0:-2, -1]) / (2 * dy))**2))

    # Periodic BC Pressure @ x = 0
    b[1:-1, 0] = (rho * (1 / dt * ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx) +
                                   (v[2:, 0] - v[0:-2, 0]) / (2 * dy)) -
                         ((u[1:-1, 1] - u[1:-1, -1]) / (2 * dx))**2 -
                         2 * ((u[2:, 0] - u[0:-2, 0]) / (2 * dy) *
                              (v[1:-1, 1] - v[1:-1, -1]) / (2 * dx))-
                         ((v[2:, 0] - v[0:-2, 0]) / (2 * dy))**2))

    return b

def pressure_poisson_periodic(p, dr, b, nit=50):
    dx, dy = dr
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Periodic BC Pressure @ x = 2
        p[1:-1, -1] = (((pn[1:-1, 0] + pn[1:-1, -2])* dy**2 +
                        (pn[2:, -1] + pn[0:-2, -1]) * dx**2) /
                       (2 * (dx**2 + dy**2)) -
                       dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, -1])

        # Periodic BC Pressure @ x = 0
        p[1:-1, 0] = (((pn[1:-1, 1] + pn[1:-1, -1])* dy**2 +
                       (pn[2:, 0] + pn[0:-2, 0]) * dx**2) /
                      (2 * (dx**2 + dy**2)) -
                      dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 0])

        # Wall boundary conditions, pressure
        p[-1, :] =p[-2, :]  # dp/dy = 0 at y = 2
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0

    return p

# (nt, uv, dt, dr, p, rho, nu)
def channel_flow(uv, dt, dr, p, rho, nu, udiff_target=.001):
    u, v = uv
    dx, dy = dr

    stepcount = 0
    udiff = 1
    while udiff > udiff_target:
        un = u.copy()
        vn = v.copy()
        b = build_b(rho, dt, dr, uv)
        p = pressure_poisson_periodic(p, dr, b)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                          un[1:-1, 1:-1] * dt / dx *
                         (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                          vn[1:-1, 1:-1] * dt / dy *
                         (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                          dt / (2 * rho * dx) *
                         (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                          nu * (dt / dx**2 *
                         (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                          dt / dy**2 *
                         (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])) +
                          F * dt)

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                          un[1:-1, 1:-1] * dt / dx *
                         (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                          vn[1:-1, 1:-1] * dt / dy *
                         (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                          dt / (2 * rho * dy) *
                         (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                          nu * (dt / dx**2 *
                         (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                          dt / dy**2 *
                         (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        # Periodic BC u @ x = 2
        u[1:-1, -1] = (un[1:-1, -1] - un[1:-1, -1] * dt / dx *
                       (un[1:-1, -1] - un[1:-1, -2]) -
                        vn[1:-1, -1] * dt / dy *
                       (un[1:-1, -1] - un[0:-2, -1]) -
                        dt / (2 * rho * dx) *
                       (p[1:-1, 0] - p[1:-1, -2]) +
                        nu * (dt / dx**2 *
                       (un[1:-1, 0] - 2 * un[1:-1,-1] + un[1:-1, -2]) +
                        dt / dy**2 *
                       (un[2:, -1] - 2 * un[1:-1, -1] + un[0:-2, -1])) + F * dt)

        # Periodic BC u @ x = 0
        u[1:-1, 0] = (un[1:-1, 0] - un[1:-1, 0] * dt / dx *
                      (un[1:-1, 0] - un[1:-1, -1]) -
                       vn[1:-1, 0] * dt / dy *
                      (un[1:-1, 0] - un[0:-2, 0]) -
                       dt / (2 * rho * dx) *
                      (p[1:-1, 1] - p[1:-1, -1]) +
                       nu * (dt / dx**2 *
                      (un[1:-1, 1] - 2 * un[1:-1, 0] + un[1:-1, -1]) +
                       dt / dy**2 *
                      (un[2:, 0] - 2 * un[1:-1, 0] + un[0:-2, 0])) + F * dt)

        # Periodic BC v @ x = 2
        v[1:-1, -1] = (vn[1:-1, -1] - un[1:-1, -1] * dt / dx *
                      (vn[1:-1, -1] - vn[1:-1, -2]) -
                       vn[1:-1, -1] * dt / dy *
                      (vn[1:-1, -1] - vn[0:-2, -1]) -
                       dt / (2 * rho * dy) *
                      (p[2:, -1] - p[0:-2, -1]) +
                       nu * (dt / dx**2 *
                      (vn[1:-1, 0] - 2 * vn[1:-1, -1] + vn[1:-1, -2]) +
                       dt / dy**2 *
                      (vn[2:, -1] - 2 * vn[1:-1, -1] + vn[0:-2, -1])))

        # Periodic BC v @ x = 0
        v[1:-1, 0] = (vn[1:-1, 0] - un[1:-1, 0] * dt / dx *
                     (vn[1:-1, 0] - vn[1:-1, -1]) -
                      vn[1:-1, 0] * dt / dy *
                     (vn[1:-1, 0] - vn[0:-2, 0]) -
                      dt / (2 * rho * dy) *
                     (p[2:, 0] - p[0:-2, 0]) +
                      nu * (dt / dx**2 *
                     (vn[1:-1, 1] - 2 * vn[1:-1, 0] + vn[1:-1, -1]) +
                      dt / dy**2 *
                     (vn[2:, 0] - 2 * vn[1:-1, 0] + vn[0:-2, 0])))


        # Wall BC: u,v = 0 @ y = 0,2
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0

        udiff = (np.sum(u) - np.sum(un)) / np.sum(u)
        stepcount += 1

    return (u, v), p

# constants
rho = 1
nu = .1
dt = .001
F = 1

def run(nr, dr):
    u = np.zeros(nr)
    v = np.zeros(nr)
    p = np.zeros(nr)
    nt = 100
    return channel_flow((u, v), dt, dr, p, rho, nu)

def plot_field(X, Y, p, uv):
    u, v = uv
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    pyplot.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3])
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

uv, p = run(nr, dr)
plot_field(*R, p, uv)
pyplot.show()
