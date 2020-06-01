import math
import numpy as np
from matplotlib import pyplot
import sympy
from sympy import init_printing
from sympy.utilities.lambdify import lambdify

from _lib import Periodic, plot, compare

tau = 2*math.pi
init_printing(use_latex=True)

def v(*a): return a[0]
boundary = Periodic()

# constants
space_size = v(2,)
num_x = v(100,)
dx = space_size / num_x
num_t = v(24,)
nu = 0.3
sigma = 0.2
dt = sigma * dx**2 / nu

def rolldiff(a, shift=1):
    return np.roll(np.diff(a, append=a[0:shift]), shift)

# iterative
def _step_i(u):
    u = u.copy()
    for n in range(num_t):
        un = u.copy()
        for i in range(1, num_x-1):
            u[i] = un[i] - un[i] * dt / dx *(un[i] - un[i-1]) + nu * dt / dx**2 *\
                    (un[i+1] - 2 * un[i] + un[i-1])
        u[0] = un[0] - un[0] * dt / dx * (un[0] - un[-2]) + nu * dt / dx**2 *\
                    (un[1] - 2 * un[0] + un[-2])
        u[-1] = u[0]
    return u

# vectorized
def _step_v(u):
    for n in range(num_t):
        u -= u * dt / dx * rolldiff(u)
        u += nu * dt/dx**2 * (rolldiff(np.roll(u, -1)) - rolldiff(u))
    return u

# initial condition
def u0_func():
    x, nu, t = sympy.symbols('x nu t')
    phi = (sympy.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) +
           sympy.exp(-(x - 4 * t - 2 * sympy.pi)**2 / (4 * nu * (t + 1))))
    return lambdify((t, x, nu),
                    -2 * nu * (phi.diff(x) / phi) + 4)

x = np.linspace(0, tau, num_x)
u0 = u0_func()
_u = np.asarray([u0(0, x0, nu) for x0 in x])

# not equal due to off-by-1 size of iterative w/ copied boundary cell
print(compare(x, _u, (_step_v, _step_i)))
plot(x, _u, 32, _step_i)
