import numpy as np
from sympy import symbols, cancel, simplify

import _util
from _lib import Window, s_
from _sym import symarray

import _11_old
import _11

simp = np.vectorize(cancel)

nr = (4,4)
mask = s_[1:-1,1:-1]

dx, dy, dt = symbols('dx dy dt')
rho, nu = symbols('rho nu')
dr = np.array([dx, dy])

u = symarray(nr, 'u')
v = symarray(nr, 'v')
b = symarray(nr, 'b')
p = symarray(nr, 'p')
uv = np.transpose((u, v), (1,2,0))

P = Window(p.copy(), mask=mask)
B = Window(b, mask=mask)

def test_b():
    uvt = (u.T, v.T)
    UV = Window(uv, 2, mask)
    bold = _11_old.build_b(rho, dt, uvt, dr).T
    bnew = _11.build_b(rho, dt, UV, dr).array
    assert np.all(bold == bnew)

def test_poisson():
    pold = _11_old.pressure_poisson(p.T, dr, b.T, nt=1).T
    pnew = _11.poisson(P, B, dr, nt=1).array
    assert np.all(simp(pold[1,1]) == simp(pnew[1,1]))

def test_cavity_flow():
    nt = 1
    UV = Window(uv, 2, mask)
    uvt = (u.T, v.T)

    # expects k, j, i
    _11_old.cavity_flow(uvt, p.T, dr, dt, rho, nu)
    _11.cavity_flow(UV, P, dr, dt, rho, nu)

    # comparing as i, j, k
    uvt = np.transpose(uvt)
    assert uvt.shape == UV.array.shape

    for k in (0,1):
        a, b = cancel(uvt[1,1,k]), cancel(UV.array[1,1,k])
        assert a == b, a-b
