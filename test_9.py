import numpy as np
from sympy import symbols

import _util
from _lib import Window
from _sym import symarray

import _9

dx, dy = symbols('dx dy')
dr = np.array([dx, dy])
nr = (4,4)
p = symarray(nr, 'p')
p11 = p[1,1]
r = _util.linspaces([[0,0], [2,1]], nr)

def test_laplace():
    # swap beforehand because calc_* modifies array
    pold = p.copy().T
    pnew = p.copy()

    _9.calc_laplace_old(pold, dr, r)
    _9.calc_laplace(pnew, dr, r)
    assert np.all(pold.T == pnew)
