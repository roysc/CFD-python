import numpy as np
from sympy import symbols

import _util
from _lib import Window
from _sym import symarray

import _10

nr = (4,4)
dx, dy = symbols('dx dy')
dr = np.array([dx, dy])
dt = symbols('dt')
p = symarray(nr, 'p')
b = symarray(nr, 'b')

def test_poisson():
    pnew = p.copy()
    pold = _10.poisson_old(p, b, dr, nt=1)
    pnew = _10.poisson(pnew.T, b.T, dr, nt=1).T

    assert np.all(pold == pnew)
