import numpy as np
from _util import *

def eq(a,b): return np.all(a == b)

def test_diffprev():
    a = np.array([1, 2, 4, 6])
    ad = np.array([-5, 1, 2, 2])
    assert eq(diffprev(a), ad)
    ad2 = np.array([-7, 6, 1, 0])
    assert eq(diffprev(a, 2), ad2)

    m = np.array([1, 2, 4, 6]).reshape((2,2))
    m0d = np.array([[-3, -4], [ 3,  4]])
    m1d = np.array([[-1,  1], [-2,  2]])
    assert eq(diffprev(m, axis=0), m0d)
    assert eq(diffprev(m), m1d)

def test_diffnext():
    a = np.array([1, 2, 4, 6])
    ad = np.array([1, 2, 2, -5])
    assert eq(diffnext(a), ad)
    ad2 = np.array([1, 0, -7, 6])
    assert eq(diffnext(a, 2), ad2)

    m = np.array([1, 2, 4, 6]).reshape((2,2))
    m0d = np.array([[ 3,  4], [-3, -4]])
    m1d = np.array([[ 1, -1], [ 2, -2]])
    assert eq(diffnext(m, axis=0), m0d)
    assert eq(diffnext(m), m1d)
