from itertools import product
import pytest

from _window import *
from _util import s_
import _sym

@pytest.mark.window
@pytest.mark.parametrize('shape', [(2,), (3,), (4,)])
def test_window_1d(shape):
    u = _sym.symarray(shape, 'u')
    s = Window(u)
    assert s[:].shape == shape

    Nn = s.neighbors(1)
    assert np.shape(Nn) == shape
    assert (Nn[0] == u[1])

    Np = s.neighbors(-1)
    assert np.shape(Np) == shape
    assert (Np[0] == u[-1])

    sdp = s.diff_prev()
    assert sdp[0] == u[0] - u[-1]
    sdn = s.diff_next()
    assert sdn[0] == u[1] - u[0]
    sdc = s.diff_central()
    assert sdc[0] == u[1] - 2*u[0] + u[-1]

    wshape = shape + (2,)
    w = _sym.symarray(wshape, 'w')
    s = Window(w, 2)
    assert s.D == len(shape)
    assert s[:].shape == wshape
    assert s.neighbors(1).shape == wshape

@pytest.mark.parametrize('shape', [(2,), (3,), (4,)])
def test_masking_1d(shape):
    u = _sym.symarray(shape, 'u')
    _shape = np.array(shape)

    s = Window(u, mask=s_[1:])
    Nn = s.neighbors(1)
    assert (Nn[0] == u[2 % shape[0]])

    s = Window(u, mask=s_[1:])
    Np = s.neighbors(-1)
    assert (Np[0] == u[0])

    s = Window(u, mask=s_[:-1])
    Nn = s.neighbors(1)
    assert (Nn[0] == u[1])

    s = Window(u, mask=s_[:-1])
    Np = s.neighbors(-1)
    assert (Np[0] == u[-1])


@pytest.mark.parametrize('shape', product(*(range(2,4),)*2))
def test_window_2d(shape):
    u = _sym.symarray(shape, 'u')
    s = Window(u)
    assert s[:].shape == shape

    Nn = s.neighbors(1)
    assert np.all(Nn[:,0,0] == [u[1,0], u[0,1]])

    Np = s.neighbors(-1)
    assert np.all(Np[:,0,0] == [u[-1,0], u[0,-1]])

    sdp = s.diff_prev()
    assert sdp[0,0,0] == u[0,0] - u[-1,0]
    assert sdp[1,0,0] == u[0,0] - u[0,-1]

    sdn = s.diff_next()
    assert sdn[0,0,0] == u[1,0] - u[0,0]
    assert sdn[1,0,0] == u[0,1] - u[0,0]

    sdc = s.diff_central()
    assert sdc[0,0,0] == u[1,0] - 2*u[0,0] + u[-1,0]
    assert sdc[1,0,0] == u[0,1] - 2*u[0,0] + u[0,-1]

@pytest.mark.parametrize('shape', product(*(range(2,4),)*2))
def test_masking_2d(shape):
    shape = (2,2)
    u = _sym.symarray(shape, 'u')
    _shape = np.array(shape)

    slice_shapediffs = [
        (s_[1:], (1,0)),
        (s_[:-1], (1,0)),
        (s_[1:-1], (2,0)),
        (s_[:, 1:], (0,1)),
        (s_[:, :-1], (0,1)),
    ]
    for slices, shapediff in slice_shapediffs:
        s = Window(u, mask=slices)
        wshape = tuple(_shape - shapediff)
        assert s[:].shape == wshape
        Nn = s.neighbors(1)
        assert np.shape(Nn) == (2,) + wshape

    s = Window(u, mask=s_[1:])
    Nn = s.neighbors(1)
    assert (Nn[0,0,0] == u[2 % shape[0],0])
    assert (Nn[1,0,0] == u[1,1])
    Np = s.neighbors(-1)
    assert (Np[0,0,0] == u[0,0])
    assert (Np[1,0,0] == u[1,-1])

    s = Window(u, mask=s_[1:, 1:])
    Nn = s.neighbors(1)
    assert (Nn[0,0,0] == u[2 % shape[0],1])
    assert (Nn[1,0,0] == u[1, 2 % shape[1]])
    Np = s.neighbors(-1)
    assert (Np[0,0,0] == u[0,1])
    assert (Np[1,0,0] == u[1,0])

    s = Window(u, mask=s_[:-1, :-1])
    Nn = s.neighbors(1)
    assert (Nn[0,0,0] == u[1, 0])
    assert (Nn[1,0,0] == u[0, 1])
    Np = s.neighbors(-1)
    assert (Np[0,0,0] == u[-1,0])
    assert (Np[1,0,0] == u[0,-1])

def test_masking_2d_4x4():
    shape = (4,4)
    u = _sym.symarray(shape, 'u')
    _shape = np.array(shape)

    s = Window(u, mask=s_[1:-1, 1:-1])
    Nn = s.neighbors(1)
    assert (Nn[0,0,0] == u[2,1])
    assert (Nn[1,0,0] == u[1,2])
    Np = s.neighbors(-1)
    assert (Np[0,0,0] == u[0,1])
    assert (Np[1,0,0] == u[1,0])
