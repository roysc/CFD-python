from _lib import *
import _sym

def test_window():
    shape = (3,3)
    u = _sym.array(shape, 'u')
    s = SliceWindow(u)
    assert s[:].shape == shape

    sdp = s.diff_prev()
    assert sdp[0,0,0] == u[0,0] - u[-1,0]
    assert sdp[1,0,0] == u[0,0] - u[0,-1]

    sdn = s.diff_next()
    assert sdn[0,0,0] == u[1,0] - u[0,0]
    assert sdn[1,0,0] == u[0,1] - u[0,0]

    sdc = s.diff_central()
    assert sdc[0,0,0] == u[1,0] - 2*u[0,0] + u[-1,0]
    assert sdc[1,0,0] == u[0,1] - 2*u[0,0] + u[0,-1]
