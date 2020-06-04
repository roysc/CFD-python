import sympy
import numpy as np
from itertools import product

def array(shape, var='a'):
    names = var + ''.join(f'(0:{n})' for n in shape)
    return np.array(sympy.symbols(names)).reshape(shape)
