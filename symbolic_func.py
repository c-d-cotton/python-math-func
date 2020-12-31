#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

def matchcoeffs_vec_python(allcoeff, eqs):
    import sympy
    tosolve = sympy.zeros(len(eqs), len(allcoeff))
    for i in range(len(eqs)):
        eq = eqs[i]
        for j in range(len(allcoeff)):
            var = allcoeff[j]
            tosolve[i,j] = sympy.Poly(eq, var).all_coeffs()[0]

    tosolve.reshape(len(eqs) * len(allcoeff), 1)

    return(tosolve, tosolve.free_symbols)
