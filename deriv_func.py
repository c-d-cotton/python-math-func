#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import numpy as np
from functools import partial

# Use With Functions:{{{1
def getderiv(f1, x, derivnum = 1, epsilon = 1e-5):
    """
    Need to add possibility for higher derivatives.
    epsilon is the size of the band for the last derivative taken.

    Doesn't work very well!
    """
    epsilon = epsilon * 2 ** (derivnum - 1)

    derivfunclist = []
    derivfunclist.append(f1)

    for i in range(derivnum):
        # def nextderiv(y):
        #     ret = (derivfunclist[i](y + epsilon) - derivfunclist[i](y - epsilon)) / (2 * epsilon)
        #     return(ret)

        def nextderiv(y, j):
            ret = (derivfunclist[j](y + epsilon) - derivfunclist[j](y - epsilon)) / (2 * epsilon)
            return(ret)
        nextderiv2 = partial(nextderiv, j = i)

        derivfunclist.append(nextderiv2)

        # derivfunclist.append( lambda y: (derivfunclist[i](y + epsilon) - derivfunclist[i](y - epsilon)) / (2 * epsilon)   )

        epsilon = epsilon / 2
            
    ret = derivfunclist[-1](x)

    return(ret)

    
def getderiv(f1, x, derivnum = 1, epsilon = 1e-5):
    """
    Need to add possibility for higher derivatives.
    epsilon is the size of the band for the last derivative taken.
    """
    ret = (f1(x + epsilon) - f1(x - epsilon)) / (2 * epsilon)
    return(ret)

    
# Use With Arrays:{{{1
def basicderiv(xvec, yvec, derivnum = 1, returnfulldict = False):
    """
    Very basic derivative calculations for use with simulated lines.
    """
    xvec = np.array(xvec)
    yvec = np.array(yvec)

    derivdict = {}
    derivdict[0] = [xvec, yvec]
    xnew = []
    d = []

    for i in range(1, derivnum + 1):
        xv = derivdict[i - 1][0]
        yv = derivdict[i - 1][1]

        dx = xv[1: ] - xv[0: len(xv) - 1]
        dy = yv[1: ] - yv[0: len(yv) - 1]
        dydx = np.divide(dy, dx)
        xnew = xvec[: len(xv) - 1] + dx / 2

        derivdict[i] = [xnew, dydx]

    if returnfulldict is True:
        return(derivdict)
    else:
        return(derivdict[derivnum][0], derivdict[derivnum][1])


def test():
    xvec = np.linspace(0, 1, 100)
    yvec = xvec ** 2

    xnew, d2ydx2 = basicderiv(xvec, yvec, derivnum = 2)
    print('xvec:')
    print(xnew)
    print('d2ydx2:')
    print(d2ydx2)

        
