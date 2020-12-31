#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import numpy as np
import sympy

def reducematrix(A, B):
    """
    AX = BY where A is m1 x n and B is m2 x n where m2 < n, m1 + m2 >= n and B is rank m2.
    This function returns C, D where CX = 0 and Y = DX.
    It works by doing row reduction on B until we have removed all Y.

    Basic idea:
    Start with something like [[3,4],[1,2]][x1; x2] = [1; 1] [y]
    Can always subtract row2 from row1 to get
    [[2,2],[1,2]][x1; x2] = [0; 1][y]
    This can be rewritten as:
    [2, 2][x1; x2] = 0 and [1, 2][x1; x2] = [1][y]
    """
    merged = np.column_stack((-B, A))
    merged = sympy.Matrix(merged)
    echelonform, pivotcols = merged.rref()

    m2 = np.shape(B)[1]

    # verify that D is of rank m2:
    for i in range(0, m2):
        if i not in pivotcols:
            print(echelonform)
            raise ValueError('D is not of rank m2. Column ' + str(i) + ' is not a pivot column')

    echelonform = np.array(echelonform)

    # we know that echelonform should take form [[identity_{m2}, -D], [0, C]] [Y; X] = [0; 0]
    D = -echelonform[0: m2, m2: ]
    C = echelonform[m2: , m2: ]

    return(C, D)


def test0():
    A = np.array([[3, 4], [1, 2]])
    B = np.array([[1], [1]])


    C, D = reducematrix(A, B)
    print('C')
    print(C)
    print('D')
    print(D)


def test1():
    A = np.array([[1,2,3], [4,5,6]])
    B = np.array([[1], [0.5]])


    C, D = reducematrix(A, B)
    print('C')
    print(C)
    print('D')
    print(D)

# test1()


