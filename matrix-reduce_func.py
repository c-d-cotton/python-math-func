#!/usr/bin/env python3
# PYTHON_PREAMBLE_START_STANDARD:{{{

# Christopher David Cotton (c)
# http://www.cdcotton.com

# modules needed for preamble
import importlib
import os
from pathlib import Path
import sys

# Get full real filename
__fullrealfile__ = os.path.abspath(__file__)

# Function to get git directory containing this file
def getprojectdir(filename):
    curlevel = filename
    while curlevel is not '/':
        curlevel = os.path.dirname(curlevel)
        if os.path.exists(curlevel + '/.git/'):
            return(curlevel + '/')
    return(None)

# Directory of project
__projectdir__ = Path(getprojectdir(__fullrealfile__))

# Function to call functions from files by their absolute path.
# Imports modules if they've not already been imported
# First argument is filename, second is function name, third is dictionary containing loaded modules.
modulesdict = {}
def importattr(modulefilename, func, modulesdict = modulesdict):
    # get modulefilename as string to prevent problems in <= python3.5 with pathlib -> os
    modulefilename = str(modulefilename)
    # if function in this file
    if modulefilename == __fullrealfile__:
        return(eval(func))
    else:
        # add file to moduledict if not there already
        if modulefilename not in modulesdict:
            # check filename exists
            if not os.path.isfile(modulefilename):
                raise Exception('Module not exists: ' + modulefilename + '. Function: ' + func + '. Filename called from: ' + __fullrealfile__ + '.')
            # add directory to path
            sys.path.append(os.path.dirname(modulefilename))
            # actually add module to moduledict
            modulesdict[modulefilename] = importlib.import_module(''.join(os.path.basename(modulefilename).split('.')[: -1]))

        # get the actual function from the file and return it
        return(getattr(modulesdict[modulefilename], func))

# PYTHON_PREAMBLE_END:}}}

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


