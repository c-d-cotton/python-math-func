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

# Markov Solve for Stationary Probs:{{{1
def getstationarydist(transmissionstararray, crit = 1e-9):
    """
    Use the transmissionstararray to find the stationary distribution of the problem from an arbitrary starting point.
    """
    ns = np.shape(transmissionstararray)[0]

    statetoday = np.ones([ns]) * 1/float(ns)

    while True:
        stateprime = np.dot(statetoday, transmissionstararray)

        diff = np.max(np.abs(stateprime - statetoday))
        statetoday = stateprime
        if diff < crit:
            break

    return(stateprime)
            

def getstationarydist2(A):
    """
    Took this from: https://dilawarnotes.wordpress.com/2017/11/07/stationary-distribution-of-markov-chains-using-python/
    Does a regression of zero on (I - A)^t
    Adjustment: Add row of 1s that should multiply by coefficients to equal 1 to ensure sums to 1

    Works for up to about 1000 then gets too slow
    
    x = xA where x is the answer
    x - xA = 0
    x( I - A ) = 0 and sum(x) = 1
    """
    n = A.shape[0]
    a = np.eye( n ) - A
    a = np.vstack( (a.T, np.ones( n )) )
    b = np.matrix( [0] * n + [ 1 ] ).T
    mat = np.linalg.lstsq( a, b )[0]
    return(np.vectorize(mat))


def getstationarydist3(transmissionstararray):
    """
    Get the non-one eigenvalue and then apply probs * transmission = probs
    """
    eigenvalues, eigenvectors = np.linalg.eig(transmissionstararray)
    for notonei in range(len(eigenvalues)):
        if eigenvalues[notonei].real - 1 < 1e-6:
            break
    stationarydist = eigenvectors[notonei, :]
    return(stationarydist)


def getstationarydist4(p):
    """
    Based on this: https://math.stackexchange.com/questions/1020681/finding-steady-state-probabilities-by-solving-equation-system
    """
    dim = p.shape[0]
    q = (p-np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q,ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    return np.linalg.solve(QTQ,bQT)


def markov_stationary_test(size = 100, printprobs = False, second = True):
    import datetime

    A = np.random.uniform(size = [size, size])
    for i in range(size):
        A[i, :] = A[i, :] / np.sum(A[i, :])

    b = np.random.uniform(size = [size])
    b = b / np.sum(b)

    # by main method
    start = datetime.datetime.now()
    stateprobs = importattr(__projectdir__ / Path('vfi_func.py'), 'getstationarydist')(A)
    if printprobs is True:
        print(stateprobs)
    print('Time taken main method: ' + str(datetime.datetime.now() - start))

    if second is True:
        # by second method
        start = datetime.datetime.now()
        stateprobs = importattr(__projectdir__ / Path('vfi_func.py'), 'getstationarydist2')(A)
        if printprobs is True:
            print(stateprobs)
        print('Time taken alternative method: ' + str(datetime.datetime.now() - start))


