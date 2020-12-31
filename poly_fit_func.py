#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import functools
import numpy as np

# Imported Functions:{{{1
sys.path.append(str(__projectdir__ / Path('submodules/python-general-func/')))
from print_func import printorwrite
printorwrite = printorwrite

# General Fit Functions:{{{1
def getfullX(xvec, coeff):
    xvec = np.array(xvec).reshape([len(xvec), 1])

    X = np.ones([len(xvec), 1])

    for i in range(1, coeff + 1):
        X = np.concatenate([X, xvec ** i], axis = 1)

    return(X)


def getfullX_test():
    xvec = [1,2,3]
    X = getfullX(xvec, 2)
    print(X)


def polyestimate(xvec, coeff, yvec, outputfilename = None, printdetails = True):
    yvec = np.array(yvec).reshape([len(xvec), 1])

    X = getfullX(xvec, coeff)

    betahat = np.linalg.solve(X.transpose().dot(X), X.transpose().dot(yvec))

    # print/write details on polynomial
    printorwrite('Estimated polynomial: ' + ' + '.join([str(betahat[i]) + 'x**' + str(i) for i in range(len(betahat))]) + '.', outputfilename, printmessage = printdetails)

    return(betahat)


def polyestimate_test():
    xvec = [1,2,3]
    yvec = [2,5,10]

    betahat = polyestimate(xvec, 0, yvec)
    print(betahat)

    betahat = polyestimate(xvec, 1, yvec)
    print(betahat)

    betahat = polyestimate(xvec, 2, yvec)
    print(betahat)


def polyfit(coeff, betahat, xvec):
    X = getfullX(xvec, coeff)
    betahat = np.array(betahat).reshape([coeff + 1, 1])

    yhat = X.dot(betahat)

    # convert to a vector rather than a matrix
    yhat = yhat.reshape([len(yhat)])

    return(yhat)


def polyfit_test():
    xvec = [1, 2, 3]
    coeff = 2
    betahat = [1, 0, 1]

    yhat = polyfit(coeff, betahat, xvec)
    print(yhat)


def polyestfit_getfunc(coeff, xvec, yvec, outputfilename = None, printdetails = True):
    """
    Returns a function that converts x2vec into y2hatvec based upon the estimated coefficients from xvec, yvec
    """
    betahat = polyestimate(xvec, coeff, yvec, outputfilename = outputfilename, printdetails = printdetails)

    f1 = functools.partial(polyfit, coeff, betahat)
    # so if input [1, 2] into f1 then f1 returns the best fit of [1, 2] based upon the estimate of the polynomial of coeff from xvec and yvec

    return(f1)


def polyestfit_getfunc_test():
    coeff = 2
    xvec = [1, 2, 3, 4]
    yvec = [2, 5, 10, 17]

    f1 = polyestfit_getfunc(coeff, xvec, yvec)

    x2vec = [1.5, 2.5, 3.5]
    y2hatvec = f1(x2vec)
    print(y2hatvec)


def polyestfit_getvec(coeff, xvec, yvec, xvec_tofit, outputfilename = None, printdetails = True):
    f1 = polyestfit_getfunc(coeff, xvec, yvec, outputfilename = outputfilename, printdetails = printdetails)
    yhatvec = f1(xvec_tofit)
    return(yhatvec)


def polyestfit_getvec_test():
    coeff = 2
    xvec = [1, 2, 3, 4]
    yvec = [2, 5, 10, 17]
    xvec_tofit = [1.5, 2.5, 3.5]
    yfitvec = polyestfit_getvec(coeff, xvec, yvec, xvec_tofit)
    print(yfitvec)

