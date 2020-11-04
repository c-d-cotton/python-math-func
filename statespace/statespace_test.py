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

# gendata:{{{1
def simplestatespace():
    import numpy as np

    A = np.array([[0.5, 0.3], [0.2, 0.1]])
    B = np.array([[1, 0], [0, 0.9]])
    C = np.array([[0.1, 0.3]])
    D = np.array([[0, 0]])

    Omega = np.array([[1, 0], [0, 0.5]])

    return(A, B, C, D, Omega)


def test_gendata1():
    import numpy as np

    A, B, C, D, Omega = simplestatespace()

    v = np.zeros((100, 2))

    X0 = np.array([[1, 0]])

    X, Y = importattr(__projectdir__ / Path('statespace/statespace_func.py'), 'statespace_gendata')(A, B, C, D, v, X0 = X0)
    print(X)
    print(Y)


def test_gendata2():
    import numpy as np

    A, B, C, D, Omega = simplestatespace()

    v = np.random.normal(size = (100, 2))

    X, Y = importattr(__projectdir__ / Path('statespace/statespace_func.py'), 'statespace_gendata')(A, B, C, D, v)
    print(X)
    print(Y)

# simdata:{{{1
def test_simdata():
    A, B, C, D, Omega = simplestatespace()

    X, Y, v = importattr(__projectdir__ / Path('statespace/statespace_func.py'), 'statespace_simdata')(A, B, C, D, 40, Omega)

    print(Y)

# kalmanfilter:{{{1
def test_kalmanfilter():
    A, B, C, D, Omega = simplestatespace()

    x, y, v = importattr(__projectdir__ / Path('statespace/statespace_func.py'), 'statespace_simdata')(A, B, C, D, 40)

    x_t_tm1, P_t_tm1, x_t_t, P_t_t, y_t_tm1, Q_t_tm1, R_t_tm1 = importattr(__projectdir__ / Path('statespace/statespace_func.py'), 'kalmanfilter')(y, A, B, C, D, Omega = Omega)

    print(x_t_tm1)
    print(x_t_t)
    print(P_t_t)

def test2_kalmanfilter():
    import numpy as np

    y = np.array([[i] for i in [0, 2, -2, 0, 1, -1, 0, 0, 0]])
    A = np.array([[0.5, 0.1], [0.3, 0.1]])
    B = np.array([[1, 0], [0, 0]])
    C = np.array([[1, 1]])
    D = np.array([[0, 1]])
    
    x_t_tm1, P_t_tm1, x_t_t, P_t_t, y_t_tm1, Q_t_tm1, R_t_tm1 = importattr(__projectdir__ / Path('statespace/statespace_func.py'), 'kalmanfilter')(y, A, B, C, D)


def test_logl():
    """
    Simulate data for a true value of A. Then compare log likelihoods for alternative values of A.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    A, B, C, D, Omega = simplestatespace()

    x, y, v = importattr(__projectdir__ / Path('statespace/statespace_func.py'), 'statespace_simdata')(A, B, C, D, 1000)

    A_adjusted = A.copy()
    Aparam_val = np.linspace(0, 1, 100)
    logl_val = []
    for Aparam in Aparam_val:
        A_adjusted[0, 0] = Aparam

        x_t_tm1, P_t_tm1, x_t_t, P_t_t, y_t_tm1, Q_t_tm1, R_t_tm1 = importattr(__projectdir__ / Path('statespace/statespace_func.py'), 'kalmanfilter')(y, A_adjusted, B, C, D, Omega)

        logl_val.append(importattr(__projectdir__ / Path('statespace/statespace_func.py'), 'logl_prop_kalmanfilter')(y, y_t_tm1, Q_t_tm1))

    plt.plot(Aparam_val, logl_val)
    plt.xlabel(r'Parameter Value')
    plt.ylabel(r'Log Likelihood')
    plt.show()
    plt.clf()

    
# Run:{{{1
# test_logl()
test2_kalmanfilter()
