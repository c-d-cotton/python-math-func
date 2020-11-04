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
import matplotlib.pyplot as plt

# Plot Time Subplots:{{{1
def gentimegraphs(vardict, varnames, vardescdict = None, overalldesc = None, width = 2, savename = None, pltshow = False):
    import matplotlib.pyplot as plt

    height = (len(varnames) - 1) // width + 1

    if vardescdict is None:
        vardescdict = {}
    for var in varnames:
        if var not in vardescdict:
            vardescdict[var] = var

    timexvar = [t for t in list(range(1, len(vardict[varnames[0]]) + 1))]

    fig = plt.figure()
    axdict = {}
    for i in range(len(varnames)):
        j = i + 1
        axdict[j] = fig.add_subplot(height, width, j)
        axdict[j].plot(timexvar, vardict[varnames[i]])
        axdict[j].set_title(vardescdict[varnames[i]])

    if overalldesc is not None:
        plt.title('Hello World')

    plt.tight_layout()

    if savename is not None:
        plt.savefig(savename)

    if pltshow is True:
        plt.show()


def gentimeplots_basic(array, varnames, timex = None, overalldesc = None, width = 2, savename = None):
    """
    Generate basic plot from array. Row i of array corresponds to variable i in varnames.
    """
    import matplotlib.pyplot as plt

    if timex is None:
        timex = list(range(np.shape(array)[1]))

    height = (len(varnames) - 1) // width + 1

    fig = plt.figure()
    axdict = {}
    for i in range(len(varnames)):
        j = i + 1
        axdict[j] = fig.add_subplot(height, width, j)
        axdict[j].plot(timex, array[i, :])
        axdict[j].set_title(varnames[i])

    if overalldesc is not None:
        plt.title(overalldesc)

    plt.tight_layout()

    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()




    

# Plot Functions:{{{1
def plotfunctions(funcs, xarray, showtotal = False, derivnum = 0, logvec = False):
    """
    Plot a set of functions on the same graph.
    Maybe include a total.
    """
    funcvals = []
    for f in funcs:
        funcvals.append([f(x) for x in xarray])

    if logvec is True:
        xarray = np.log(xarray)
        for i in range(len(funcvals)):
            funcvals[i] = np.log(funcvals[i])

    funcvals2 = []
    for i in range(len(funcs)):
        xarray2, funcval2 = importattr(__projectdir__ / Path('deriv_func.py'), 'basicderiv')(xarray, funcvals[i], derivnum = derivnum)
        funcvals2.append(funcval2)

    for i in range(len(funcs)):
        plt.plot(xarray2, funcvals2[i], label = funcs[i].__name__)

    if showtotal is True:
        total = []
        for i in range(len(funcvals2[0])):
            total.append(np.sum([funcval2[i] for funcval2 in funcvals2]))
        plt.plot(xarray2, total, label = 'Total')

    plt.legend()
    
    plt.show()


def plotfunctions_example():
    def f1(x):
        return(x)
    def f2(x):
        return(1 - x)
    plotfunctions([f1, f2], np.linspace(0, 10, 100), showtotal = True, derivnum = 0)
