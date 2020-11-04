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

def masmoothseries(series, ma):
    series = np.array(series)
    smoothedseries = np.array([np.nan] * len(series))

    for i in range(ma // 2, len(series) - ma // 2):
        if ma % 2 == 0:
            # even case
            smoothedseries[i] = 1 / ma * (0.5 * series[i - ma // 2] + np.sum(series[i - ma // 2 + 1: i + ma //2]) + 0.5 * series[i + ma // 2])
        else:
            # odd case
            smoothedseries[i] = 1 / ma * ( np.sum(series[i - ma // 2: i + ma //2 + 1]) )

    return(smoothedseries)


def masmoothseries_test():
    print( masmoothseries([0, 1, 2, 3, 4], 2) )
    print( masmoothseries([0, 1, 2, 3, 4], 3) )
    print( masmoothseries([0, 1, 2, 3, 4, 56], 4) )
