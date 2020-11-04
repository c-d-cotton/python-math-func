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
import scipy.io as sio

def getmatlabmatrixstring(pythonarray):
    string = ''
    for i in range(len(pythonarray)):
        row = pythonarray[i]
        rowstring = '[' + ', '.join([str(cell) for cell in row]) + ']'
        string = string + rowstring + '; '
        # for j in len(pythonarray[0]):
        #     cell = pythonarray[i,j]

    # add closing and opening brackets
    string = '[' + string + ']'

    return(string)

    

def getmatlabmatrixstring_test():
    pythonarray = [[1,0,2], [3,2,4]]
    print(getmatlabmatrixstring(pythonarray))
def irf_matlab(matlabsave, names, varindex = None, matlabsavefunc = None, rowisperiods = False):
    """
    Get IRFs based upon a matlab save file
    matlabsavefunc allows me to apply a function to load a specific part of a matlabsave file

    By default, data in form vars x periods. If I specify, rowisperiods = True then data is input in the form periods x vars
    """
        
    if os.path.isfile(matlabsave):
        data = sio.loadmat(matlabsave)
    else:
        raise ValueError(matlabsave + ' does not exist')

    # use this function to get to a specific part of the matlab save file i.e. if many matrices are saved in it
    if matlabsavefunc is not None:
        data = matlabsavefunc(data)

    if rowisperiods is True:
        # want data ordered as vars x periods
        data = np.transpose(data)

    if varindex is not None:
        data = data[varindex, :]

    if len(data) != len(names):
        raise ValueError('Number of rows in data (' + str(len(data)) + ') should match length of names (' + str(len(names)) + '). May need to change rowisperiods = True/False.')

    importattr(__projectdir__ / Path('matplotlib_func.py'), 'gentimeplots_basic')(data, names)


