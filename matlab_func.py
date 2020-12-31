#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

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

    from matplotlib_func import gentimeplots_basic
    gentimeplots_basic(data, names)


