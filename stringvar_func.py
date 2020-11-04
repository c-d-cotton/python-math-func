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

import re

def varsinstring(string):
    revar = re.compile('(?<![a-zA-Z0-9_])([a-zA-Z][a-zA-Z0-9_]*)(?![a-zA-Z0-9_])')
    thevars = set()
    while True:
        match = revar.search(string)
        if match is not None:
            # add match to set of variables
            thevars.add(match.group(1))
            # only replace first match
            string = string.replace(match.group(1), '', 1)
        else:
            break

    return(thevars)


def varsinstring_test():
    print(varsinstring('a + ssssss * 1 = sssssssssss'))

    
def varsinstringanddict(string, listofvars):
    """
    Return set of variables that are in listofvars and string.
    """
    truevars = set()
    for var in listofvars:
        match = re.compile('(?<![a-zA-Z0-9_])' + var + '(?![a-zA-Z0-9_])').search(string)
        if match is not None:
            truevars.add(var)

    return(truevars)


def varsinstringanddict_test():
    listofvars = ['a', 'b', 'c']
    string = 'ssss + 2 * b = c'
    print(varsinstringanddict(string, listofvars))
   

def replacestring(equation, var, varvalue):
    return(re.sub('(?<![a-zA-Z0-9_])' + var + '(?![a-zA-Z0-9_])', str(varvalue), equation))


def replacevardict(equation, vardict):
    for var in vardict:
        equation = replacestring(equation, var, vardict[var])

    return(equation)


def replacevardict_test():
    equation = 'a + 333 * sssss = b * c'
    vardict = {'a': 1, 'b': 2}
    print(replacevardict(equation, vardict))


def evalstring(string):
    """
    
    """
    # import them this way so eval works fine with exp and log
    from numpy import exp
    from numpy import log

    string = string.replace('^', '**')

    try:
        number = eval(string)
    except Exception:
        raise ValueError('Eval failed')

    return(number)


def evalstring_test():
    print(evalstring("1 + 2 - 3 + log(1)"))
