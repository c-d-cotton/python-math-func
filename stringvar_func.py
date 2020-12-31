#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

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
