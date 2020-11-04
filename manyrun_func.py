#!/usr/bin/env python3
"""
Run function and save for many different values.
Then have function to access those values.
"""
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

import copy
import datetime
import decimal
import functools
import itertools
import multiprocessing
import pickle
import shutil

# Defaults:{{{1
starttime = datetime.datetime.now()

# Save Runs:{{{1
def functiontorun2_topartial(functiontorun, savefolder, printdetails, starttime, skipfileifexists, retdict_saveelements, retdict_saveelementsskipbad, retdict_stringsave, retdict_stringsaveelements, retdict_stringsaveskipbad, inputvalue):
    """
    functiontorun is the basic function I input into runfunc_manyvalues
    savefolder is where I save the results
    printdetails = True means I save details when start a new folder and when I start/end a function run
    starttime is what I compare the current time to in printdetails
    skipfileifexists = True means I skip the file if there is already a file in the folder with its savename
    inputvalue is the actual value inputted into functiontorun - note that this could be a list if I want to pass multiple values

    Elements if the functiontorun returns a dictionary:
    retdict_saveelements: Only save certain elements of the retdict
    retdict_saveelementsskipbad = True. If elements aren't there, skip them
    retdict_stringsave = True: Save a text file with info on the returned elements in the retdict
    retdict_stringsaveelements: If this is a list, only save elements in this list in the text file. By default, this equals retdict_saveelements
    retdict_stringsaveskipbad = True: If an item is not in retdict_stringsaveelements or if saving an element as a string fails then do ignore that element and do not return an error.

    functiontorun2 saves the outputted version of functiontorun.
    """
    if isinstance(inputvalue, list) or isinstance(inputvalue, tuple):
        # if inputvalue = [0.1, 0.2], save as '0.1_0.2'
        saveinputvaluename = '_'.join([str(inputvalue[i]) for i in range(len(inputvalue))])
    else:
        saveinputvaluename = str(inputvalue)

    savepicklename = saveinputvaluename + '.pickle'
    savetxtname = saveinputvaluename + '.txt'

    if os.path.isfile(os.path.join(savefolder, savepicklename)) and skipfileifexists is True:
        print('Skipped inputvalue since exists already: ' + savepicklename)
        return(0)

    if printdetails is True:
        print('Started inputvalue value: ' + saveinputvaluename + '. Time elapsed: ' + str(datetime.datetime.now() - starttime))

    # get returnvalue
    # convert input value to float (in case I changed it to decimal when getting the value to save as)
    ret = functiontorun(inputvalue)

    # only if dictionary
    if isinstance(ret, dict):
        # get string dict to save
        # need to do before deleting elements from ret
        if retdict_stringsave is True:
            if retdict_stringsaveelements is not None:
                # save elements for string save
                if retdict_stringsaveskipbad is True:
                    retstringdict = {element: ret[element] for element in retdict_stringsaveelements if element in ret}
                else:
                # since no boolean, element must be in ret otherwise error
                    retstringdict = {element: ret[element] for element in retdict_stringsaveelements}
            else:
                retstringdict = copy.deepcopy(ret)

            # save string list
            with open(os.path.join(savefolder, savetxtname), 'w+') as f:
                f.write('\n'.join([str(element) + ': ' + str(retstringdict[element]) for element in retstringdict]))

        # delete elements not in retdict_saveelements from ret
        if retdict_saveelements is not None:
            if retdict_saveelementsskipbad is True:
                ret = {element: ret[element] for element in retdict_saveelements if element in ret}
            else:
                # since no boolean, element must be in ret otherwise error
                ret = {element: ret[element] for element in retdict_saveelements}


    with open(os.path.join(savefolder, savepicklename), 'wb') as f:
        pickle.dump(ret, f, pickle.HIGHEST_PROTOCOL)

    if printdetails is True:
        print('Finished inputvalue value: ' + saveinputvaluename + '. Time elapsed: ' + str(datetime.datetime.now() - starttime))


def runfunc_manyvalues(functiontorun, inputvalues, savefolder, deletefolder = False, decimalplaces = None, printdetails = True, dopool = True, skipfolderifexists = False, skipfileifexists = True, starttime = starttime, retdict_saveelements = None, retdict_saveelementsskipbad = True, retdict_stringsave = True, retdict_stringsaveelements = None, retdict_stringsaveskipbad = True):
    """
    Save return value of functiontorun as pickle in savefolder for each inputvalue in inputvalues.

    Options:
    - skipfolderifexists is True means skip savefolder if it exists already
    - skipfileifexists = True means skip file if it exists already
    - deletefolder = True means delete folder
    - dopool = True means use multiprocessing
    - starttime: base starttime when printing details
    - printdetails = True: print details during run
    - decimalplaces = 5: Any floats in inputvalues are rounded to the appropriate number of decimal places
    - retdict stuff: see functiontorun2_topartial

    - toproduct = False: If I have two parameters, ALPHA = [0.3, 0.4] and BETA = [0.9, 1], then I could input different combinations of these by doing inputvalues = [[0.3, 0.9], [0.3, 1], [0.4, 0.9], [0.4, 1]]. Alternatively, I can set toproduct = True and set inputvalues = [[0.3, 0.4], [0.9, 1]] and this will be converted to the case we need by the function.
    """
    if skipfolderifexists is True and os.path.isdir(savefolder):
        print('Skipped manyrun_func.py on ' + savefolder)
        return(0)

    if deletefolder is True:
        if os.path.isdir(savefolder):
            shutil.rmtree(savefolder)
    if not os.path.isdir(savefolder):
        os.mkdir(savefolder)

    if decimalplaces is not None:
        if isinstance(inputvalues[0], list):
            # if a list of lists
            for i in range(len(inputvalues)):
                for j in range(len(inputvalues[i])):
                    if isinstance(inputvalues[i][j], float) is True:
                        inputvalues[i][j] = round(decimal.Decimal(inputvalues[i][j]), decimalplaces)
            # inputvalues = [round(decimal.Decimal(inputvalues[i][j], decimalplaces)) for i in range(len(inputvalues)) for j in range(len(inputvalues[i])) if isinstance(inputvalues[i][j], float) is True]
        else:
            # if a list of elements
            for i in range(len(inputvalues)):
                if isinstance(inputvalues[i], float) is True:
                    inputvalues[i] = round(decimal.Decimal(inputvalues[i]), decimalplaces)
            # inputvalues = [round(decimal.Decimal(inputvalues[i]), decimalplaces) for i in range(len(inputvalues)) if isinstance(inputvalues[i], float) is True]

    if retdict_stringsaveelements is None:
        retdict_stringsaveelements = copy.deepcopy(retdict_saveelements)

    # get functiontorun2 which calls and saves functiontorun
    functiontorun2 = functools.partial(functiontorun2_topartial, functiontorun, savefolder, printdetails, starttime, skipfileifexists, retdict_saveelements, retdict_saveelementsskipbad, retdict_stringsave, retdict_stringsaveelements, retdict_stringsaveskipbad)

    if dopool is True:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # chunksize means do in order
        pool.map(functiontorun2, inputvalues, chunksize = 1)
    else:
        for i in range(len(inputvalues)):
            inputvalue = functiontorun2(inputvalues[i])


# Load Runs:{{{1
def runs_get(savefolder):
    """
    """
    savenames = sorted([filename for filename in os.listdir(savefolder) if filename.endswith('.pickle')])

    inputvalues = []
    outputvalues = []
    for savename in savenames:
        inputvalue = savename[: -7]

        with open(os.path.join(savefolder, savename), 'rb') as handle:
            ret = pickle.load(handle)

        inputvalues.append(inputvalue)
        outputvalues.append(ret)

    # may as well also return p since need this to get n etc.
    return(inputvalues, outputvalues)


# savings_supply_get(__projectdir__ / Path('temp/olg-idio/basic-attempt/'))
        


# Test:{{{1
def testfunc1var(x):
    retdict = {}
    retdict['x'] = x
    retdict['y'] = x + 1
    return(retdict)


def testfunc1var_many():
    inputvalues = [0, 1, 2, 2.5, 3]
    savefolder = '/tmp/testmanyrun/'

    runfunc_manyvalues(testfunc1var, inputvalues, savefolder, deletefolder = True, decimalplaces = 2, retdict_stringsaveelements = ['y', 'z'])

    a, b = runs_get(savefolder)
    a = [float(a) for a in a]
    print(a)
    print(b)

def testfunc2var(alist):
    x = alist[0]
    y = alist[1]

    retdict = {}
    retdict['x'] = x
    retdict['y'] = y
    retdict['z'] = x + y
    return(retdict)


def testfunc2var_many():
    xvalues = [0, 1.1]
    yvalues = [1.1, 2.2]
    inputvalues = list(itertools.product(xvalues, yvalues))
    print(inputvalues)

    savefolder = '/tmp/testmanyrun/'

    runfunc_manyvalues(testfunc2var, inputvalues, savefolder, deletefolder = True, decimalplaces = 2, retdict_stringsaveelements = ['y', 'z'])

    a, b = runs_get(savefolder)
    # a = [float(a) for a in a]
    print(a)
    print(b)

