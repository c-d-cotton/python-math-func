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

# Continuous variables represented by discrete points:{{{1
def weightvaluediscretevec(value, discretevec):
    """
    I have a value 3.4 and I want to reflect this as best I can in a discrete vector, [1,2,3,4,5].
    This function would output [0,0,0.6,0.4,0].
    discretevec should obviously be ordered.
    Useful for value function iteration.
    """

    # istar is the first element of discretevec which value is less than
    istar = 0
    for i in range(len(discretevec)):
        if discretevec[i] < value:
            istar += 1

    probvec = np.zeros(len(discretevec))
    if istar == 0:
        probvec[0] = 1
    elif istar == len(discretevec):
        probvec[-1] = 1
    else:
        # (3.4 - 3)/(4 - 3)
        probvec[istar] = (value - discretevec[istar - 1]) / (discretevec[istar] - discretevec[istar - 1])
        # (4 - 3.4)/(4 - 3)
        probvec[istar - 1] = (discretevec[istar] - value) / (discretevec[istar] - discretevec[istar - 1])

    return(probvec)

        


def weightvaluediscretevec_test():
    print(weightvaluediscretevec(-0.6, [0, 1, 2]))
    print(weightvaluediscretevec(0, [0, 1, 2]))
    print(weightvaluediscretevec(0.4, [0, 1, 2]))
    print(weightvaluediscretevec(1, [0, 1, 2]))
    print(weightvaluediscretevec(1.4, [0, 1, 2]))
    print(weightvaluediscretevec(2, [0, 1, 2]))
    print(weightvaluediscretevec(2.4, [0, 1, 2]))


def weightvalue_orderedlist(orderedlist, discretevec):
    """
    I have some ordered list [1,2,3,4] and I want to express it in terms of weights on discretevec [2,2.5,3,3.5].
    Return two lists and two integers: firstnonzeroweightindex, weightonfirst, firstindexonly, lastindexonly
    """
    # j denotes the index of discretevec that we are currently considering
    firstindexonly = 0
    while orderedlist[firstindexonly] <= discretevec[0]:
        firstindexonly += 1
        if firstindexonly == len(orderedlist):
            break

    lastindexonly = 0
    while orderedlist[len(orderedlist) - lastindexonly - 1] >= discretevec[-1]:
        lastindexonly += 1
        if lastindexonly == len(orderedlist):
            break

    firstnonzeroweightindex = []
    weightonfirst = []
    # j is index of discretevec we're considering
    j = 0
    for i in range(firstindexonly, len(orderedlist) - lastindexonly):
        # raise discretevec element we're considering until in right range
        while discretevec[j + 1] <= orderedlist[i]:
            j = j + 1

        firstnonzeroweightindex.append(j)
        weight = (discretevec[j + 1] - orderedlist[i]) / (discretevec[j + 1] - discretevec[j])
        weightonfirst.append(weight)


    return(firstnonzeroweightindex, weightonfirst, firstindexonly, lastindexonly)


def weightvalue_orderedlist_test():
    print(weightvalue_orderedlist([1,2,3,4], [2,3,4]))
    print(weightvalue_orderedlist([2.4, 3.6], [2,3,4]))
    print(weightvalue_orderedlist([1.4, 2.4, 3.6, 4.6], [2,3,4]))


def weightvaluediscretevec_i(value, discretevec):
    """
    Rather than return a vector of weights, just return the relevant index of the last element of the array which value is less than
    """
    # istar is the first element of discretevec which value is less than
    istar = -1
    for i in range(len(discretevec)):
        if discretevec[i] < value:
            istar += 1
    if discretevec[i] == -1:
        raise ValueError('value < all values in discretevec')
    if discretevec[i] == len(discretevec) - 1:
        raise ValueError('value > all values in discretevec')

    return(istar)


def intervals_gt(value, discretevec):
    """
    Get probability that value is greater than points in set of intervals.
    discretevec is ordered list
    If discretevec is [0, 1, 2] then intervals are given by [0 < x < 1, 1 < x < 2]
    """
    # cover extreme cases first
    if discretevec[0] >= value:
        return(np.zeros(len(discretevec) - 1))
    if discretevec[-1] <= value:
        return(np.ones(len(discretevec) - 1))


    ret = []
    # cover points where value > interval quickly and just set 1
    for i in range(0, len(discretevec) - 1):
        if discretevec[i + 1] <= value:
            ret.append(1)
        else:
            break
    # in between point
    ret.append((value - discretevec[i]) / (discretevec[i + 1] - discretevec[i]))
    # 0 for other points
    ret = ret + [0] * (len(discretevec) - len(ret) - 1)

    return(ret)
            

def intervals_gt_test():
    print(intervals_gt(-0.6, [0, 1, 2]))
    print(intervals_gt(0, [0, 1, 2]))
    print(intervals_gt(0.4, [0, 1, 2]))
    print(intervals_gt(1, [0, 1, 2]))
    print(intervals_gt(1.4, [0, 1, 2]))
    print(intervals_gt(2, [0, 1, 2]))
    print(intervals_gt(2.4, [0, 1, 2]))


# Discrete Stats:{{{1
def discretestats(probs, values):
    """
    Get basic expectation and variance of a finite distribution with certain probabilities.
    """
    import numpy
    expectation = numpy.dot(probs, values)
    variance = numpy.dot(probs, numpy.square([value - expectation for value in values]))
    return(expectation, variance)



# Normal Functions:{{{1
def discretenormaldist(states, mean = 0, sd = 1):
    """
    Get probability weights for the vector states based upon the normal distribution.
    For example:
    f([0]) = [1]
    f([-1, 1]) = [0.5, 0.5]
    f([-1, 0, 1]) = [0.31, 0.38, 0.31]
    """
    import scipy.stats

    # get points in middle of values of a
    states2 = [float('-inf')] + [states[i] + (states[i + 1] - states[i]) / float(2) for i in range(len(states) - 1)] + [float('inf')]
    states2_normalised = [(x - mean) / float(sd) for x in states2]

    # calculate probabilities of each a
    dists = []
    for i in range(0, len(states2_normalised) - 1):
        dists.append(scipy.stats.norm.cdf(states2_normalised[i + 1]) - scipy.stats.norm.cdf(states2_normalised[i]))

    return(dists)


def test_discretenormaldist():
    states = [-1, 1]

    print('equidistant')
    print(discretenormaldist(states, mean = 0, sd = 1))

    print('towards right, high sd')
    print(discretenormaldist(states, mean = 1, sd = 1))

    print('towards right, low sd')
    print(discretenormaldist(states, mean = 0, sd = 0.1))


def getnormalpercentiles(n, mean = 0, sd = 1):
    """
    Specify number of values to return n and then return the points along the cdf of these points.

    Get percentiles spaced out evenly.
    Get cdf of these percentiles.
    """
    import numpy as np
    from scipy.stats import norm

    percentilebounds = np.linspace(0, 1, n + 1)
    percentiles = (percentilebounds[0: n] + percentilebounds[1: n + 1])/2

    values = norm.ppf(percentiles)
    values = values * sd + mean

    return(values)

def test_getnormalpercentiles():
    # this gives 0.025, 0.075, ..., 0.925, 0.975 percentiles
    print(getnormalpercentiles(20, mean = 0, sd = 1))


def markov_normal(values, mean = 0, rho = 0, sd = 1):
    """
    Markov transmission matrix for A = rho * A_{t - 1} + (1 - rho) * mean + \epsilon_t
    where \epsilon_t \sim N(0, sd^2)
    B < 1 (otherwise not a Markov matrix)

    Input Normal values together with mean and 
    """
    Nstates = len(values)

    markov = np.empty([Nstates, Nstates])

    for i in range(Nstates):
        currentstate = values[i]
        markov[i, :] = discretenormaldist(values, mean = rho * currentstate + (1 - rho) * mean, sd = sd)

    return(markov)


def markov_normal_test():
    values = [-1, 0, 1]
    print(markov_normal(values, mean = 0, rho = 0, sd = 1))


def values_markov_normal(Nstates, mean = 0, rho = 0, sdshock = None, sdvar = None):
    """
    Markov transmission matrix for A = rho * A_{t - 1} + (1 - rho) * mean + \epsilon_t
    Where epsilon_t \sim N(0, sd^2)

    """
    if sdshock is not None and sdvar is not None:
        raise ValueError('Should not specify both sd and sdvar')
    # sdvar is sd of variable, not shock
    if sdvar is not None:
        sdshock = (sdvar ** 2 * (1 - rho ** 2)) ** 0.5
    if sdshock is None:
        sdshock = 1
    if sdvar is None:
        sdvar = (sdshock ** 2 / (1 - rho ** 2)) ** 0.5

    # get percentiles to use - based upon Normal with rho = 0 but with same variance as variable
    # should really get normal percentiles including rho
    values = getnormalpercentiles(Nstates, mean = mean, sd = sdvar)
    markov = markov_normal(values, mean = 0, rho = rho, sd = sdshock)

    return(values, markov)


def values_markov_normal_test():
    Nstates = 5
    mean = 0.1
    rho = 0.5
    sdshock = 0.01
    sdvar = (sdshock ** 2 / (1 - rho ** 2)) ** 0.5
    print(values_markov_normal(Nstates, mean = mean, rho = rho, sdshock = sdshock, sdvar = None))
    print(values_markov_normal(Nstates, mean = mean, rho = rho, sdshock = None, sdvar = sdvar))