#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

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


def weightvalues_inputlist(inputlist, discretevec):
    """
    Run weightvaluediscretevec many times on an inputlist (rather than just for a single value)
    Inputlist does not need to be ordered
    Note this could be more efficient if I used an ordered input list and adjusted the function

    Returnlist is an array
    Element i,j corresponds to weight of j in discretevec that is used in constructing value i in inputlist
    """
    returnlists = []
    for value in inputlist:
        returnlists.append( weightvaluediscretevec(value, discretevec) )

    return(np.array(returnlists))


def weightvalues_inputlist_test():
    discretevec = [0, 1, 2, 3, 4]
    print(weightvalues_inputlist([-1, 1, 1.5, 4, 5], discretevec))


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

    So I compute P(val > points in interval [0, 1]) and P(val > points in interval [1, 2])
    If val < the lower bound of the interval then the probability is zero
    If val > the upper bound of the interval then the probability is 1
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
    Markov transmission matrix for A = rho * A_{t - 1} + (1 - rho) * mean + \\epsilon_t
    where \\epsilon_t \\sim N(0, sd^2)
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
    Markov transmission matrix for A = rho * A_{t - 1} + (1 - rho) * mean + \\epsilon_t
    Where epsilon_t \\sim N(0, sd^2)

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
