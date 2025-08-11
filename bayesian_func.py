#!/usr/bin/env python3
""" Introduction: {{{
Basic functions for doing Bayesian analysis.
Have basic metropolis function over bounds. Problem: with a lot of parameters, hit boundaries frequently.
Also have boundedrandommle which just looks for general maximum randomly.

SGU News Shocks proceeds by first finding good maximum point using random draws, setting scale parameter so get lots of draws and then doing metropolis. I capture the first and third elements here with boundedrandommle and metropolis_bounds_do.

LATER:
Want to add better metrolis function with bounds which allows me to do random search just over bounded area whilst still getting distribution at the same time.

""" # End Introduction }}}
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import datetime
import decimal
import functools
import multiprocessing
import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.special import gamma
from scipy.stats import norm
from scipy.stats import uniform
import shutil

# Imported Functions:{{{1
sys.path.append(str(__projectdir__ / Path('submodules/python-general-func/')))
from print_func import printorwrite
printorwrite = printorwrite

# Convert from Mean, Variance to Prior Parameters:{{{1
def beta_prior_conversion_aux(mean, variance, BETA):
    """
    Function that I can solve to get BETA in the beta distribution prior conversion function
    """
    ret = mean / (1 - mean) / ( (mean / (1 - mean) + 1)**2 * (mean / (1 - mean) * BETA + BETA + 1) ) - variance
    return(ret)


def beta_prior_conversion(mean, sd):
    variance = sd ** 2

    f1 = functools.partial(beta_prior_conversion_aux, mean, variance)

    # need to set upper bound to be large but not so large that we don't get convergence
    BETA = brentq(f1, 0, 1e10)

    ALPHA = BETA * mean / (1 - mean)

    return(ALPHA, BETA)


def getprior_conversion_single(priortype, mean, sd):
    """
    Convert from specifying the mean and standard deviation of a distribution to specifying the parameters of a distribution.
    """
    if priortype == 'normal':
        return([mean, sd])
    elif priortype == 'uniform':
        BETA = 3 ** 0.5 * sd + mean
        ALPHA = 2 * mean - BETA
        return(ALPHA, BETA)
    elif priortype == 'beta':
        ALPHA, BETA = beta_prior_conversion(mean, sd)
        return(ALPHA, BETA)
    elif priortype == 'gamma':
        BETA = mean / sd ** 2
        ALPHA = mean * BETA
        return(ALPHA, BETA)
    elif priortype == 'invgamma':
        ALPHA = mean ** 2 / sd ** 2 + 2
        BETA = mean * (ALPHA - 1)
        return(ALPHA, BETA)
    else:
        raise ValueError('Misspecified priortype.')
        

# Prior Details:{{{1
def getprior_mean_single(priortype, parameters):
    """
    Get the mean of the distribution
    I can also use this as the start value
    """
    if priortype == 'normal':
        return(parameters[0])
    elif priortype == 'uniform':
        return((parameters[0] + parameters[1]) / 2)
    elif priortype == 'beta':
        # mean of beta distribution is 1 / (1 + BETA/ALPHA)
        return(1 / (1 + parameters[1] / parameters[0]))
    elif priortype == 'gamma':
        # mean of gamma distribution is ALPHA / BETA
        return(parameters[0] / parameters[1])
    elif priortype == 'invgamma':
        return(parameters[1] / (parameters[0] - 1))
    else:
        raise ValueError('Misspecified priortype.')
        

def getprior_std_single(priortype, parameters):
    """
    Get the mean of the distribution
    I can also use this as the start value
    """
    if priortype == 'normal':
        return(parameters[1])
    elif priortype == 'uniform':
        variance = 1/12 * (parameters[1] - parameters[0]) ** 2
        return(variance ** 0.5)
    elif priortype == 'beta':
        a = parameters[0]
        b = parameters[1]
        variance = a * b / ((a + b)**2 * (a + b + 1))
        return(variance ** 0.5)
    elif priortype == 'gamma':
        return((parameters[0] / parameters[1] ** 2) ** 0.5)
    elif priortype == 'invgamma':
        ALPHA = parameters[0]
        BETA = parameters[1]
        if ALPHA < 2:
            return(None)
        variance = BETA ** 2 / ( (ALPHA - 1)**2 * (ALPHA - 2) )
        return(variance ** 0.5)
    else:
        raise ValueError('Misspecified priortype.')
        

def getprior_lb_single(priortype, parameters):
    if priortype == 'normal':
        return(None)
    elif priortype == 'uniform':
        return(parameters[0])
    elif priortype == 'beta':
        return(0)
    elif priortype == 'gamma':
        return(0)
    elif priortype == 'invgamma':
        return(0)
    else:
        raise ValueError('Misspecified priortype.')
        

def getprior_ub_single(priortype, parameters):
    if priortype == 'normal':
        return(None)
    elif priortype == 'uniform':
        return(parameters[1])
    elif priortype == 'beta':
        return(1)
    elif priortype == 'gamma':
        return(None)
    elif priortype == 'invgamma':
        return(None)
    else:
        raise ValueError('Misspecified priortype.')


# Prior Density Function:{{{1
def getprior_density_single(priortype, parameters, value):
    if priortype == 'normal':
        # parameters = [mean, std]
        return(scipy.stats.norm.pdf(value, parameters[0], parameters[1]))
    elif priortype == 'uniform':
        # parameters = [lb, ub]
        return(scipy.stats.uniform.pdf(value, parameters[0], parameters[1]))
    elif priortype == 'beta':
        # parameters = [a, b]
        return(scipy.stats.beta.pdf(value, parameters[0], parameters[1]))
    elif priortype == 'gamma':
        # parameters = [shape, inversescale]
        shape = parameters[0]
        inversescale = parameters[1]

        return(inversescale ** shape * value ** (shape - 1) * np.exp(-inversescale * value) / scipy.special.gamma(shape))
    elif priortype == 'invgamma':
        shape = parameters[0]
        scale = parameters[1]
        return(scale ** shape / scipy.special.gamma(shape) * value ** (-shape - 1) * np.exp(-scale / value))
    else:
        raise ValueError('Misspecified priortype.')


# Prior List functions:{{{1
def getpriorlist_convert(priorlist_meansd):
    """
    Convert a priorlist of the form [[priortype, mean, sd] into a priorlist of the form [[priortype, parameterlist]] where parameterlist corresponds to the parameters in getprior_density_single
    """
    priorlist_parameters = []
    for i in range(len(priorlist_meansd)):
        parameters = getprior_conversion_single(priorlist_meansd[i][0], priorlist_meansd[i][1], priorlist_meansd[i][2])
        priorlist_parameters.append([priorlist_meansd[i][0], parameters])
    return(priorlist_parameters)


def getpriorlist_mean(priorlist):
    """
    Note that the priorlist should be formed of parameters not meansd
    """
    means = []
    for i in range(len(priorlist)):
        means.append(getprior_mean_single(priorlist[i][0], priorlist[i][1]))

    means = np.array(means)
    return(means)


def getpriorlist_sd(priorlist):
    """
    Note that the priorlist should be formed of parameters not meansd
    """
    sds = []
    for i in range(len(priorlist)):
        sds.append(getprior_std_single(priorlist[i][0], priorlist[i][1]))

    sds = np.array(sds)
    return(sds)


def getpriorlist_lb(priorlist):
    """
    Note that the priorlist should be formed of parameters not meansd
    """
    lbs = []
    for i in range(len(priorlist)):
        lbs.append(getprior_lb_single(priorlist[i][0], priorlist[i][1]))

    return(lbs)


def getpriorlist_ub(priorlist):
    """
    Note that the priorlist should be formed of parameters not meansd
    """
    ubs = []
    for i in range(len(priorlist)):
        ubs.append(getprior_ub_single(priorlist[i][0], priorlist[i][1]))

    return(ubs)


def getpriorlistdetails_parameters(priorlist):
    """
    Input a priorlist i.e. [['uniform', [0, 1]], ['normal', [-1, 2]]] and back out details about the priorlist
    Note that the parameters of the priorlist should be the parameters corresponding to the getprior_density_single rather than the mean variance i.e. ['uniform', [0, 1]] means a uniform with lb of 0 and ub of 1 rather than a uniform with mean 0 and standard deviation 1
    """
    means = getpriorlist_mean(priorlist)
    sds = getpriorlist_sd(priorlist)
    lbs = getpriorlist_lb(priorlist)
    ubs = getpriorlist_ub(priorlist)

    return(means, sds, lbs, ubs)


def getpriordensityfunc_aux(priorlist, x):
    """
    x is a vector of the values that each distribution takes
    priorlist takes the form given in getpriorlistdetails_parameters

    I can then get a function for the prior density using functools.partial()
    """
    ret = 1
    for i in range(len(priorlist)):
        ret = ret * getprior_density_single(priorlist[i][0], priorlist[i][1], x[i])
    return(ret)


# Prior Checks:{{{1
def checkpriors():
    """
    I get the parameters corresponding to mean, variance from my conversion functions.
    I then back out the mean, variance from the parameters
    They should be approximately the same
    """
    def checkone(priortype, mean, sd):
        parameters = getprior_conversion_single(priortype, mean, sd)

        mean2 = getprior_mean_single(priortype, parameters)
        sd2 = getprior_std_single(priortype, parameters)

        if abs(mean - mean2) > 1e-4:
            print('Mean failed for ' + priortype)
            print(mean)
            print(mean2)
        if abs(sd - sd2) > 1e-4:
            print('Variance failed for ' + priortype)
            print(sd)
            print(sd2)

    checkone('uniform', -1, 0.5)
    checkone('beta', 0.5, 0.05)
    checkone('gamma', 2, 0.5)
    checkone('invgamma', 2, 0.5)


# Metropolis Hastings Basic:{{{1
def posterioriteration(posteriorfunc, newvalues, currentposterior, logposterior = True):
    newposterior = posteriorfunc(newvalues)
    if logposterior is False:
        replace = newposterior / currentposterior >= np.random.uniform()
    else:
        replace = newposterior - currentposterior >= np.log(np.random.uniform())
    # for some reason this doesn't work like a normal true value
    # so convert to boolean
    # otherwise replace is True can return False even when True
    replace = bool(replace)

    return(replace, newposterior)


def metropolis_hastings(posteriorfunc, scalelist, startvallist, numiterations, lowerboundlist = None, upperboundlist = None, savefile = None, deletefile = False, printdetails = False, logposterior = False, raiseerror = True, logfile = None):
    """
    savefile is a csv file where I record the distribution.
    If logposterior is True then the posteriorfunc = log(posterior). (Of course, it's normally possible to just convert to the exponential but if the loglikelihood is very high then this may not be possible (as we get infinity).

    savefile is where I save what I print out

    """

    # adjust basic lists
    if lowerboundlist is None:
        lowerboundlist = [None] * len(startvallist)
    if upperboundlist is None:
        upperboundlist = [None] * len(startvallist)
    startvallist = np.array(startvallist)

    numunknowns = len(lowerboundlist)
    
    currentresult = startvallist
    # set currentposterior so initial guess is accepted
    currentposterior = -np.inf
    # if continuing from previous run, get last values from the file if possible
    if savefile is not None:
        if deletefile is True:
            if os.path.isfile(savefile):
                os.remove(savefile)

    # replace logfile if exists
    # note this will continue from the prior file
    printorwrite('', logfile, printmessage = printdetails)

    # now do actual iteration
    lastsave = 0
    unsavedresults = []
    for i in range(0, int(numiterations)):

        # get this iteration values for parameters
        newvalues = []
        for j in range(0, len(lowerboundlist)):
            # find this iteration value for a single parameter
            while True:
                newvalue = currentresult[j] + np.random.normal() * scalelist[j]
                # keep going until find new parameter value that satisfies bounds
                if (lowerboundlist[j] is None or newvalue > lowerboundlist[j]) and (upperboundlist[j] is None or newvalue < upperboundlist[j]):
                    break
            newvalues.append(newvalue)

        # print out details on iteration
        printorwrite('\nIteration ' + str(i + 1) + '.', logfile, printmessage = printdetails)
        printorwrite(str(datetime.datetime.now()), logfile, printmessage = printdetails)
        printorwrite('Attempted values:', logfile, printmessage = printdetails)
        printorwrite(newvalues, logfile, printmessage = printdetails)


        posteriorfailed = False
        try:
            # replace is boolean for whether make replacement or not
            replace, newposterior = posterioriteration(posteriorfunc, newvalues, currentposterior, logposterior = logposterior)
        except Exception:
            # allow possibility of not returning an error
            if raiseerror is True:
                posterioriteration(posteriorfunc, newvalues, currentposterior, logposterior = logposterior)
            else:
                posteriorfailed = True
                replace = False

        if replace is True:
            currentresult = newvalues
            currentposterior = newposterior

        # print details
        if posteriorfailed is True:
            printorwrite('Posterior failed', logfile, printmessage = printdetails)
        else:
            printorwrite('New posterior value:', logfile, printmessage = printdetails)
            printorwrite(newposterior, logfile, printmessage = printdetails)
        printorwrite('Current Parameters:', logfile, printmessage = printdetails)
        printorwrite(currentresult, logfile, printmessage = printdetails)
        printorwrite('Current Posterior:', logfile, printmessage = printdetails)
        printorwrite(currentposterior, logfile, printmessage = printdetails)
        printorwrite('Replace:', logfile, printmessage = printdetails)
        printorwrite(replace, logfile, printmessage = printdetails)
        
        # save results
        unsavedresults.append(currentresult)
        if savefile is not None and i == numiterations - 1:
            with open(savefile, 'a+') as f:
                # need to ensure add additional \n at end
                f.write('\n'.join([','.join([str(element) for element in line]) for line in unsavedresults]) + '\n')
            unsavedresults = []
            printorwrite('Saved up to iteration ' + str(i) + '.', logfile, printmessage = printdetails)
    return(unsavedresults)


def distfromfile(filename, burnindelete = None):
    """
    If burnin not specified, take minimum of 1000 and 0.5*len smallest file
    """

    with open(filename) as f:
        data = [[float(num) for num in line.split(',')] for line in f.read().splitlines()]

    if burnindelete is not None:
        data = data[burnindelete: ]

    data = np.array(data)

    return(data)


# Metropolis Hastings Pool:{{{1
def metropolis_hastings_pool_aux(posteriorfunc, scalelist, startvallist, numiterations, lowerboundlist, upperboundlist, deletefile, printdetails, logposterior, raiseerror, savefile):
    """
    Auxilliary function that I use to get the function I apply the mapping to when using multiple processes.
    """
    metropolis_hastings(posteriorfunc, scalelist, startvallist, numiterations, lowerboundlist = lowerboundlist, upperboundlist = upperboundlist, savefile = savefile, deletefile = deletefile, printdetails = printdetails, logposterior = logposterior, raiseerror = raiseerror)


def metropolis_hastings_pool(savefolder, posteriorfunc, scalelist, startvallist, numiterations, lowerboundlist = None, upperboundlist = None, deletefile = True, printdetails = False, logposterior = False, raiseerror = True, numprocesses = None):
    """
    Run Metropolis Hastings in multiple processes
    """
    if os.path.isdir(savefolder) and deletefile is True:
        shutil.rmtree(savefolder)
    if not os.path.isdir(savefolder):
        os.mkdir(savefolder)

    if numprocesses is None:
        numprocesses = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(numprocesses)

    savefiles = []
    for i in range(numprocesses):
        savefiles.append(os.path.join(savefolder, str(i) + '.csv'))

    f1 = functools.partial(metropolis_hastings_pool_aux, posteriorfunc, scalelist, startvallist, numiterations, lowerboundlist, upperboundlist, deletefile, printdetails, logposterior, raiseerror)

    pool.map(f1, savefiles)


def distfromfolder(folder, burnindelete = None, concatenate = True):
    """
    If burnin not specified, take minimum of 1000 and 0.5*len smallest file
    """

    filenames = os.listdir(folder)
    datalist = []
    for filename in filenames:
        datalist.append(distfromfile(os.path.join(folder, filename), burnindelete = burnindelete))

    if concatenate is True:
        data = np.concatenate(datalist, axis = 0)
        return(data)
    else:
        return(datalist)


