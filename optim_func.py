#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')


from scipy.optimize import brentq
import numpy as np

# Solve Equation:{{{1
def brentq_onesideerror(f1, lb, ub, maxit = 1000, precision = 1e-11):
    """
    One side of this can return an error.
    Split in two until get only one side that returns error.
    Both sides cannot return error because then I would have to split many sides in two until I find an area that works and it would be a more complicated function.
    """
    try:
        lv = f1(lb)
    except Exception:
        lv = None
    try:
        uv = f1(ub)
    except Exception:
        uv = None

    if lv is None and uv is None:
        print('Both lb and ub return an exception. ub:' + str(ub) + '. lb: ' + str(lb))
        print('Running function again on lb to show what error was.')
        f1(lb)

    # would be weird if the values equal exactly 0
    if lv == 0:
        raise ValueError('lv == 0')
    if uv == 0:
        raise ValueError('uv == 0')



    if lv is None or uv is None:
        # find whether I need the side that failed to be positive or negative
        if lv is None:
            if uv < 0:
                needmediumgt0 = True
            else:
                needmediumgt0 = False
        else:
            if lv < 0:
                needmediumgt0 = True
            else:
                needmediumgt0 = False
            
        # iterate until I find two numbers for which the function does not fail and returns either side of 0
        i = 0
        while i < maxit:
            medium = 0.5 * (lb + ub)
            try:
                mediumv = f1(medium)
            except Exception:
                mediumv = None
            # if value not exist, set lb as value tried
            if mediumv is None:
                # still getting a failed attempt so reduce size of range considered and try again
                if lv is None:
                    lb = medium
                else:
                    ub = medium
            elif mediumv > 0 and needmediumgt0 is True or mediumv < 0 and needmediumgt0 is False:
                # mediumv is a value we need
                # set failed side to be medium and break
                if lv is None:
                    lb = medium
                else:
                    ub = medium
                break
            else:
                # function worked for medium
                # but returned a value on the same side as the side that initially worked
                # set new bound on side that worked to be the medium value and repeat
                if lv is None:
                    ub = medium
                else:
                    lb = medium

            i += 1

            if ub - lb < precision:
                if lv is None:
                    print('Upper bound which worked from start value converged to: ' + str(ub) + '.')
                    print('f(ub) = ' + str(f1(ub)))
                if uv is None:
                    print('Lower bound which worked from start value converged to: ' + str(lb) + '.')
                    print('f(lb) = ' + str(f1(lb)))
                raise ValueError('No solution exists in the range of values given.')

        if i == maxit:
            raise ValueError('Reached maximum iteration. Brentq on one side failed.')








    # adjust_lb = lb
    # adjust_ub = ub
    #
    # if lv is None:
    #     for i in range(maxit): 
    #         medium = 0.5 * (adjust_lb + adjust_ub)
    #         try:
    #             mediumv = f1(medium)
    #         except Exception:
    #             mediumv = None
    #         # if value not exist, set adjust_lb as value tried
    #         if mediumv is None:
    #             adjust_lb = medium
    #         elif (mediumv > 0 and uv > 0) or (mediumv < 0 and uv < 0):
    #             adjust_ub = medium
    #         else:
    #             lb = medium
    #             break
    #
    #     if i == maxit - 1:
    #         raise ValueError('Reached maximum iteration.')
    #
    # if uv is None:
    #     for i in range(maxit): 
    #         medium = 0.5 * (adjust_lb + adjust_ub)
    #         try:
    #             mediumv = f1(medium)
    #         except Exception:
    #             mediumv = None
    #         # if value not exist, set adjust_ub as value tried
    #         if mediumv is None:
    #             adjust_ub = medium
    #         elif (mediumv > 0 and uv > 0) or (mediumv < 0 and lv < 0):
    #             adjust_ub = medium
    #         else:
    #             ub = medium
    #             break
    #
    #     if i == maxit - 1:
    #         raise ValueError('Reached maximum iteration.')
    
    sol = brentq(f1, lb, ub, xtol = precision)

    return(sol)


def brentq_widebounderrors(f1, lb, ub, linspace = None, numlinspace = 21, loglinspace = False, strictlyincreasing = False, strictlydecreasing = False, increasing = False, decreasing = False, printdetails = False, alwaysuserange = False):
    """
    Relevant if:
    - want to do brentq on bounds when have curve that goes down then up or up then down
    - f1 sometimes returns errors at high or low values
    .
    If is always strictly strictlyincreasing or always strictly strictlydecreasing and does not return an error, can just do this with brentq in a try/except. WHAT DOES THIS MEAN???

    Often I want to solve functions that yield errors for high or low values. If so, I'll search within a linspace to try to find a better range over to use brentq.

    increasing means solve for the crossing point when the function is increasing. strictlyincreasing means return an error if the function decreases at any point.
    """
    # if it is True then we always consider a range of values so even if lb < ub we still want to do range
    if alwaysuserange is not True:
        try:
            lb_val = f1(lb)
        except Exception:
            lb_val = None
        try:
            ub_val = f1(ub)
        except Exception:
            ub_val = None

    if alwaysuserange is True or lb_val is None or ub_val is None or (lb_val < 0 and ub_val < 0) or (lb_val > 0 and ub_val > 0) or (lb_val < 0 and ub_val > 0 and decreasing is True) or (lb_val > 0 and ub_val < 0 and increasing is True):
        if strictlyincreasing is True or strictlydecreasing is True:
            if (lb_val > 0 and ub_val > 0) or (lb_val < 0 and ub_val < 0):
                # print('lb_val: ' + str(lb_val))
                # print('ub_val: ' + str(ub_val))
                raise ValueError('Both lb_val and ub_val have same value when should be strictlyincreasing/strictlydecreasing.')

        
        if strictlyincreasing is True:
            increasing is True
        if strictlydecreasing is True:
            decreasing is True

        # parse different potential values
        if linspace is None:
            if loglinspace is False:
                xtry = np.linspace(lb, ub, numlinspace)
            else:
                xtry = np.exp(np.linspace(np.log(lb), np.log(ub), numlinspace))
        xval = []
        yval = []
        for x in xtry:
            try:
                yval.append(f1(x))
                xval.append(x)
            except Exception:
                None

        if len(xval) == 0:
            raise ValueError('No solutions.')

        if printdetails is True:
            print('xval:')
            print(xval)
            print('yval:')
            print(yval)
        sol = []
        for i in range(len(yval) - 1):
            if strictlyincreasing is True:
                if yval[i + 1] < yval[i]:
                    raise ValueError('yval are not strictlyincreasing as expected')
            if strictlydecreasing is True:
                if yval[i + 1] > yval[i]:
                    raise ValueError('yval are not strictlydecreasing as expected')
            if increasing is True:
                if yval[i + 1] > 0 and yval[i] < 0:
                    sol.append(i)
            elif decreasing is True:
                if yval[i + 1] < 0 and yval[i] > 0:
                    sol.append(i)
            else:
                if (yval[i + 1] > 0 and yval[i] < 0) or (yval[i + 1] < 0 and yval[i] > 0):
                    sol.append(i)
        if printdetails is True:
            print('sol:')
            print(sol)
        if len(sol) == 1:
            lb = xval[sol[0]]
            ub = xval[sol[0] + 1]
        else:
            raise ValueError('Wrong number of solutions. Number of solutions: ' + str(len(sol)))
    else:
        if printdetails is True:
            print('lb and ub have opposite sign.')

    sol = brentq(f1, lb, ub)

    return(sol)


def brentq_onesideerror_test_aux(x):
    if x > -1 and x < 1:
        return(x)
    else:
        raise ValueError('error')


def brentq_onesideerror_test():
    brentq_onesideerror(brentq_onesideerror_test_aux, -2, 0.5)


# Minimize Quasiconvex:{{{1
def solvequasiconvex(f1, lowval, highval, crit = 1e-8, printiterations = False):
    """
    Solve for the minimum of a strictly quasiconvex function
    Works by dividing the current interval into 2 and then considering two points in the center
    If the the first point is lower than the second, the solution must be less than the second so we can reduce the interval.
    If the second point is lower than the first, the solution must be less than the first so we can reduce the interval.
    Keep repeating until the interval is smaller than crit. The solution is then the middle of the interval.
    """

    i = 0
    while True:
        i += 1
        midpointlow = (lowval + highval) / 2 - crit / 10
        midpointhigh = (lowval + highval) / 2 + crit / 10

        lowf = f1(midpointlow)
        highf = f1(midpointhigh)

        if lowf < highf:
            highval = midpointhigh
        elif lowf > highf:
            lowval = midpointlow
        else:
            lowval = midpointlow
            highval = midpointhigh

        if printiterations is True:
            print('Iteration ' + str(i) + ': lowval = ' + str(lowval) + '. highval = ' + str(highval) + '.')

        if highval - lowval < crit:
            return(0.5 * (highval + lowval))


def solvequasiconvex_test_aux(x):
    return(x ** 2)


def solvequasiconvex_test():
    solvequasiconvex(solvequasiconvex_test_aux, -1, 2.5, printiterations = True)


