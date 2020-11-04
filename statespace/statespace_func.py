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

# Getting data:{{{1
def statespace_gendata(A, B, C, D, v, X0 = None):
    """
    Input a matrix of shocks and iterate from starting value.
    If no start value specified, use zero.

    Input shocks as matrix with first dimension time.
    """

    import numpy as np
    
    # get parameters for model
    dim = C.shape
    numcontrols = dim[0]
    numstates = dim[1]
    dim = v.shape
    T = dim[0]
    numshocks = dim[1]

    X = np.empty([T, numstates])
    Y = np.empty([T, numcontrols])

    if X0 is None:
        X0 = np.zeros([numstates])

    for t in range(T):
        if t == 0:
            X[t] = X0
        else:
            X[t] = np.reshape(np.dot(A, X[t - 1]) + np.dot(B, v[t, :]), [numstates])

        Y[t] = np.reshape(np.dot(C, X[t]) + np.dot(D, v[t, :]), [numcontrols])

    return(X, Y)


def statespace_simdata(A, B, C, D, T, Omega = None, X0 = None):
    import numpy as np

    if Omega is None:
        Omega = np.identity(B.shape[1])

    # v = np.random.normal(size = (T, B.shape[1]))
    v = np.random.multivariate_normal(np.zeros(B.shape[1]), Omega, size = (T))

    X, Y = importattr(__projectdir__ / Path('statespace/statespace_func.py'), 'statespace_gendata')(A, B, C, D, v, X0 = X0)

    return(X, Y, v)


# Kalman Filter:{{{1
def solvevariance_quick(A, B, Omega = None, maxiter = None, crit = None):
    """
    For large matrices, computing the variance matrix directly does not work well since it requires taking the inverse of a large matrix.
    An alternative is to iterate to find a guess for the variance.

    If both maxiter and crit is None, just do crit = 1e-10 and maxiter = 100
    """
    numstates = A.shape[0]
    numshocks = B.shape[1]

    variance = np.eye(numstates)
    oldvariance = variance
    if Omega is None:
        Omega = np.eye(numshocks)

    if crit is None:
        # with crit = 1e-10, I had some issues sometimes with getting convergence (presumably due to precision of multiplication)
        crit = 1e-8

    i = 0
    while True:
        variance = A.dot(variance).dot(A.transpose()) + B.dot(Omega).dot(B.transpose())

        i += 1
        if i == maxiter:
            break

        if crit is not None:
            maxval = np.max(np.abs(variance - oldvariance))
            if maxval < crit:
                break

        oldvariance = variance

    return(variance)
        
def solvevariance_quick_test():
    A = np.array([[0.9, 0], [0.5, 0.5]])
    B = np.array([[1], [0.3]])
    variance = solvevariance_quick(A, B, Omega = None, maxiter = 1000, crit = None)       
    return(variance)


def kalmanfilter(y, A, B, C, D, Omega = None, x0 = None, P0 = None):
    """
    x_t_tm1[0] represents x_0|-1
    x_t_t[0] represents x_0|0

    P is the variance of x.
    Q is the variance of y.
    R is the covariance.
    t_tm1 means var at t given info at t-1.
    y has dimension T x numobservedvars
    """
    import numpy as np

    # get basic parameters
    dim = C.shape
    numcontrols = dim[0]
    numstates = dim[1]
    T = len(y)
    numshocks = B.shape[1]

    if Omega is None:
        Omega = np.identity(numshocks)

    x_t_tm1 = np.empty([T, numstates])
    P_t_tm1 = np.empty([T, numstates, numstates])
    x_t_t = np.empty([T, numstates])
    P_t_t = np.empty([T, numstates, numstates])
    y_t_tm1 = np.empty([T, numcontrols])
    Q_t_tm1 = np.empty([T, numcontrols, numcontrols])
    R_t_tm1 = np.empty([T, numcontrols, numstates])

    if x0 is None:
        x0 = np.zeros([numstates])

    if P0 is None:
        P0 = np.reshape(np.linalg.solve((np.identity(numstates ** 2) - np.kron(A, A)),(np.reshape(B.dot(Omega).dot(B.transpose()), numstates ** 2))), [numstates, numstates])

    for t in range(0, T):
        if t == 0:
            x_tm1_tm1 = x0
            P_tm1_tm1 = P0
        else:
            x_tm1_tm1 = x_t_t[t - 1]
            P_tm1_tm1 = P_t_t[t - 1]

        x_t_tm1[t] = A.dot(x_tm1_tm1)
        y_t_tm1[t] = C.dot(x_tm1_tm1)

        P_t_tm1[t] = A.dot(P_tm1_tm1).dot(A.transpose()) + B.dot(Omega).dot(B.transpose())
        Q_t_tm1[t] = C.dot(P_tm1_tm1).dot(C.transpose()) + D.dot(Omega).dot(D.transpose())
        R_t_tm1[t] = C.dot(P_tm1_tm1).dot(A.transpose()) + D.dot(Omega).dot(B.transpose())

        # x_t_t[t] = x_t_tm1[t] + R_t_tm1[t].transpose().dot(np.linalg.inv(Q_t_tm1[t])).dot((y[t] - y_t_tm1[t]))
        # P_t_t[t] = P_t_tm1[t] - R_t_tm1[t].transpose().dot(np.linalg.inv(Q_t_tm1[t])).dot(R_t_tm1[t])


        # need to use solve rather than inv as affords a more precise solution
        # otherwise can get explosive system, especially in the case where P_t_t is small
        x_t_t[t] = x_t_tm1[t] + R_t_tm1[t].transpose().dot(np.linalg.solve(Q_t_tm1[t], (y[t] - y_t_tm1[t])))
        P_t_t_temp = P_t_tm1[t] - R_t_tm1[t].transpose().dot(np.linalg.solve(Q_t_tm1[t], R_t_tm1[t]))

        # P_t_t[t] will not solve perfectly
        # when the signals fully capture the model, you get something like [[1e-20, 1e-19], [-1e-19, 1e-20]]
        # notice the [2,1] and [1,2] elements are not the same
        # these errors can compound over time leading to explosive results
        # to correct for this, we sum P_t_t[t] and its transpose and divide by 2
        P_t_t[t] = (P_t_t_temp + P_t_t_temp.transpose()) / 2

    return(x_t_tm1, P_t_tm1, x_t_t, P_t_t, y_t_tm1, Q_t_tm1, R_t_tm1)


def kalmanfilter_test1():
    numy = 1
    T = 3
    y = np.random.normal(size = [T, numy])
    A = np.array([[0.5]])
    B = np.array([[1]])
    C = np.array([[1]])
    D = np.array([[0.1]])
    x_t_tm1, P_t_tm1, x_t_t, P_t_t, y_t_tm1, Q_t_tm1, R_t_tm1 = kalmanfilter(y, A, B, C, D, P0 = np.array([[0.001]]))


def kalmanfilter_test2():
    A = np.array([[0.9, 0], [0.26339, 0.83800]])
    B = np.array([[0.1], [0.029266]])
    C = np.array([[0.9, 0.3], [0.4758, 0.5250]])
    D = np.array([[0.1], [0.0528684]])

    numy = 1
    T = 1000
    y = np.random.normal(size = [T, C.shape[0]])

    x_t_tm1, P_t_tm1, x_t_t, P_t_t, y_t_tm1, Q_t_tm1, R_t_tm1 = kalmanfilter(y, A, B, C, D)
    for i in range(0, 1000):
        print(i)
        print(P_t_tm1[i])
        print(Q_t_tm1[i])
        print(R_t_tm1[i])
        print(P_t_t[i])


def kalmanfilter_ytfunctionofxt(y, A, B, C, D, Omega = None, x0 = None):
    """
    x_t_tm1[0] represents x_0|-1
    x_t_t[0] represents x_0|0

    P is the variance of x.
    Q is the variance of y.
    R is the covariance.
    t_tm1 means var at t given info at t-1.
    y has dimension T x numobservedvars
    """
    import numpy as np

    # get basic parameters
    dim = C.shape
    numcontrols = dim[0]
    numstates = dim[1]
    T = len(y)
    numshocks = B.shape[1]

    if Omega is None:
        Omega = np.identity(numshocks)

    x_t_tm1 = np.empty([T, numstates])
    P_t_tm1 = np.empty([T, numstates, numstates])
    x_t_t = np.empty([T, numstates])
    P_t_t = np.empty([T, numstates, numstates])
    y_t_tm1 = np.empty([T, numcontrols])
    Q_t_tm1 = np.empty([T, numcontrols, numcontrols])
    R_t_tm1 = np.empty([T, numcontrols, numstates])

    if x0 is None:
        x_t_tm1[0, :] = np.zeros([numstates])

    P_t_tm1[0, :, :] = np.reshape(np.linalg.solve(np.identity(numstates ** 2) - np.kron(A, A), np.reshape(B.dot(Omega).dot(B.transpose()), numstates ** 2)), [numstates, numstates])

    for t in range(0, T):
        y_t_tm1[t, :] = C.dot(x_t_tm1[t, :])
        Q_t_tm1[t, :, :] = C.dot(P_t_tm1[t, :, :]).dot(C.transpose()) + D.dot(Omega).dot(D.transpose())
        R_t_tm1[t, :, :] = C.dot(P_t_tm1[t, :, :]) + D.dot(Omega).dot(B.transpose())

        x_t_t[t, :] = x_t_tm1[t, :] + R_t_tm1[t, :, :].transpose().dot(np.linalg.inv(Q_t_tm1[t, :, :])).dot((y[t, :] - y_t_tm1[t, :]))
        P_t_t[t, :, :] = P_t_tm1[t, :, :] - R_t_tm1[t, :, :].transpose().dot(np.linalg.inv(Q_t_tm1[t, :, :])).dot(R_t_tm1[t, :, :])

        # only do up to x_T_Tm1
        if t < T - 1:
            x_t_tm1[t + 1, :] = A.dot(x_t_t[t, :])
            P_t_tm1[t + 1, :, :] = A.dot(P_t_t[t, :, :]).dot(A.transpose()) + B.dot(Omega).dot(B.transpose())

    return(x_t_tm1, P_t_tm1, x_t_t, P_t_t, y_t_tm1, Q_t_tm1, R_t_tm1)

def logl_prop_kalmanfilter(y, y_t_tm1, Q_t_tm1, addconstant = False):
    """
    _prop since I skip a constant.
    """
    import numpy as np

    logl = 0
    for t in range(len(y)):
        logl = logl + -0.5 * np.log(np.linalg.det(Q_t_tm1[t, :, :])) - 0.5 * (y[t, :] - y_t_tm1[t, :]).transpose().dot(np.linalg.solve(Q_t_tm1[t, :, :], y[t, :] - y_t_tm1[t, :]))

    if addconstant is True:
        # T * -0.5 * n * log(2pi)
        dim = y.shape
        logl = logl + dim[0] * -0.5 * dim[1] * np.log(2*np.pi)

    return(logl)

# Conversion into just States:{{{1
def ABCD_convert(A, B, C, D = None):
    """
    Code to get rid of the distinction between states and controls starting from ABCD form.

    X_t = AX_{t - 1} + BV_t
    Y_t = CX_t + DV_t

    Note that Y_t = CAX_{t-1} + (CB + D)v_t

    Then we can write Z_t' = (X_t' Y_t') and we have that:
    X_t = (A 0) Z_{t - 1} + BV_t
    Y_t = (CA 0) Z_{t - 1} + (CB + D)V_t
    """
    import numpy as np

    dim = A.shape
    numstates = dim[0]
    numshocks = dim[1]

    dim = C.shape
    numcontrols = dim[0]

    if D is None:
        D = np.zeros([numcontrols, numshocks])

    # X_t concatenate
    bigA_X = np.concatenate((A, np.zeros([numstates, numcontrols])), axis = 1)

    # Y_t concatenate
    bigA_Y = np.concatenate((C.dot(A), np.zeros([numcontrols, numcontrols])), axis = 1)

    bigA = np.concatenate((bigA_X, bigA_Y), axis = 0)

    bigB = np.concatenate((B, C.dot(B) + D), axis = 0)

    return(bigA, bigB)

# IRFs:{{{1
def irgraphs(irarray, names = None, subplotwidth = None, pltshow = False, pltsavename = None, Trange = None):
    """
    Plot irfs.
    irarray is T x numberofgraphs
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # adjust defaults
    if subplotwidth is None:
        subplotwidth = 2

    irarrayshape = np.shape(irarray)
    T = irarrayshape[0]
    numgraphs = irarrayshape[1]

    subplotheight = (numgraphs + subplotwidth - 1) // subplotwidth

    if Trange is None:
        Trange = list(range(0, T))

    fig = plt.figure()
    for i in range(0, numgraphs):
        ax = fig.add_subplot(subplotheight, subplotwidth, i + 1)
        if names is not None:
            ax.set_title(names[i])
        ax.plot(Trange, irarray[:, i])

    plt.tight_layout()

    if pltsavename is not None:
        plt.savefig(pltsavename)

    if pltshow is True:
        plt.show()

    plt.close()


def irgraphs_multiplelines(irarray, linenames = None, graphnames = None, subplotwidth = None, pltshow = False, pltsavename = None, Trange = None, graphswithlegend = None, Trangeislist = False):
    """
    Plots IRFS of multiple different runs
    I show multiple graphs and on each graphs there is a line for each model

    irarray is numvarseaachgraph x T x numberofgraphs or a list of T x numberofgraphs numpy arrays
    If Trangeislist then Trange = [[0, ..., 40], [0, ..., 40]] so that I can vary the number of observations for different IRFs
    """
    import matplotlib.pyplot as plt
    import numpy as np

    numlines = len(irarray)
    T = len(irarray[0])
    numgraphs = len(irarray[0][0])

    # adjust defaults
    if subplotwidth is None:
        subplotwidth = 2
    if linenames is None:
        linenames = [None] * numlines

    if graphswithlegend is None:
        # just show legend in first subplot
        graphswithaxis = [0]

    subplotheight = (numgraphs + subplotwidth - 1) // subplotwidth

    if Trangeislist is False:
        if Trange is None:
            Trange2 = list(range(T))
        else:
            Trange2 = Trange

    fig = plt.figure()
    for numgraph in range(numgraphs):
        ax = fig.add_subplot(subplotheight, subplotwidth, numgraph + 1)
        if graphnames is not None:
            ax.set_title(graphnames[numgraph])
        for numline in range(numlines):
            # adjust Trange if necessary
            if Trangeislist is True:
                Trange2 = Trange[numline]
            # plot the actual graph
            ax.plot(Trange2, irarray[numline][:, numgraph], label = linenames[numline])

        if numgraph in graphswithaxis:
            ax.legend()

    plt.tight_layout()

    if pltsavename is not None:
        plt.savefig(pltsavename)
    else:
        plt.show()

    plt.close()


# Variance Decomposition:{{{1
def vardecomp(A, B, Omega = None, T = 40, longrun = False, maxiterations = 1000, convergencecrit = 1e-6):
    """
    Form: X_t = AX_{t - 1} + BV_t
    Find the Var_t(X_{t + i}).
    I can also find in the process Var(X_t) by taking large enough T such that Var_t(X_{t + T}) Var_t(X_{t + T + 1}). This might be better than finding it by using (I - A^{-1})X_t = V_t
    """
    import numpy as np

    dim = B.shape
    numstates = dim[0]
    numshocks = dim[1]

    if Omega is None:
        Omega = np.identity(numshocks)

    Var_byshock_bytime = np.empty([numshocks, T + 1, numstates, numstates])
    Var_bytime = np.empty([T + 1, numstates, numstates])
    Var_byshock_lt = np.empty([numshocks, numstates, numstates])
    Var_lt = np.empty([numstates, numstates])

    if longrun is True:
        maxiterations = maxiterations
    else:
        maxiterations = T + 1

    # get Omega by shocks
    Omega_byshock = np.zeros([numshocks, numshocks, numshocks])
    for i in range(0, numshocks):
        Omega_byshock[i, i, i] = Omega[i, i]

    # initialise shocks
    Var_lt = B.dot(Omega).dot(B.transpose())
    Var_lt_previous = Var_lt
    Var_bytime[0, :, :] = Var_lt
    for i in range(0, numshocks):
        Var_byshock_lt[i, :, :] = B.dot(Omega_byshock[i, :, :]).dot(B.transpose())
        Var_byshock_bytime[i, 0, :, :] = Var_byshock_lt[i, :, :]

    # now do for the rest of time
    for t in range(1, maxiterations):
        # by shock
        for i in range(0, numshocks):
            Var_byshock_lt[i, :, :] = A.dot(Var_byshock_lt[i, :, :]).dot(A.transpose()) + Var_byshock_bytime[i, 0, :, :]

            if t <= T:
                Var_byshock_bytime[i, t, :, :] = Var_byshock_lt[i, :, :]

        # overall variance
        Var_lt = A.dot(Var_lt).dot(A.transpose()) + Var_bytime[0, :, :]
        if t <= T:
            Var_bytime[t, :, :] = Var_lt

        # break if difference between long-term overall variance is small enough
        if np.max(np.abs(Var_lt - Var_lt_previous)) < convergencecrit and t >= T:
            break
        Var_lt_previous = Var_lt

        # if fail to converge
        if t == maxiterations - 1 and longrun is True:
            print('Variance failed to converge in the long-term')
            sys.exit(1)

    if longrun is False:
        return(Var_byshock_bytime, Var_bytime)
    else:
        return(Var_byshock_bytime, Var_bytime, Var_byshock_lt, Var_lt)



        



def vardecomptable(Var_byshock_bytime, Var_bytime, Var_byshock_lt = None, Var_lt = None):
    """
    Convert variance decomposition into standard tables.
    """
    import numpy as np

    if Var_byshock_lt is not None and Var_lt is not None:
        addlt = True
    else:
        addlt = False

    dim = Var_byshock_bytime.shape
    numshocks = dim[0]
    numtime = dim[1]
    numvars = dim[2]

    if addlt is True:
        numtimecol = numtime + 1
    else:
        numtimecol = numtime

    table = np.empty([numvars, numtimecol, numshocks])

    for i in range(0, numvars):
        for s in range(0, numshocks):
            for t in range(0, numtime):
                table[i, t, s] = Var_byshock_bytime[s, t, i, i] / Var_bytime[t, i, i]
            
            if addlt is True:
                table[i, numtimecol - 1, s] = Var_byshock_lt[s, i, i] / Var_lt[i, i]

    return(table)


