#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/../')

# gendata:{{{1
def simplestatespace():
    import numpy as np

    A = np.array([[0.5, 0.3], [0.2, 0.1]])
    B = np.array([[1, 0], [0, 0.9]])
    C = np.array([[0.1, 0.3]])
    D = np.array([[0, 0]])

    Omega = np.array([[1, 0], [0, 0.5]])

    return(A, B, C, D, Omega)


def test_gendata1():
    import numpy as np

    A, B, C, D, Omega = simplestatespace()

    v = np.zeros((100, 2))

    X0 = np.array([[1, 0]])

    from statespace_func import statespace_gendata
    X, Y = statespace_gendata(A, B, C, D, v, X0 = X0)
    print(X)
    print(Y)


def test_gendata2():
    import numpy as np

    A, B, C, D, Omega = simplestatespace()

    v = np.random.normal(size = (100, 2))

    from statespace_func import statespace_gendata
    X, Y = statespace_gendata(A, B, C, D, v)
    print(X)
    print(Y)

# simdata:{{{1
def test_simdata():
    A, B, C, D, Omega = simplestatespace()

    from statespace_func import statespace_simdata
    X, Y, v = statespace_simdata(A, B, C, D, 40, Omega)

    print(Y)

# kalmanfilter:{{{1
def test_kalmanfilter():
    A, B, C, D, Omega = simplestatespace()

    from statespace_func import statespace_simdata
    x, y, v = statespace_simdata(A, B, C, D, 40)

    from statespace_func import kalmanfilter
    x_t_tm1, P_t_tm1, x_t_t, P_t_t, y_t_tm1, Q_t_tm1, R_t_tm1 = kalmanfilter(y, A, B, C, D, Omega = Omega)

    print(x_t_tm1)
    print(x_t_t)
    print(P_t_t)

def test2_kalmanfilter():
    import numpy as np

    y = np.array([[i] for i in [0, 2, -2, 0, 1, -1, 0, 0, 0]])
    A = np.array([[0.5, 0.1], [0.3, 0.1]])
    B = np.array([[1, 0], [0, 0]])
    C = np.array([[1, 1]])
    D = np.array([[0, 1]])
    
    from statespace_func import kalmanfilter
    x_t_tm1, P_t_tm1, x_t_t, P_t_t, y_t_tm1, Q_t_tm1, R_t_tm1 = kalmanfilter(y, A, B, C, D)


def test_logl():
    """
    Simulate data for a true value of A. Then compare log likelihoods for alternative values of A.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    A, B, C, D, Omega = simplestatespace()

    from statespace_func import statespace_simdata
    x, y, v = statespace_simdata(A, B, C, D, 1000)

    A_adjusted = A.copy()
    Aparam_val = np.linspace(0, 1, 100)
    logl_val = []
    for Aparam in Aparam_val:
        A_adjusted[0, 0] = Aparam

        from statespace_func import kalmanfilter
        x_t_tm1, P_t_tm1, x_t_t, P_t_t, y_t_tm1, Q_t_tm1, R_t_tm1 = kalmanfilter(y, A_adjusted, B, C, D, Omega)

        from statespace_func import logl_prop_kalmanfilter
        logl_val.append(logl_prop_kalmanfilter(y, y_t_tm1, Q_t_tm1))

    plt.plot(Aparam_val, logl_val)
    plt.xlabel(r'Parameter Value')
    plt.ylabel(r'Log Likelihood')
    plt.show()
    plt.clf()

    
# Run:{{{1
# test_logl()
test2_kalmanfilter()
