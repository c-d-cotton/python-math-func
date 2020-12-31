#!/usr/bin/env python3
import os
from pathlib import Path
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

import numpy as np

def masmoothseries(series, ma):
    series = np.array(series)
    smoothedseries = np.array([np.nan] * len(series))

    for i in range(ma // 2, len(series) - ma // 2):
        if ma % 2 == 0:
            # even case
            smoothedseries[i] = 1 / ma * (0.5 * series[i - ma // 2] + np.sum(series[i - ma // 2 + 1: i + ma //2]) + 0.5 * series[i + ma // 2])
        else:
            # odd case
            smoothedseries[i] = 1 / ma * ( np.sum(series[i - ma // 2: i + ma //2 + 1]) )

    return(smoothedseries)


def masmoothseries_test():
    print( masmoothseries([0, 1, 2, 3, 4], 2) )
    print( masmoothseries([0, 1, 2, 3, 4], 3) )
    print( masmoothseries([0, 1, 2, 3, 4, 56], 4) )
