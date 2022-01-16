import math
import autograd.numpy as np

def Hardlim(x, threshold = 0, Symmetrical=False):
    if Symmetrical:
        return 0 if x<0 else 1
    elif not Symmetrical:
        return -1 if x<0 else 1

def purelin(x):
    return x

def satlin(x, Symmetrical=False):
    if Symmetrical:
        if x < -1:
            return -1
        elif x > 1:
            return 1
        else:
            return x

    if not Symmetrical:
        if x < 0:
            return 0
        elif x > 1:
            return 1
        else:
            return x

def logsig(x):
    return 1./(1. + np.exp(-x))

def tansig(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def poslin(x):
    return max(x, 0)