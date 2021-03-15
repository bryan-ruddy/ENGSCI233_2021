import numpy as np

def approx_factorial(x):
    if x < 0:
        raise ValueError
    return (2*np.pi*x)**0.5*(x/np.e)**x