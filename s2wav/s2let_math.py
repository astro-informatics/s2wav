"""Compute basic math tiling. """
import numpy as np


def f_s2dw(k, B):
    """Tiling function for S2DW wavelets"""
    t = (k - (1 / B)) * (2.0 * B / (B - 1)) - 1

    return np.exp(-2.0 / (1.0 - t**2.0)) / k



def f_needlet(t):
    """Tiling function for needlets"""

    return np.exp(-1.0 / (1.0 - t**2.0))


def b3_spline(x):
    """Computes cubis B-spline function"""
    if (np.abs(x) < 10E-16):
        return 0

    A1 = np.abs((x - 2) * (x - 2) * (x - 2))
    A2 = np.abs((x - 1) * (x - 1) * (x - 1))
    A3 = np.abs(x * x * x)
    A4 = np.abs((x + 1) * (x + 1) * (x + 1))
    A5 = np.abs((x + 2) * (x + 2) * (x + 2))
    Val = 1.0 / 12.0 * (A1 - 4.0 * A2 + 6.0 * A3 - 4.0 * A4 + A5)

    return Val


def s2let_math_spline_scalingfct(x, y):
    """Computes spline scaling function"""

    res = 1.5 * b3_spline(2.0 * x / y)

    return res


def s2let_math_kappa0_quadtrap_s2dw(a, b, n, B):
    """Computes smooth "Schwartz" functions for scale-discretised wavelets"""
    
    assert isinstance(n, int) == True, "n must be an integer"

    sum = 0.0
    h = (b - a) / n

    if (a == b):
        return 0
    
    else:
        for i in range(n):
            f1 = f_s2dw(a + i * h, B)
            f2 = f_s2dw(a + (i + 1) * h, B)

            if (not np.isnan(f1) and not np.isinf(f1) and not np.isnan(f2) and not np.isinf(f2)):
                sum += ((f1 + f2) * h) / 2
            
    
    return sum


def s2let_math_kappa0_quadtrap_needlet(a, b, n):
    """Computes smooth "Schwartz" functions for needlets"""

    assert isinstance(n, int) == True, "n must be an integer"

    sum = 0
    h = (b - a) / n

    if (a == b):
        return 0
    else:
        for i in range(n):
            f1 = f_needlet(a + i * h)
            f2 = f_needlet(a + (i + 1) * h)

            if (not np.isnan(f1) and not np.isinf(f1) and not np.isnan(f2) and not np.isinf(f2)):
                sum += ((f1 + f2) * h) / 2
    
    return sum