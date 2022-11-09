import numpy as np
import pytest


def tiling_integrand(t: float, lam: float) -> float:
    """Tiling integrand for wavelets. Intermediate step used to compute the wavelet 
        and scaling function generating functions.

        One of the basic mathematical functions needed to carry out the tiling of the harmonic space. 

    Args:
        t (float): Real argument over which we integrate.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        
    Returns:
        float: Value of tiling integrand for given t and scaling factor.
    """
    s_arg = (t - (1 / lam)) * (2.0 * lam / (lam - 1)) - 1
    print("Term 1: ", (t - (1 / lam)), " Term 2: ", (2.0 * lam / (lam - 1)), " t : ", t)

    integrand = np.exp(-2.0 / (1.0 - s_arg**2.0)) / t

    return integrand


def part_scaling_fn(a: float, b: float, n: int, lam: float) -> float:
    """Computes integral used to calculate smoothly decreasing function k_lambda.
    Intermediate step used to compute the wavelet and scaling function generating functions.


    Uses the trapezium method to integrate tiling_integrand() in the limits from a to b
    with scaling parameter lam. One of the basic mathematical functions needed to carry out the tiling of the harmonic space. 

    Args:
        a (float): Lower limit of the numerical integration.
        b (float): Upper limit of the numerical integration.
        n (int): Number of steps to be performed during integration.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.

    Returns:
        float: Integral of the tiling integrand from a to b.

    Raises:
        TypeError: If n is not of type integer
    """
    
    if not isinstance(n, int) == True:
        raise TypeError("n must be an integer")

    sum = 0.0
    h = (b - a) / n

    if (a == b):
        return 0
    
    else:
        for i in range(n):
            if (a + i*h not in [1/lam, 1.] and a + (i+1)*h not in [1/lam, 1.]):
                f1 = tiling_integrand(a + i * h, lam)
                f2 = tiling_integrand(a + (i + 1) * h, lam)

                sum += ((f1 + f2) * h) / 2

    return sum
