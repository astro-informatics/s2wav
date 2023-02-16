import numpy as np
from scipy.special import loggamma


def logfact(n: int) -> float:
    r"""Computes the natural logarithm of the factorial of an integer.

    The engine of this function actually calculates the gamma function
    :math:`\Gamma(n)`, for which the real argument is :math:`x = n + 1`.

    Args:
        n (int): Integer for which to compute logarithm of the factorial.

    Returns:
        (float): Stable natural logarithm of large factorial, i.e. :math:`\log(n!)`.
    """

    # Fitting constants
    c = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ]

    # This calculates the gamma function, which takes the argument x = n +1.
    x = n + 1.0

    # Add up the fit.
    temp = x + 5.5 - (x + 0.5) * np.log(x + 5.5)
    sum = 1.000000000190015
    y = x

    for i in range(0, 6):
        y += 1
        sum = sum + c[i] / y

    return -temp + np.log(2.5066282746310005 * sum / x)


def binomial_coefficient(n: int, k: int) -> int:
    r"""Computes the binomial coefficient :math:`\binom{n}{k}`.

    Args:
        n (int): Number of elements to choose from.

        k (int): Number of elements to pick.

    Returns:
        (int): Number of possible subsets.
    """
    return np.floor(
        0.5 + np.exp(loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1))
    )
