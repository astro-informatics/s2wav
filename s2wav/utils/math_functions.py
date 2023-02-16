import numpy as np
from scipy.special import loggamma


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
