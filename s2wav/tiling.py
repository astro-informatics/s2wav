import numpy as np
from s2wav.math_utils import binomial_coefficient


def tiling_integrand(t: float, lam: float = 2.0) -> float:
    r"""Tiling integrand for scale-discretised wavelets `[1] <https://arxiv.org/pdf/1211.1680.pdf>`_.

    Intermediate step used to compute the wavelet and scaling function generating functions. One of the basic mathematical functions needed to carry out the tiling of the harmonic space.

    Args:
        t (float): Real argument over which we integrate.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

    Returns:
        float: Value of tiling integrand for given :math:`t` and scaling factor.

    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
    """
    s_arg = (t - (1 / lam)) * (2.0 * lam / (lam - 1)) - 1

    integrand = np.exp(-2.0 / (1.0 - s_arg**2.0)) / t

    return integrand


def part_scaling_fn(a: float, b: float, n: int, lam: float = 2.0) -> float:
    r"""Computes integral used to calculate smoothly decreasing function :math:`k_{\lambda}`.

    Intermediate step used to compute the wavelet and scaling function generating functions. Uses the trapezium method to integrate :func:`~tiling_integrand` in the limits from :math:`a \rightarrow b` with scaling parameter :math:`\lambda`. One of the basic mathematical functions needed to carry out the tiling of the harmonic space.

    Args:
        a (float): Lower limit of the numerical integration.

        b (float): Upper limit of the numerical integration.

        n (int): Number of steps to be performed during integration.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

    Raises:
        TypeError: If :math:`n` is not of type integer

    Returns:
        float: Integral of the tiling integrand from :math:`a \rightarrow b`.
    """

    if not isinstance(n, int) == True:
        raise TypeError("n must be an integer")

    sum = 0.0
    h = (b - a) / n

    if a == b:
        return 0

    else:
        for i in range(n):
            if a + i * h not in [1 / lam, 1.0] and a + (i + 1) * h not in [
                1 / lam,
                1.0,
            ]:
                f1 = tiling_integrand(a + i * h, lam)
                f2 = tiling_integrand(a + (i + 1) * h, lam)

                sum += ((f1 + f2) * h) / 2

    return sum


def tiling_direction(L: int, N: int = 1) -> np.ndarray:
    r"""Generates the harmonic coefficients for the directionality component of the tiling functions.

    Formally, this function implements the follow equation

    .. math::

        _{s}\eta_{\el m} = \nu \vu \sqrt{\frac{1}{2^{\gamma}} \big ( \binom{\gamma}{(\gamma - m)/2} \big )}

    which was first derived in `[1] <https://arxiv.org/pdf/1211.1680.pdf>`_.

    Args:
        L (int): Harmonic band-limit.

        N (int, optional): Upper orientational band-limit. Only flmn with :math:`n < N` will be stored.
            Defaults to 1.

    Returns:
        np.ndarray: Harmonic coefficients of directionality components :math:`_{s}\eta_{\el m}`.

    Notes:
        [1] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint arXiv:1509.06749 (2015).
    """
    if N % 2:
        nu = 1
    else:
        nu = 1j

    ind = 1

    s_elm = np.zeros(L * L, dtype=np.complex128)

    for el in range(1, L):
        if (N + el) % 2:
            gamma = min(N - 1, el)
        else:
            gamma = min(N - 1, el - 1)

        for m in range(-el, el + 1):
            if abs(m) < N and (N + m) % 2:
                s_elm[ind] = nu * np.sqrt(
                    (binomial_coefficient(gamma, ((gamma - m) / 2))) / (2**gamma)
                )
            else:
                s_elm[ind] = 0.0

            ind += 1

    return s_elm


def spin_normalization(el: int, spin: int = 0) -> float:
    r"""Computes the normalization factor for spin-lowered wavelets, which is :math:`\sqrt{\frac{(l+s)!}{(l-s)!}}`.

    Args:
        el (int): Harmonic index :math:`\el`.

        spin (int): Spin of field over which to perform the transform. Defaults to 0.

    Returns:
        float: Normalization factor for spin-lowered wavelets.
    """
    factor = 1

    for s in range(-abs(spin) + 1, abs(spin) + 1):
        factor *= el + s

    if spin > 0:
        return np.sqrt(factor)
    else:
        return np.sqrt(1.0 / factor)
