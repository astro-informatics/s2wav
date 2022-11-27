import numpy as np
from s2wav.math_utils import binomial_coefficient


def tiling_direction(L: int, N: int = 1) -> np.ndarray:
    r"""Generates the harmonic coefficients for the directionality component of the
        tiling functions.

    Formally, this function implements the follow equation

    .. math::

        _{s}\eta_{\el m} = \nu \vu \sqrt{\frac{1}{2^{\gamma}} \big ( \binom{\gamma}{
                (\gamma - m)/2} \big )}

    which was first derived in `[1] <https://arxiv.org/pdf/1211.1680.pdf>`_.

    Args:
        L (int): Harmonic band-limit.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

    Returns:
        np.ndarray: Harmonic coefficients of directionality components
            :math:`_{s}\eta_{\el m}`.

    Notes:
        [1] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint
            arXiv:1509.06749 (2015).
    """
    if N % 2:
        nu = 1
    else:
        nu = 1j

    ind = 1

    s_elm = np.zeros((L, 2 * L - 1), dtype=np.complex128)

    for el in range(1, L):
        if (N + el) % 2:
            gamma = min(N - 1, el)
        else:
            gamma = min(N - 1, el - 1)

        for m in range(-el, el + 1):
            if abs(m) < N and (N + m) % 2:
                s_elm[el, L - 1 + m] = nu * np.sqrt(
                    (binomial_coefficient(gamma, ((gamma - m) / 2)))
                    / (2**gamma)
                )
            else:
                s_elm[el, L - 1 + m] = 0.0

            ind += 1

    return s_elm


def spin_normalization(el: int, spin: int = 0) -> float:
    r"""Computes the normalization factor for spin-lowered wavelets, which is
        :math:`\sqrt{\frac{(l+s)!}{(l-s)!}}`.

    Args:
        el (int): Harmonic index :math:`\el`.

        spin (int): Spin of field over which to perform the transform. Defaults to 0.

    Returns:
        float: Normalization factor for spin-lowered wavelets.
    """
    factor = 1.0

    for s in range(-abs(spin) + 1, abs(spin) + 1):
        factor *= el + s

    if spin > 0:
        return np.sqrt(factor)
    else:
        return np.sqrt(1.0 / factor)


def spin_normalization_vectorised(el: np.ndarray, spin: int = 0) -> float:
    r"""Vectorised version of :func:`~spin_normalization`.

    Args:
        el (int): Harmonic index :math:`\el`.

        spin (int): Spin of field over which to perform the transform. Defaults to 0.

    Returns:
        float: Normalization factor for spin-lowered wavelets.
    """
    factor = np.arange(-abs(spin) + 1, abs(spin) + 1).reshape(
        1, 2 * abs(spin) + 1
    )
    factor = el.reshape(len(el), 1).dot(factor)
    return np.sqrt(np.prod(factor, axis=1) ** (np.sign(spin)))
