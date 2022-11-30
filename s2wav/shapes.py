import numpy as np
from s2wav import samples
from typing import Tuple

# TODO: Add support for symmetries etc.


def f_scal(L: int, sampling: str = "mw") -> Tuple[int, int]:
    r"""Computes the shape of scaling coefficients in pixel-space.

    Args:
        L (int): Harmonic bandlimit.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}. Defaults to "mw".

    Returns:
        Tuple[int, int]: Scaling coefficients shape :math:`[n_{\theta}, n_{\phi}]`.
    """
    return samples.ntheta(L, sampling), samples.nphi(L, sampling)


def f_wav(
    L: int, N: int = 1, J_min: int = 0, lam: float = 2.0, sampling: str = "mw"
) -> Tuple[int, int, int, int]:
    r"""Computes the shape of wavelet coefficients in pixel-space.

    Args:
        L (int): Harmonic bandlimit.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}.
            Defaults to "mw".

    Returns:
        Tuple[int, int, int, int]: Wavelet coefficients shape :math:`[n_{J}, n_{N}, L^2]`.
    """
    J = samples.j_max(L, lam)
    return (
        (J - J_min + 1),
        (2 * N - 1),
        samples.ntheta(L, sampling),
        samples.nphi(L, sampling),
    )


def flm_scal(L: int) -> Tuple[int, int]:
    r"""Returns the shape of scaling coefficients in harmonic space.

    Args:
        L (int): Harmonic bandlimit.

    Returns:
        Tuple[int, int]: Scaling coefficients shape :math:`[L, 2*L-1]`.
    """
    return L, 2 * L - 1


def flmn_wav(
    L: int, N: int = 1, J_min: int = 0, lam: float = 2.0
) -> Tuple[int, int, int, int]:
    """Returns the shape of wavelet coefficients in Wigner space.

    Args:
        L (int): Harmonic bandlimit.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

    Returns:
        Tuple[int, int, int, int]: Wavelet coefficients shape :math:`[n_{J}, n_{N}, L, 2L-1]`.
    """
    J = samples.j_max(L, lam)
    return (J - J_min + 1), (2 * N - 1), L, 2 * L - 1
