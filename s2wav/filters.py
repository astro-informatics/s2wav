import numpy as np
from s2wav import tiling, samples, kernels
from typing import Tuple


def filters_axisym(L: int, lam: float, J_min: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""Computes wavelet kernels :math:`\Psi^j_{\el m}` and scaling kernel :math:`\Phi_{\el m}` in harmonic space.

    Specifically, these kernels are derived in [1], where the wavelet kernels are defined (15) for scale :math:`j` to be

    .. math::

        \Psi^j_{\el m} \equiv \sqrt{\frac{2\el+1}{4\pi}} \kappa_{\lambda}(\frac{\el}{\lambda^j})\delta_{m0},

    where :math:`\kappa_{\lambda} = \sqrt{k_{\lambda}(t/\lambda) - k_{\lambda}(t)}` for :math:`k_{\lambda}` given in :func:`~k_lam`. Similarly, the scaling kernel is defined (16) as

    .. math::

        \Phi_{\el m} \equiv \sqrt{\frac{2\el+1}{4\pi}} \nu_{\lambda} (\frac{\el}{\lambda^{J_0}})\delta_{m0},

    where :math:`\nu_{\lambda} = \sqrt{k_{\lambda}(t)}` for :math:`k_{\lambda}` given in :func:`~k_lam`. Notice that :math:`\delta_{m0}` enforces that these kernels are axisymmetric, i.e. coefficients for :math:`m \not = \el` are zero. In this implementation the normalisation constant has been omitted as it is nulled in subsequent functions.

    Args:
        L (int): Harmonic band-limit.

        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.

        J_min (int): First wavelet scale used.

    Raises:
        ValueError: L is not an integer.

        ValueError: L is a negative integer.

        ValueError: J_min is not an integer.

        ValueError: J_min is negative or greater than J.

    Returns:
        (Tuple[np.ndarray, np.ndarray]): Unnormalised wavelet kernels :math:`\Psi^j_{\el m}` with shape :math:`[(J+1)*L], and scaling kernel :math:`\Phi_{\el m}` with shape :math:`[L]` in harmonic space.

    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
    """
    if not isinstance(L, int):
        raise TypeError("L must be an integer")

    if L < 0:
        raise ValueError("L must be non-negative")

    if not isinstance(J_min, int):
        raise TypeError("J_min must be an integer")

    J = samples.j_max(L, lam)

    if J_min >= J or J_min < 0:
        raise ValueError(
            "J_min must be non-negative and less than J= "
            + str(J)
            + " for given L and lam."
        )

    previoustemp = 0.0
    k = kernels.k_lam(L, lam)
    psi = np.zeros((J + 1) * L)
    phi = np.zeros(L)
    for l in range(L):
        phi[l] = np.sqrt(k[l + J_min * L])

    for j in range(J_min, J + 1):
        for l in range(L):
            diff = k[l + (j + 1) * L] - k[l + j * L]
            if diff < 0:
                psi[l + j * L] = previoustemp
            else:
                temp = np.sqrt(diff)
                psi[l + j * L] = temp
            previoustemp = temp

    return psi, phi


def filters_directional(
    L: int, lam: float, spin: int, original_spin: int, N: int, J_min: int
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Generates the harmonic coefficients for the directional tiling wavelets.

    This implementation is based on equation (33) in the wavelet computation paper [1].

    Args:
        L (int): Harmonic band-limit.

        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.

        spin (int): Spin (integer) to perform the transform.

        original_spin (int): Spin number the wavelet was lowered from.

        N (int): Upper orientational band-limit. Only flmn with :math:`n < N` will be stored.

        J_min (int): First wavelet scale used.

    Returns:
        (Tuple[np.ndarray, np.ndarray]): Tuple of wavelet and scaling kernels (:math:`\Psi^j_{\el n}`, :math:`\Phi_{\el m}`)
            psi (np.ndarray): Harmonic coefficients of directional wavelets with shape :math:`[L^2(J+1)]`.
            phi (np.ndarray): Harmonic coefficients of scaling function with shape :math:`[L]`.

    Notes:
        [1] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint arXiv:1509.06749 (2015).
    """
    J = samples.j_max(L, lam)
    el_min = max(abs(spin), abs(original_spin))

    phi = np.zeros(L, dtype=np.float64)
    psi = np.zeros((J + 1) * L * L, dtype=np.complex128)

    kappa, kappa0 = filters_axisym(L, lam, J_min)
    s_elm = tiling.tiling_direction(N, L)

    for el in range(el_min, L):
        phi[el] = np.sqrt((2 * el + 1) / 4.0 * np.pi) * kappa0[el]
        if original_spin != 0:
            phi[el] *= (
                tiling.spin_normalization(el, original_spin) * (-1) ** original_spin
            )

    for j in range(J_min, J + 1):
        ind = el_min * el_min
        for el in range(el_min, L):
            for m in range(-el, el + 1):
                psi[j * L * L + ind] = (
                    np.sqrt((2 * el + 1) / (8.0 * np.pi * np.pi))
                    * kappa[j * L + el]
                    * s_elm[ind]
                )
                if original_spin != 0:
                    psi[j * L * L + ind] *= (
                        tiling.spin_normalization(el, original_spin)
                        * (-1) ** original_spin
                    )
                ind += 1

    return psi, phi
