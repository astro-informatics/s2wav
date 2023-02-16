import numpy as np
from s2wav.filter_factory import tiling, kernels
from s2wav.utils import samples
from typing import Tuple


def filters_axisym(
    L: int, J_min: int = 0, lam: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Computes wavelet kernels :math:`\Psi^j_{\el m}` and scaling kernel :math:`\Phi_{\el m}` in harmonic space.

    Specifically, these kernels are derived in `[1] <https://arxiv.org/pdf/1211.1680.pdf>`_, where the wavelet kernels are defined (15) for scale :math:`j` to be

    .. math::

        \Psi^j_{\el m} \equiv \sqrt{\frac{2\el+1}{4\pi}} \kappa_{\lambda}(\frac{\el}{\lambda^j})\delta_{m0},

    where :math:`\kappa_{\lambda} = \sqrt{k_{\lambda}(t/\lambda) - k_{\lambda}(t)}` for :math:`k_{\lambda}` given in :func:`~k_lam`. Similarly, the scaling kernel is defined (16) as

    .. math::

        \Phi_{\el m} \equiv \sqrt{\frac{2\el+1}{4\pi}} \nu_{\lambda} (\frac{\el}{\lambda^{J_0}})\delta_{m0},

    where :math:`\nu_{\lambda} = \sqrt{k_{\lambda}(t)}` for :math:`k_{\lambda}` given in :func:`~k_lam`. Notice that :math:`\delta_{m0}` enforces that these kernels are axisymmetric, i.e. coefficients for :math:`m \not = \el` are zero. In this implementation the normalisation constant has been omitted as it is nulled in subsequent functions.

    Args:
        L (int): Harmonic band-limit.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

    Raises:
        ValueError: J_min is negative or greater than J.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Unnormalised wavelet kernels :math:`\Psi^j_{\el m}` with shape :math:`[(J+1)*L], and scaling kernel :math:`\Phi_{\el m}` with shape :math:`[L]` in harmonic space.

    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
    """
    J = samples.j_max(L, lam)

    if J_min >= J or J_min < 0:
        raise ValueError(
            "J_min must be non-negative and less than J= "
            + str(J)
            + " for given L and lam."
        )

    previoustemp = 0.0
    k = kernels.k_lam(L, lam)
    psi = np.zeros((J + 1, L), np.float64)
    phi = np.zeros(L, np.float64)
    for l in range(L):
        phi[l] = np.sqrt(k[J_min, l])

    for j in range(J_min, J + 1):
        for l in range(L):
            diff = k[j + 1, l] - k[j, l]
            if diff < 0:
                psi[j, l] = previoustemp
            else:
                temp = np.sqrt(diff)
                psi[j, l] = temp
            previoustemp = temp

    return psi, phi


def filters_directional(
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    spin0: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Generates the harmonic coefficients for the directional tiling wavelets.

    This implementation is based on equation 36 in the wavelet computation paper `[1] <https://arxiv.org/pdf/1509.06749.pdf>`_.

    Args:
        L (int): Harmonic band-limit.

        N (int, optional): Upper azimuthal band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

        spin (int, optional): Spin (integer) to perform the transform. Defaults to 0.

        spin0 (int, optional): Spin number the wavelet was lowered from. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of wavelet and scaling kernels (:math:`\Psi^j_{\el n}`, :math:`\Phi_{\el m}`)
            psi (np.ndarray): Harmonic coefficients of directional wavelets with shape :math:`[L^2(J+1)]`.
            phi (np.ndarray): Harmonic coefficients of scaling function with shape :math:`[L]`.

    Notes:
        [1] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint arXiv:1509.06749 (2015).
    """
    J = samples.j_max(L, lam)
    el_min = max(abs(spin), abs(spin0))

    phi = np.zeros(L, dtype=np.float64)
    psi = np.zeros((J + 1, L, 2 * L - 1), dtype=np.complex128)

    kappa, kappa0 = filters_axisym(L, J_min, lam)
    s_elm = tiling.tiling_direction(L, N)

    for el in range(el_min, L):
        if kappa0[el] != 0:
            phi[el] = np.sqrt((2 * el + 1) / (4.0 * np.pi)) * kappa0[el]
            if spin0 != 0:
                phi[el] *= tiling.spin_normalization(el, spin0) * (-1) ** spin0

    for j in range(J_min, J + 1):
        for el in range(el_min, L):
            if kappa[j, el] != 0:
                for m in range(-el, el + 1):
                    if s_elm[el, L - 1 + m] != 0:
                        psi[j, el, L - 1 + m] = (
                            np.sqrt((2 * el + 1) / (8.0 * np.pi * np.pi))
                            * kappa[j, el]
                            * s_elm[el, L - 1 + m]
                        )
                        if spin0 != 0:
                            psi[j, el, L - 1 + m] *= (
                                tiling.spin_normalization(el, spin0)
                                * (-1) ** spin0
                            )

    return psi, phi


def filters_axisym_vectorised(
    L: int, J_min: int = 0, lam: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Vectorised version of :func:`~filters_axisym`.

    Args:
        L (int): Harmonic band-limit.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.

    Raises:
        ValueError: J_min is negative or greater than J.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Unnormalised wavelet kernels :math:`\Psi^j_{\el m}`
        with shape :math:`[(J+1)*L], and scaling kernel :math:`\Phi_{\el m}` with shape
        :math:`[L]` in harmonic space.
    """
    J = samples.j_max(L, lam)

    if J_min >= J or J_min < 0:
        raise ValueError(
            "J_min must be non-negative and less than J= "
            + str(J)
            + " for given L and lam."
        )

    k = kernels.k_lam(L, lam)
    diff = (np.roll(k, -1, axis=0) - k)[:-1]
    diff[diff < 0] = 0
    return np.sqrt(diff), np.sqrt(k[J_min])


def filters_directional_vectorised(
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    spin0: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Vectorised version of :func:`~filters_directional`.

    Args:
        L (int): Harmonic band-limit.

        N (int, optional): Upper azimuthal band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        spin (int, optional): Spin (integer) to perform the transform. Defaults to 0.

        spin0 (int, optional): Spin number the wavelet was lowered from. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of wavelet and scaling kernels (:math:`\Psi^j_{\el n}`, :math:`\Phi_{\el m}`)
            psi (np.ndarray): Harmonic coefficients of directional wavelets with shape :math:`[L^2(J+1)]`.
            phi (np.ndarray): Harmonic coefficients of scaling function with shape :math:`[L]`.
    """
    el_min = max(abs(spin), abs(spin0))

    spin_norms = (
        (-1) ** spin0
        * tiling.spin_normalization_vectorised(np.arange(L), spin0)
        if spin0 != 0
        else 1
    )

    kappa, kappa0 = filters_axisym_vectorised(L, J_min, lam)
    s_elm = tiling.tiling_direction(L, N)

    kappa0 *= np.sqrt((2 * np.arange(L) + 1) / (4.0 * np.pi))
    kappa0 = kappa0 * spin_norms if spin0 != 0 else kappa0

    kappa *= np.sqrt((2 * np.arange(L) + 1) / 8.0) / np.pi
    kappa = np.einsum("ij,jk->ijk", kappa, s_elm)
    kappa = np.einsum("ijk,j->ijk", kappa, spin_norms) if spin0 != 0 else kappa

    kappa0[:el_min] = 0
    kappa[:, :el_min, :] = 0
    return kappa, kappa0
