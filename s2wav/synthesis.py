import numpy as np
from s2wav import samples, filters, shapes
from typing import Tuple
import s2fft


def synthesis_transform_looped(
    f_wav: np.ndarray,
    f_scal: np.ndarray,
    L: int,
    N: int,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    spin0: int = 0,
    sampling: str = "mw",
    multiresolution: bool = False,
) -> np.ndarray:
    r"""Computes the synthesis directional wavelet transform [1,2].
    Specifically, this transform synthesises the signal :math:`_{s}f(\omega) \in \mathbb{S}^2` by summing the contributions from wavelet and scaling coefficients in harmonic space, see equation 27 from `[2] <https://arxiv.org/pdf/1509.06749.pdf>`_.
    Args:
        f_wav (np.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.
        f_scal (np.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.
        L (int): Harmonic bandlimit.
        N (int, optional): Upper azimuthal band-limit. Defaults to 1.
        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.
        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.
        spin (int, optional): Spin (integer) of input signal. Defaults to 0.
        spin0 (int, optional): Spin (integer) of output signal. Defaults to 0.
        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}. Defaults to "mw".
        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.
    Raises:
        AssertionError: Shape of wavelet/scaling coefficients incorrect.
    Returns:
        np.ndarray: Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.
    Notes:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
        [2] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint arXiv:1509.06749 (2015).
    """
    shapes.wavelet_shape_check(
        f_wav, f_scal, L, N, J_min, lam, sampling, multiresolution
    )
    J = samples.j_max(L, lam)
    Ls = shapes.scal_bandlimit(L, J_min, lam, multiresolution)
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    f_scal_lm = s2fft.transform.forward(f_scal, Ls, spin, sampling)

    # Generate the directional wavelet kernels
    wav_lm, scal_l = filters.filters_directional(L, N, J_min, lam, spin, spin0)

    # Sum the all wavelet wigner coefficients for each lmn
    # Note that almost the entire compute is concentrated at the highest J
    for j in range(J_min, J + 1):
        Lj, Nj = shapes.LN_j(L, j, N, lam, multiresolution)
        temp = s2fft.wigner.transform.forward(
            f_wav[j - J_min], Lj, Nj, 0, sampling
        )
        for n in range(-Nj + 1, Nj, 2):
            for el in range(max(abs(spin), abs(n)), Lj):
                psi = wav_lm[j, el, L - 1 + n]
                for m in range(-el, el + 1):
                    flm[el, L - 1 + m] += temp[Nj - 1 + n, el, Lj - 1 + m] * psi

    # Sum the all scaling harmonic coefficients for each lm
    for el in range(np.abs(spin), Ls):
        phi = np.sqrt(4 * np.pi / (2 * el + 1)) * scal_l[el]
        for m in range(-el, el + 1):
            flm[el, L - 1 + m] += f_scal_lm[el, Ls - 1 + m] * phi

    return s2fft.transform.inverse(flm, L, spin, sampling)


def synthesis_transform_vectorised(
    f_wav: np.ndarray,
    f_scal: np.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    spin0: int = 0,
    sampling: str = "mw",
    reality: bool = False,
    multiresolution: bool = False,
) -> np.ndarray:
    r"""Computes the synthesis directional wavelet transform [1,2].
    Specifically, this transform synthesises the signal :math:`_{s}f(\omega) \in \mathbb{S}^2` by summing the contributions from wavelet and scaling coefficients in harmonic space, see equation 27 from `[2] <https://arxiv.org/pdf/1509.06749.pdf>`_.
    Args:
        f_wav (np.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.
        f_scal (np.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.
        L (int): Harmonic bandlimit.
        N (int, optional): Upper azimuthal band-limit. Defaults to 1.
        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.
        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.
        spin (int, optional): Spin (integer) of input signal. Defaults to 0.
        spin0 (int, optional): Spin (integer) of output signal. Defaults to 0.
        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}. Defaults to "mw".
        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.
        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.
    Raises:
        AssertionError: Shape of wavelet/scaling coefficients incorrect.
    Returns:
        np.ndarray: Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.
    Notes:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
        [2] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint arXiv:1509.06749 (2015).
    """
    shapes.wavelet_shape_check(
        f_wav, f_scal, L, N, J_min, lam, sampling, multiresolution
    )

    J = samples.j_max(L, lam)
    Ls = shapes.scal_bandlimit(L, J_min, lam, multiresolution)
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    f_scal_lm = s2fft.transform.forward(f_scal, Ls, spin, sampling)

    # Generate the directional wavelet kernels
    wav_lm, scal_l = filters.filters_directional_vectorised(
        L, N, J_min, lam, spin, spin0
    )

    # Sum the all wavelet wigner coefficients for each lmn
    # Note that almost the entire compute is concentrated at the highest J
    for j in range(J_min, J + 1):
        Lj, Nj = shapes.LN_j(L, j, N, lam, multiresolution)
        temp = s2fft.wigner.transform.forward(
            f_wav[j - J_min], Lj, Nj, 0, sampling
        )
        flm[:Lj, L - Lj : L - 1 + Lj] += np.einsum(
            "ln,nlm->lm",
            wav_lm[j, :Lj, L - Nj : L - 1 + Nj : 2],
            temp[::2, :, :],
        )

    # Sum the all scaling harmonic coefficients for each lm
    for el in range(np.abs(spin), L):
        phi = np.sqrt(4 * np.pi / (2 * el + 1)) * scal_l[el]
        for m in range(-el, el + 1):
            flm[el, L - 1 + m] += f_scal_lm[el, L - 1 + m] * phi

    return ssht.inverse(s2wav_to_ssht(flm, L), L, Reality=reality)


def so3_to_s2wav(flmn, L, N):
    """Temporary function to convert flmn from so3 to s2wav indexing"""
    temp = flmn.reshape(2 * N - 1, L * L)
    flmn_out = np.zeros((2 * N - 1, L, 2 * L - 1), dtype=np.complex128)
    for n in range(-N + 1, N):
        ind = 0
        for el in range(L):
            for m in range(-el, el + 1):
                flmn_out[N - 1 + n, el, L - 1 + m] = temp[N - 1 + n, ind]
                ind += 1
    return flmn_out


def ssht_to_s2wav(flm, L):
    """Temporary function to convert flm from ssht to s2wav indexing"""
    flm_out = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    ind = 0
    for el in range(L):
        for m in range(-el, el + 1):
            flm_out[el, L - 1 + m] = flm[ind]
            ind += 1
    return flm_out


def s2wav_to_ssht(flm, L):
    """Temporary function to convert flm from s2wav to ssht indexing"""
    flm_out = np.zeros(L * L, dtype=np.complex128)
    ind = 0
    for el in range(L):
        for m in range(-el, el + 1):
            flm_out[ind] = flm[el, L - 1 + m]
            ind += 1
    return flm_out
