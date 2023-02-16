import numpy as np
from s2wav import samples, filters, shapes
from typing import Tuple
from s2fft import base_transforms as base


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
    nside: int = None,
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
        sampling (str, optional): Spherical sampling scheme from {"mw","mwss", "dh", "healpix"}. Defaults to "mw".
        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.
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
        f_wav, f_scal, L, N, J_min, lam, sampling, nside, multiresolution
    )
    J = samples.j_max(L, lam)
    Ls = shapes.scal_bandlimit(L, J_min, lam, multiresolution)
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    f_scal_lm = base.spherical.forward(
        f_scal, Ls, spin, sampling, nside, reality
    )

    # Generate the directional wavelet kernels
    wav_lm, scal_l = filters.filters_directional(L, N, J_min, lam, spin, spin0)

    # Sum the all wavelet wigner coefficients for each lmn
    # Note that almost the entire compute is concentrated at the highest J
    for j in range(J_min, J + 1):
        Lj, Nj = shapes.LN_j(L, j, N, lam, multiresolution)
        temp = base.wigner.forward(
            f_wav[j - J_min], Lj, Nj, 0, sampling, reality, nside
        )
        for n in range(-Nj + 1, Nj, 2):
            for el in range(max(abs(spin), abs(n)), Lj):
                psi = wav_lm[j, el, L - 1 + n]
                for m in range(-el, el + 1):
                    flm[el, L - 1 + m] += temp[Nj - 1 + n, el, Lj - 1 + m] * psi

    # Sum the all scaling harmonic coefficients for each lm
    for el in range(np.abs(spin), Ls):
        phi = np.sqrt(4 * np.pi / (2 * el + 1)) * scal_l[el]

        flm[el, 0 + L - 1] += f_scal_lm[el, Ls - 1 + 0] * phi
        for m in range(1, el + 1):
            flm[el, L - 1 + m] += f_scal_lm[el, Ls - 1 + m] * phi
            if reality:
                flm[el, -m + L - 1] += (-1) ** m * np.conj(flm[el, m + L - 1])
            else:
                flm[el, -m + L - 1] += f_scal_lm[el, Ls - 1 - m] * phi

    return base.spherical.inverse(flm, L, spin, sampling, nside, reality)


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
    nside: int = None,
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
        sampling (str, optional): Spherical sampling scheme from {"mw","mwss", "dh", "healpix"}. Defaults to "mw".
        nside (int, optional): HEALPix Nside resolution parameter.  Only required if sampling="healpix".  Defaults
            to None.
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
        f_wav, f_scal, L, N, J_min, lam, sampling, nside, multiresolution
    )

    J = samples.j_max(L, lam)
    Ls = shapes.scal_bandlimit(L, J_min, lam, multiresolution)
    flm = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    f_scal_lm = base.spherical.forward(
        f_scal, Ls, spin, sampling, nside, reality
    )

    # Generate the directional wavelet kernels
    wav_lm, scal_l = filters.filters_directional_vectorised(
        L, N, J_min, lam, spin, spin0
    )

    # Sum the all wavelet wigner coefficients for each lmn
    # Note that almost the entire compute is concentrated at the highest J
    for j in range(J_min, J + 1):
        Lj, Nj = shapes.LN_j(L, j, N, lam, multiresolution)
        temp = base.wigner.forward(
            f_wav[j - J_min], Lj, Nj, 0, sampling, reality, nside
        )
        flm[:Lj, L - Lj : L - 1 + Lj] += np.einsum(
            "ln,nlm->lm",
            wav_lm[j, :Lj, L - Nj : L - 1 + Nj : 2],
            temp[::2, :, :],
        )

    # Sum the all scaling harmonic coefficients for each lm
    phi = scal_l[:Ls] * np.sqrt(4 * np.pi / (2 * np.arange(Ls) + 1))
    flm[:Ls, L - Ls : L - 1 + Ls] += np.einsum("lm,l->lm", f_scal_lm, phi)

    return base.spherical.inverse(flm, L, spin, sampling, nside, reality)
