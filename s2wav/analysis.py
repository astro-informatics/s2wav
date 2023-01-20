import numpy as np
from typing import Tuple
from s2wav import samples, filters, shapes
import so3
import pyssht as ssht
import s2fft


def analysis_transform_looped(
    f: np.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    spin0: int = 0,
    sampling: str = "mw",
    reality: bool = False,
    multiresolution: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Wavelet analysis from pixel space to wavelet space for complex signals.

    Args:
        f (np.ndarray): Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

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

    Returns:
        f_wav (np.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (np.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.
    """
    J = samples.j_max(L, lam)
    Ls = shapes.scal_bandlimit(L, J_min, lam, multiresolution)

    f_scal_lm = shapes.construct_flm(L, J_min, lam, multiresolution)
    f_wav_lmn = shapes.construct_flmn(L, N, J_min, lam, multiresolution)
    wav_lm, scal_l = filters.filters_directional(L, N, J_min, lam, spin, spin0)

    flm = s2fft.transform.forward(f, L, spin, sampling)

    for j in range(J_min, J + 1):
        Lj, Nj = shapes.LN_j(L, j, N, lam, multiresolution)
        for n in range(-Nj + 1, Nj, 2):
            for el in range(max(abs(spin), abs(n)), Lj):
                if el != 0:
                    psi = np.conj(wav_lm[j, el, L - 1 + n])
                    psi *= 8 * np.pi**2 / (2 * el + 1)
                    for m in range(-el, el + 1):
                        f_wav_lmn[j - J_min][Nj - 1 + n, el, Lj - 1 + m] = (
                            flm[el, L - 1 + m] * psi
                        )

    for el in range(abs(spin), Ls):
        phi = np.sqrt(4.0 * np.pi / (2 * el + 1)) * scal_l[el]
        for m in range(-el, el + 1):
            f_scal_lm[el, Ls - 1 + m] = flm[el, L - 1 + m] * phi

    f_wav = shapes.construct_f(L, N, J_min, lam, sampling, multiresolution)
    for j in range(J_min, J + 1):
        Lj, Nj = shapes.LN_j(L, j, N, lam, multiresolution)
        f_wav[j - J_min] = s2fft.wigner.transform.inverse(
            f_wav_lmn[j - J_min], Lj, Nj, 0, sampling
        )

    f_scal = s2fft.transform.inverse(f_scal_lm, Ls, spin, sampling)
    return f_wav, f_scal


def analysis_transform_vectorised(
    f: np.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    spin0: int = 0,
    sampling: str = "mw",
    reality: bool = False,
    multiresolution: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Wavelet analysis from pixel space to wavelet space for complex signals.

    Args:
        f (np.ndarray): Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

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

    Returns:
        f_wav (np.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (np.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.
    """
    J = samples.j_max(L, lam)
    Ls = shapes.scal_bandlimit(L, J_min, lam, multiresolution)

    f_scal_lm = shapes.construct_flm(L, J_min, lam, multiresolution)
    f_wav_lmn = shapes.construct_flmn(L, N, J_min, lam, multiresolution)
    f_wav = shapes.construct_f(L, N, J_min, lam, sampling, multiresolution)

    # Generate the directional wavelet kernels
    wav_lm, scal_l = filters.filters_directional_vectorised(
        L, N, J_min, lam, spin, spin0
    )
    wav_lm = np.einsum(
        "jln, l->jln", np.conj(wav_lm), 8 * np.pi**2 / (2 * np.arange(L) + 1)
    )

    flm = s2fft.transform.forward(f, L, spin, sampling)

    # Project all wigner coefficients for each lmn onto wavelet coefficients
    # Note that almost the entire compute is concentrated at the highest J
    for j in range(J_min, J + 1):
        Lj, Nj = shapes.LN_j(L, j, N, lam, multiresolution)
        f_wav_lmn[j - J_min][::2] = np.einsum(
            "lm,ln->nlm",
            flm[:Lj, L - Lj : L - 1 + Lj],
            wav_lm[j, :Lj, L - Nj : L - 1 + Nj : 2],
        )
        f_wav[j - J_min] = s2fft.wigner.transform.inverse(
            f_wav_lmn[j - J_min], Lj, Nj, 0, sampling
        )

    # Project all harmonic coefficients for each lm onto scaling coefficients
    phi = scal_l[:Ls] * np.sqrt(4 * np.pi / (2 * np.arange(Ls) + 1))
    f_scal_lm = np.einsum("lm,l->lm", flm[:Ls, L - Ls : L - 1 + Ls], phi)

    return f_wav, s2fft.transform.inverse(f_scal_lm, Ls, spin, sampling)
