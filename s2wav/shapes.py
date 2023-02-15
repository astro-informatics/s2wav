from audioop import mul
import numpy as np
import math
from s2wav import samples
from typing import Tuple
from s2fft.wigner import samples as wigner_samples
from s2fft import samples as harm_samples

# TODO: Add support for symmetries etc.


def f_scal(
    L: int,
    J_min: int = 0,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    multiresolution: bool = False,
) -> Tuple[int, int]:
    r"""Computes the shape of scaling coefficients in pixel-space.

    Args:
        L (int): Harmonic bandlimit.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}.
            Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        Tuple[int, int]: Scaling coefficients shape :math:`[n_{\theta}, n_{\phi}]`.
    """
    if multiresolution:
        L_s = min(math.ceil(lam**J_min), L)
    else:
        L_s = L
    return harm_samples.f_shape(L_s, sampling, nside)


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
        Tuple[int, int, int, int]: Wavelet coefficients shape :math:`[n_{J}, n_{\theta}, n_{\phi}, n_{N}]`.
    """
    J = samples.j_max(L, lam)
    return (
        (J - J_min + 1),
        (2 * N - 1),
        samples.ntheta(L, sampling),
        samples.nphi(L, sampling),
    )


def n_wav_scales(L: int, N: int = 1, J_min: int = 0, lam: float = 2.0) -> int:
    r"""Evalutes the total number of wavelet scales.

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
        int: Total number of wavelet scales :math:`n_{j}`.
    """
    return samples.j_max(L, lam) - J_min + 1


def LN_j(
    L: int,
    j: int = 0,
    N: int = 1,
    lam: float = 2.0,
    multiresolution: bool = False,
) -> Tuple[int, int]:
    r"""Computes the harmonic bandlimit and directionality for scale :math:`j`.

    Args:
        L (int): Harmonic bandlimit.

        j (int): Wavelet scale to consider.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}.
            Defaults to "mw".

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        int: Total number of wavelet scales :math:`n_{j}`.
    """
    Lj = wav_j_bandlimit(L, j, lam, multiresolution)
    Nj = N
    if multiresolution:
        Nj = min(N, Lj)
        Nj += (Nj + N) % 2
    return Lj, Nj


def f_wav_j(
    L: int,
    j: int = 0,
    N: int = 1,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    multiresolution: bool = False,
) -> Tuple[int, int, int]:
    r"""Computes the shape of wavelet coefficients :math:`f^j` in pixel-space.

    Args:
        L (int): Harmonic bandlimit.

        j (int): Wavelet scale to consider.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}.
            Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        Tuple[int, int, int]: Wavelet coefficients shape :math:`[n_{\theta}, n_{\phi}, n_{N}]`.

    Note:
        Each wavelet filter has compact support between a lower and upper harmonic degree.
        Therefore it is possible to conserve information with only a fraction of the
        data per scale :math:`j`. As each scale :math:`j` has a different shape the overall
        wavelet coefficients are stored as a list of arrays, this being the shape of
        one array within such a list.
    """
    Lj, Nj = LN_j(L, j, N, lam, multiresolution)

    return wigner_samples.f_shape(Lj, Nj, sampling, nside)


def construct_f(
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    multiresolution: bool = False,
) -> np.ndarray:
    """Defines a list of arrays corresponding to f_wav.

    Args:
        L (int): Harmonic bandlimit.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}.
            Defaults to "mw".
        
        nside (int, optional): HEALPix Nside resolution parameter.  Only required if 
            sampling="healpix".  Defaults to None.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        Tuple[int, int, int, int]: Wavelet coefficients shape :math:`[n_{J}, L, 2L-1, n_{N}]`.
    """
    J = samples.j_max(L, lam)
    f = []
    for j in range(J_min, J + 1):
        f.append(
            np.zeros(
                f_wav_j(L, j, N, lam, sampling, nside, multiresolution),
                dtype=np.complex128,
            )
        )
    return f


def construct_flm(
    L: int, J_min: int = 0, lam: float = 2.0, multiresolution: bool = False
) -> Tuple[int, int]:
    r"""Returns the shape of scaling coefficients in harmonic space.

    Args:
        L (int): Harmonic bandlimit.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        Tuple[int, int]: Scaling coefficients shape :math:`[L, 2*L-1]`.
    """
    L_s = scal_bandlimit(L, J_min, lam, multiresolution)
    return np.zeros((L_s, 2 * L_s - 1), dtype=np.complex128)


def scal_bandlimit(
    L: int, J_min: int = 0, lam: float = 2.0, multiresolution: bool = False
) -> int:
    r"""Returns the harmominc bandlimit of the scaling coefficients.

    Args:
        L (int): Harmonic bandlimit.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        int: Harmonic bandlimit of scaling coefficients.
    """
    if multiresolution:
        return min(math.ceil(lam**J_min), L)
    else:
        return L


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
        Tuple[int, int, int, int]: Wavelet coefficients shape :math:`[n_{J}, L, 2L-1, n_{N}]`.
    """
    J = samples.j_max(L, lam)
    return (J - J_min + 1), (2 * N - 1), L, 2 * L - 1


def flmn_wav_j(
    L: int,
    j: int,
    N: int = 1,
    lam: float = 2.0,
    multiresolution: bool = False,
) -> Tuple[int, int, int]:
    r"""Returns the shape of wavelet coefficients :math:`f^j_{\ell m n}` in Wigner space.

    Args:
        L (int): Harmonic bandlimit.

        j (int): Wavelet scale to consider.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        Tuple[int, int, int]: Wavelet coefficients shape :math:`[L, 2L-1, n_{N}]`.

    Note:
        Each wavelet filter has compact support between a lower and upper harmonic degree.
        Therefore it is possible to conserve information with only a fraction of the
        data per scale :math:`j`. As each scale :math:`j` has a different shape the overall
        wavelet coefficients are stored as a list of arrays, this being the shape of
        one array within such a list.
    """
    Lj, Nj = LN_j(L, j, N, lam, multiresolution)
    return 2 * Nj - 1, Lj, 2 * Lj - 1


def wav_j_bandlimit(
    L: int, j: int, lam: float = 2.0, multiresolution: bool = False
) -> int:
    r"""Returns the harmominc bandlimit of the scaling coefficients.

    Args:
        L (int): Harmonic bandlimit.

        j (int): Wavelet scale to consider.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        int: Harmonic bandlimit of scaling coefficients.
    """
    if multiresolution:
        return min(math.ceil(lam ** (j + 1)), L)
    else:
        return L


def construct_flmn(
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    multiresolution: bool = False,
) -> np.ndarray:
    """Defines a list of arrays corresponding to flmn.

    Args:
        L (int): Harmonic bandlimit.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        Tuple[int, int, int, int]: Wavelet coefficients shape :math:`[n_{J}, L, 2L-1, n_{N}]`.
    """
    J = samples.j_max(L, lam)
    flmn = []
    for j in range(J_min, J + 1):
        flmn.append(
            np.zeros(
                flmn_wav_j(L, j, N, lam, multiresolution), dtype=np.complex128
            )
        )
    return flmn


def wavelet_shape_check(
    f_w: np.ndarray,
    f_s: np.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    multiresolution: bool = False,
):
    r"""Checks the shape of wavelet coefficients are correct.

    If multiresolution is true, this check will iterate over each scale ensuring that
    each scale :math:`j` is correctly bandlimited at :math:`\lambda^j`.

    Args:
        f_w (np.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_s (np.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.

        L (int): Harmonic bandlimit.

        j (int): Wavelet scale to consider.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}.
            Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.
    """
    assert len(f_w) == n_wav_scales(L, N, J_min, lam)
    assert f_s.shape == f_scal(L, J_min, lam, sampling, nside, multiresolution)

    J = samples.j_max(L, lam)
    for j in range(J_min, J + 1):
        assert f_w[j - J_min].shape == f_wav_j(
            L, j, N, lam, sampling, nside, multiresolution
        )
