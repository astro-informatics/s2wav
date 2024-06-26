from jax import jit
import jax.numpy as jnp
import numpy as np
import torch
import math
from functools import partial
from typing import Tuple, List
from s2fft.sampling import s2_samples, so3_samples
from scipy.special import loggamma
from jax.scipy.special import gammaln as jax_gammaln


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

        sampling (str, optional): Spherical sampling scheme from
            {"mw","mwss", "dh", "gl", "healpix"}. Defaults to "mw".

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
    return s2_samples.f_shape(L_s, sampling, nside)


def n_wav_scales(L: int, N: int = 1, J_min: int = 0, lam: float = 2.0) -> int:
    r"""Evalutes the total number of wavelet scales.

    Args:
        L (int): Harmonic bandlimit.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        sampling (str, optional): Spherical sampling scheme from {"mw", "mwss", "dh", "gl", "healpix"}.
            Defaults to "mw".

    Returns:
        int: Total number of wavelet scales :math:`n_{j}`.
    """
    return j_max(L, lam) - J_min + 1


def L0_j(j: int, lam: float = 2.0) -> int:
    r"""Computes the minimum harmonic index supported by the given wavelet scale :math:`j`.

    Args:
        j (int): Wavelet scale to consider.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

    Raises:
        ValueError: Kernel type not supported.

    Returns:
        int: The minimum harmonic multipole :math:`el` which is supported by a given wavelet scale.
    """

    return math.ceil(lam ** (j - 1)) if j != 0 else 0


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

        sampling (str, optional): Spherical sampling scheme from {"mw", "mwss", "dh", "gl", "healpix"}.
            Defaults to "mw".

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        int: Total number of wavelet scales :math:`n_{j}`.
    """
    if j == 0:
        j += 1
    Lj = wav_j_bandlimit(L, j, lam, multiresolution)
    L0j = L0_j(j, lam) if multiresolution else 0
    Nj = N
    if multiresolution:
        Nj = min(N, Lj)
        Nj += (Nj + N) % 2
    return Lj, Nj, L0j


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

        sampling (str, optional): Spherical sampling scheme from
            {"mw", "mwss", "dh", "gl", "healpix"}. Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        Tuple[int, int, int]: Wavelet coefficients shape :math:`[n_{N}, n_{\theta}, n_{\phi}]`.

    Note:
        Each wavelet filter has compact support between a lower and upper harmonic degree.
        Therefore it is possible to conserve information with only a fraction of the
        data per scale :math:`j`. As each scale :math:`j` has a different shape the overall
        wavelet coefficients are stored as a list of arrays, this being the shape of
        one array within such a list.
    """
    Lj, Nj, _ = LN_j(L, j, N, lam, multiresolution)

    return so3_samples.f_shape(Lj, Nj, sampling, nside)


def construct_f(
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    multiresolution: bool = False,
    scattering: bool = False,
) -> np.ndarray:
    """Defines a list of arrays corresponding to f_wav.

    Args:
        L (int): Harmonic bandlimit.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        sampling (str, optional): Spherical sampling scheme from
            {"mw", "mwss", "dh", "gl", "healpix"}. Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required if
            sampling="healpix".  Defaults to None.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

        scattering (bool, optional): Whether to create minimal arrays for scattering transform to
            optimise for memory. Defaults to False.

    Returns:
        np.ndarray: Empty array (or list of empty arrays) in which to write data.
    """
    J = j_max(L, lam)
    if scattering:
        f = np.zeros(
            f_wav_j(L, J - 1, N, lam, sampling, nside, multiresolution),
            dtype=np.complex128,
        )
    else:
        f = []
        for j in range(J_min, J + 1):
            f.append(
                np.zeros(
                    f_wav_j(L, j, N, lam, sampling, nside, multiresolution),
                    dtype=np.complex128,
                )
            )
    return f


@partial(jit, static_argnums=(0, 1, 2, 3))
def construct_f_jax(
    L: int,
    J_min: int = 0,
    J_max: int = None,
    lam: float = 2.0
) -> List:
    """Defines a list corresponding to f_wav.

    Args:
        L (int): Harmonic bandlimit.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        J_max (int, optional): Highest frequency wavelet scale to be used. Defaults to None.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

    Returns:
        List: Empty list in which to write data.
    """
    J = J_max if J_max is not None else j_max(L, lam)
    f = []
    for _ in range(J_min, J + 1):
        f.append([])
    return f


def construct_f_torch(
    L: int,
    J_min: int = 0,
    J_max: int = None,
    lam: float = 2.0
) -> List:
    """Defines a list corresponding to f_wav.

    Args:
        L (int): Harmonic bandlimit.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        J_max (int, optional): Highest frequency wavelet scale to be used. Defaults to None.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

    Returns:
        List: Empty list in which to write data.
    """
    J = J_max if J_max is not None else j_max(L, lam)
    f = []
    for _ in range(J_min, J + 1):
        f.append([])
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
        Tuple[int, int]: Scaling coefficients shape :math:`[L, 2L-1]`.
    """
    L_s = scal_bandlimit(L, J_min, lam, multiresolution)
    return np.zeros((L_s, 2 * L_s - 1), dtype=np.complex128)


@partial(jit, static_argnums=(0, 1, 2, 3))
def construct_flm_jax(
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
        Tuple[int, int]: Scaling coefficients shape :math:`[L, 2L-1]`.
    """
    L_s = scal_bandlimit(L, J_min, lam, multiresolution)
    return jnp.zeros((L_s, 2 * L_s - 1), dtype=jnp.complex128)


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
        Tuple[int, int, int]: Wavelet coefficients shape :math:`[n_{N}, L, 2L-1]`.

    Note:
        Each wavelet filter has compact support between a lower and upper harmonic degree.
        Therefore it is possible to conserve information with only a fraction of the
        data per scale :math:`j`. As each scale :math:`j` has a different shape the overall
        wavelet coefficients are stored as a list of arrays, this being the shape of
        one array within such a list.
    """
    Lj, Nj, _ = LN_j(L, j, N, lam, multiresolution)
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
    if j == 0:
        j += 1
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
    scattering: bool = False,
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

        scattering (bool, optional): Whether to create minimal arrays for scattering transform to
            optimise for memory. Defaults to False.

    Returns:
        np.ndarray: Empty array (or list of empty arrays) in which to write data.
    """
    J = j_max(L, lam)
    if scattering:
        flmn = np.zeros(
            flmn_wav_j(L, J - 1, N, lam, multiresolution), dtype=np.complex128
        )
    else:
        flmn = []
        for j in range(J_min, J + 1):
            flmn.append(
                np.zeros(
                    flmn_wav_j(L, j, N, lam, multiresolution),
                    dtype=np.complex128,
                )
            )
    return flmn


@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5))
def construct_flmn_jax(
    L: int,
    N: int = 1,
    J_min: int = 0,
    J_max: int = None,
    lam: float = 2.0,
    multiresolution: bool = False
) -> jnp.ndarray:
    """Defines a list of arrays corresponding to flmn.

    Args:
        L (int): Harmonic bandlimit.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        J_max (int, optional): Highest frequency wavelet scale to be used. Defaults to None.c

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        jnp.ndarray: Empty array (or list of empty arrays) in which to write data.
    """
    J = J_max if J_max is not None else j_max(L, lam)
    flmn = []
    for j in range(J_min, J + 1):
        flmn.append(
            jnp.zeros(
                flmn_wav_j(L, j, N, lam, multiresolution),
                dtype=jnp.complex128,
            )
        )
    return flmn


def construct_flmn_torch(
    L: int,
    N: int = 1,
    J_min: int = 0,
    J_max: int = None,
    lam: float = 2.0,
    multiresolution: bool = False
) -> torch.tensor:
    """Defines a list of tensors corresponding to flmn.

    Args:
        L (int): Harmonic bandlimit.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        J_max (int, optional): Highest frequency wavelet scale to be used. Defaults to None.c

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        torch.tensor: Empty tensor (or list of empty tensors) in which to write data.
    """
    J = J_max if J_max is not None else j_max(L, lam)
    flmn = []
    for j in range(J_min, J + 1):
        flmn.append(
            torch.zeros(
                flmn_wav_j(L, j, N, lam, multiresolution),
                dtype=torch.complex128,
            )
        )
    return flmn


def j_max(L: int, lam: float = 2.0) -> int:
    r"""Computes needlet maximum level required to ensure exact reconstruction.

    Args:
        L (int): Harmonic band-limit.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

    Returns:
        int: The maximum wavelet scale used.
    """
    return int(np.ceil(np.log(L) / np.log(lam)))


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

        sampling (str, optional): Spherical sampling scheme from
            {"mw", "mwss", "dh", "gl", "healpix"}. Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required
            if sampling="healpix".  Defaults to None.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.
    """
    assert len(f_w) == n_wav_scales(L, N, J_min, lam)
    assert f_s.shape == f_scal(L, J_min, lam, sampling, nside, multiresolution)

    J = j_max(L, lam)
    for j in range(J_min, J + 1):
        assert f_w[j - J_min].shape == f_wav_j(
            L, j, N, lam, sampling, nside, multiresolution
        )


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


def binomial_coefficient_jax(n: int, k: int) -> int:
    r"""Computes the binomial coefficient :math:`\binom{n}{k}`.

    Args:
        n (int): Number of elements to choose from.

        k (int): Number of elements to pick.

    Returns:
        (int): Number of possible subsets.
    """
    return jnp.floor(
        0.5 + jnp.exp(jax_gammaln(n + 1) - jax_gammaln(k + 1) - jax_gammaln(n - k + 1))
    )
