from jax import jit
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List
import s2fft
from s2wav import samples
from s2wav.transforms import construct


@partial(jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
def synthesis(
    f_wav: jnp.ndarray,
    f_scal: jnp.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    filters: Tuple[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None,
) -> jnp.ndarray:
    r"""Computes the synthesis directional wavelet transform [1,2].
        Specifically, this transform synthesises the signal :math:`_{s}f(\omega) \in \mathbb{S}^2`
        by summing the contributions from wavelet and scaling coefficients in harmonic space,
        see equation 27 from `[2] <https://arxiv.org/pdf/1509.06749.pdf>`_.

    Args:
        f_wav (jnp.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (jnp.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.

        L (int): Harmonic bandlimit.

        N (int, optional): Upper azimuthal band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 1.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

        spin (int, optional): Spin (integer) of input signal. Defaults to 0.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss", "dh",
            "healpix"}. Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required if
            sampling="healpix".  Defaults to None.

        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.

        filters (Tuple[jnp.ndarray], optional): Precomputed wavelet filters. Defaults to None.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

    Returns:
        jnp.ndarray: Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

    Notes:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
        [2] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint arXiv:1509.06749 (2015).
    """
    if precomps is None:
        precomps = construct.generate_wigner_precomputes(
            L, N, J_min, lam, sampling, nside, True, reality
        )

    J = samples.j_max(L, lam)
    Ls = samples.scal_bandlimit(L, J_min, lam, True)
    flm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)

    f_scal_lm = s2fft.forward_jax(f_scal, Ls, spin, nside, sampling, reality)

    # Sum the all wavelet wigner coefficients for each lmn
    # Note that almost the entire compute is concentrated at the highest J
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = samples.LN_j(L, j, N, lam, True)
        temp = s2fft.wigner.forward_jax(
            f_wav[j - J_min], Lj, Nj, nside, sampling, reality, precomps[j - J_min], L0j
        )
        flm = flm.at[L0j:Lj, L - Lj : L - 1 + Lj].add(
            jnp.einsum(
                "ln,nlm->lm",
                filters[0][j, L0j:Lj, L - Nj : L - 1 + Nj : 2],
                temp[::2, L0j:, :],
                optimize=True,
            )
        )

    # Sum the all scaling harmonic coefficients for each lm
    phi = filters[1][:Ls] * jnp.sqrt(4 * jnp.pi / (2 * jnp.arange(Ls) + 1))
    flm = flm.at[:Ls, L - Ls : L - 1 + Ls].add(
        jnp.einsum("lm,l->lm", f_scal_lm, phi, optimize=True)
    )
    return s2fft.inverse_jax(flm, L, spin, nside, sampling, reality)


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8))
def analysis(
    f: jnp.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    filters: Tuple[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None,
) -> Tuple[jnp.ndarray]:
    r"""Wavelet analysis from pixel space to wavelet space for complex signals.

    Args:
        f (jnp.ndarray): Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

        L (int): Harmonic bandlimit.

        N (int, optional): Upper azimuthal band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

        spin (int, optional): Spin (integer) of input signal. Defaults to 0.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss", "dh", "healpix"}. Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required if sampling="healpix".  Defaults
            to None.

        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.

        filters (Tuple[jnp.ndarray], optional): Precomputed wavelet filters. Defaults to None.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

    Returns:
        f_wav (jnp.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (jnp.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.
    """
    if precomps is None:
        precomps = construct.generate_wigner_precomputes(
            L, N, J_min, lam, sampling, nside, False, reality
        )
    J = samples.j_max(L, lam)
    Ls = samples.scal_bandlimit(L, J_min, lam, True)
    wav_lm = jnp.einsum(
        "jln, l->jln",
        jnp.conj(filters[0]),
        8 * jnp.pi**2 / (2 * jnp.arange(L) + 1),
        optimize=True,
    )

    flm = s2fft.forward_jax(f, L, spin, nside, sampling, reality)

    # Project all wigner coefficients for each lmn onto wavelet coefficients
    # Note that almost the entire compute is concentrated at the highest J
    f_wav = []
    f_wav_lmn = samples.construct_flmn_jax(L, N, J_min, J, lam, True)
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = samples.LN_j(L, j, N, lam, True)
        f_wav_lmn[j - J_min] = (
            f_wav_lmn[j - J_min]
            .at[::2, L0j:]
            .add(
                jnp.einsum(
                    "lm,ln->nlm",
                    flm[L0j:Lj, L - Lj : L - 1 + Lj],
                    wav_lm[j, L0j:Lj, L - Nj : L - 1 + Nj : 2],
                    optimize=True,
                )
            )
        )

        f_wav.append(
            s2fft.wigner.inverse_jax(
                f_wav_lmn[j - J_min],
                Lj,
                Nj,
                nside,
                sampling,
                reality,
                precomps[j - J_min],
                L0j,
            )
        )

    # Project all harmonic coefficients for each lm onto scaling coefficients
    phi = filters[1][:Ls] * jnp.sqrt(4 * jnp.pi / (2 * jnp.arange(Ls) + 1))
    temp = jnp.einsum("lm,l->lm", flm[:Ls, L - Ls : L - 1 + Ls], phi, optimize=True)

    # Handle edge case
    if Ls == 1:
        f_scal = temp * jnp.sqrt(1 / (4 * jnp.pi))
    else:
        f_scal = s2fft.inverse_jax(temp, Ls, spin, nside, sampling, reality)
    return f_wav, f_scal


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8))
def flm_to_analysis(
    flm: jnp.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    J_max: int = None,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    filters: Tuple[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None,
) -> Tuple[jnp.ndarray]:
    r"""Wavelet analysis from pixel space to wavelet space for complex signals.

    Args:
        f (jnp.ndarray): Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

        L (int): Harmonic bandlimit.

        N (int, optional): Upper azimuthal band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        J_max (int, optional): Highest frequency wavelet scale to be used. Defaults to None.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss", "dh", "healpix"}. Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required if sampling="healpix".  Defaults
            to None.

        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.

        filters (jnp.ndarray, optional): Precomputed wavelet filters. Defaults to None.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

    Returns:
        f_wav (jnp.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.
    """
    if precomps is None:
        precomps = construct.generate_wigner_precomputes(
            L, N, J_min, lam, sampling, nside, False, reality
        )

    J = J_max if J_max is not None else samples.j_max(L, lam)
    wav_lm = jnp.einsum(
        "jln, l->jln",
        jnp.conj(filters),
        8 * jnp.pi**2 / (2 * jnp.arange(L) + 1),
        optimize=True,
    )

    # Project all wigner coefficients for each lmn onto wavelet coefficients
    # Note that almost the entire compute is concentrated at the highest J
    f_wav = []
    f_wav_lmn = samples.construct_flmn_jax(L, N, J_min, J, lam, True)
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = samples.LN_j(L, j, N, lam, True)
        f_wav_lmn[j - J_min] = (
            f_wav_lmn[j - J_min]
            .at[::2, L0j:]
            .add(
                jnp.einsum(
                    "lm,ln->nlm",
                    flm[L0j:Lj, L - Lj : L - 1 + Lj],
                    wav_lm[j, L0j:Lj, L - Nj : L - 1 + Nj : 2],
                    optimize=True,
                )
            )
        )

        f_wav.append(
            s2fft.wigner.inverse_jax(
                f_wav_lmn[j - J_min],
                Lj,
                Nj,
                nside,
                sampling,
                reality,
                precomps[j - J_min],
                L0j,
            )
        )

    return f_wav
