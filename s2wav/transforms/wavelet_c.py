from jax import jit
import jax.numpy as jnp
from typing import Tuple, List
from functools import partial
import s2fft
from s2fft.transforms.c_backend_spherical import ssht_forward, ssht_inverse
from s2wav import samples


def synthesis(
    f_wav: jnp.ndarray,
    f_scal: jnp.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    sampling: str = "mw",
    reality: bool = False,
    filters: Tuple[jnp.ndarray] = None,
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

        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.

        filters (Tuple[jnp.ndarray], optional): Precomputed wavelet filters. Defaults to None.

    Raises:
        ValueError: If healpix sampling is provided to SSHT C backend.

    Returns:
        jnp.ndarray: Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

    Notes:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
        [2] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint arXiv:1509.06749 (2015).
    """
    if sampling.lower() == "healpix":
        raise ValueError("SSHT C backend does not support healpix sampling.")
    ssht_sampling = ["mw", "mwss", "dh", "gl"].index(sampling.lower())

    J = samples.j_max(L, lam)
    Ls = samples.scal_bandlimit(L, J_min, lam, True)

    f_scal_lm = ssht_forward(
        f_scal.real if reality else f_scal, Ls, spin, reality, ssht_sampling
    )
    f_wav_lmn = []
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = samples.LN_j(L, j, N, lam, True)
        f_wav_lmn.append(
            s2fft.wigner.forward_jax_ssht(
                f_wav[j - J_min], Lj, Nj, L0j, sampling, reality
            )
        )

    flm = _sum_over_wavelet_and_scaling(
        f_wav_lmn, f_scal_lm, L, N, J_min, J, lam, filters
    )
    return ssht_inverse(flm, L, spin, reality, ssht_sampling)


def analysis(
    f: jnp.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    sampling: str = "mw",
    reality: bool = False,
    filters: Tuple[jnp.ndarray] = None,
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

        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.

        filters (Tuple[jnp.ndarray], optional): Precomputed wavelet filters. Defaults to None.

    Returns:
        f_wav (jnp.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (jnp.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.
    """
    if sampling.lower() == "healpix":
        raise ValueError("SSHT C backend does not support healpix sampling.")
    ssht_sampling = ["mw", "mwss", "dh", "gl"].index(sampling.lower())

    J = samples.j_max(L, lam)
    Ls = samples.scal_bandlimit(L, J_min, lam, True)

    flm = ssht_forward(f, L, spin, reality, ssht_sampling)
    f_wav_lmn = _generate_and_apply_wavelets(flm, L, N, J_min, J, lam, filters[0])
    flm_scal = _generate_and_apply_scaling(flm, L, J_min, lam, filters[1])

    f_wav = []
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = samples.LN_j(L, j, N, lam, True)
        f_wav.append(
            s2fft.wigner.inverse_jax_ssht(
                f_wav_lmn[j - J_min], Lj, Nj, L0j, sampling, reality
            )
        )

    if Ls == 1:
        f_scal = flm_scal * jnp.sqrt(1 / (4 * jnp.pi))
    else:
        f_scal = ssht_inverse(flm_scal, Ls, spin, reality, ssht_sampling)
    return f_wav, f_scal


def flm_to_analysis(
    flm: jnp.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    J_max: int = None,
    lam: float = 2.0,
    sampling: str = "mw",
    reality: bool = False,
    filters: Tuple[jnp.ndarray] = None,
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

        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.

        filters (jnp.ndarray, optional): Precomputed wavelet filters. Defaults to None.

    Returns:
        f_wav (jnp.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.
    """
    J = J_max if J_max is not None else samples.j_max(L, lam)
    f_wav_lmn = _generate_and_apply_wavelets(flm, L, N, J_min, J, lam, filters)

    f_wav = []
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = samples.LN_j(L, j, N, lam, True)
        f_wav.append(
            s2fft.wigner.inverse_jax_ssht(
                f_wav_lmn[j - J_min], Lj, Nj, L0j, sampling, reality
            )
        )

    return f_wav


@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def _generate_and_apply_wavelets(
    flm: jnp.ndarray,
    L: int,
    N: int,
    J_min: int,
    J: int,
    lam: float = 2.0,
    filters: jnp.ndarray = None,
) -> jnp.ndarray:
    """Private internal function which generates and applies wavelet filters."""
    # f_wav = samples.construct_f_jax(L, J_min, J, lam)
    f_wav_lmn = samples.construct_flmn_jax(L, N, J_min, J, lam, True)

    wav_lm = jnp.einsum(
        "jln, l->jln",
        jnp.conj(filters),
        8 * jnp.pi**2 / (2 * jnp.arange(L) + 1),
        optimize=True,
    )

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
    return f_wav_lmn


@partial(jit, static_argnums=(1, 2, 3))
def _generate_and_apply_scaling(
    flm: jnp.ndarray,
    L: int,
    J_min: int = 0,
    lam: float = 2.0,
    filters: jnp.ndarray = None,
) -> jnp.ndarray:
    """Private internal function which generates and applies scaling filter."""
    Ls = samples.scal_bandlimit(L, J_min, lam, True)
    phi = filters[:Ls] * jnp.sqrt(4 * jnp.pi / (2 * jnp.arange(Ls) + 1))
    return jnp.einsum("lm,l->lm", flm[:Ls, L - Ls : L - 1 + Ls], phi, optimize=True)


@partial(jit, static_argnums=(2, 3, 4, 5, 6))
def _sum_over_wavelet_and_scaling(
    f_wav_lmn: jnp.ndarray,
    f_scal_lm: jnp.ndarray,
    L: int,
    N: int,
    J_min: int,
    J: int,
    lam: float = 2.0,
    filters: Tuple[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Private internal function which sums over wavelet and scaling coefficients."""
    Ls = samples.scal_bandlimit(L, J_min, lam, True)
    flm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = samples.LN_j(L, j, N, lam, True)
        flm = flm.at[L0j:Lj, L - Lj : L - 1 + Lj].add(
            jnp.einsum(
                "ln,nlm->lm",
                filters[0][j, L0j:Lj, L - Nj : L - 1 + Nj : 2],
                f_wav_lmn[j - J_min][::2, L0j:, :],
                optimize=True,
            )
        )

    phi = filters[1][:Ls] * jnp.sqrt(4 * jnp.pi / (2 * jnp.arange(Ls) + 1))
    flm = flm.at[:Ls, L - Ls : L - 1 + Ls].add(
        jnp.einsum("lm,l->lm", f_scal_lm, phi, optimize=True)
    )
    return flm
