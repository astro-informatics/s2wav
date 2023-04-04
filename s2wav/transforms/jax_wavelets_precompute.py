from jax import jit, config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from s2wav.utils import shapes
from functools import partial
from typing import Tuple, List
from s2fft.precompute_transforms.construct import wigner_kernel, spin_spherical_kernel
from s2fft.precompute_transforms import wigner, spherical


@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8))
def generate_precomputes(
    L: int,
    N: int,
    J_min: int = 0,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
    reality: bool = False,
    multiresolution: bool = False,
) -> List[jnp.ndarray]:
    r"""Generates a list of precompute arrays associated with the underlying Wigner
    transforms.

    Args:
        L (int): Harmonic bandlimit.

        N (int, optional): Upper azimuthal band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 1.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss", "dh",
            "healpix"}. Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required if sampling="healpix".  Defaults
            to None.

        forward (bool, optional): _description_. Defaults to False.

        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        List[jnp.ndarray]: Precomputed recursion arrays for underlying Wigner transforms.
    """
    precomps = []
    J = shapes.j_max(L, lam)
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = shapes.LN_j(L, j, N, lam, multiresolution)
        precomps.append(
            wigner_kernel(Lj, Nj, reality, sampling, nside, forward)
        )
    Ls = shapes.scal_bandlimit(L, J_min, lam, multiresolution)
    precompute_scaling = spin_spherical_kernel(Ls, 0, reality, sampling, nside, forward)
    precompute_full = spin_spherical_kernel(L, 0, reality, sampling, nside, not forward)
    return precompute_full, precompute_scaling, precomps


def synthesis(
    f_wav: jnp.ndarray,
    f_scal: jnp.ndarray,
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
    filters: Tuple[jnp.ndarray] = None,
    spmd: bool = False,
    precomps: List[List[jnp.ndarray]] = None,
) -> jnp.ndarray:
    r"""Computes the synthesis directional wavelet transform [1,2].
    Specifically, this transform synthesises the signal :math:`_{s}f(\omega) \in \mathbb{S}^2` by summing the contributions from wavelet and scaling coefficients in harmonic space, see equation 27 from `[2] <https://arxiv.org/pdf/1509.06749.pdf>`_.
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

        spin0 (int, optional): Spin (integer) of output signal. Defaults to 0.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss", "dh",
            "healpix"}. Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required if
            sampling="healpix".  Defaults to None.

        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

        filters (Tuple[jnp.ndarray], optional): Precomputed wavelet filters. Defaults to None.

        spmd (bool, optional): Whether to map compute over multiple devices. Currently this
            only maps over all available devices, and is only valid for JAX implementations.
            Defaults to False.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

    Raises:
        AssertionError: Shape of wavelet/scaling coefficients incorrect.

    Returns:
        jnp.ndarray: Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

    Notes:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
        [2] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint arXiv:1509.06749 (2015).
    """
    if precomps == None:
        raise ValueError("Must provide precomputed kernels for this transform!")

    J = shapes.j_max(L, lam)
    Ls = shapes.scal_bandlimit(L, J_min, lam, multiresolution)
    flm = jnp.zeros((L, 2 * L - 1), dtype=jnp.complex128)
    f_scal_lm = spherical.forward_transform_jax(f_scal, precomps[1], Ls, sampling, reality, spin, nside)

    # Sum the all wavelet wigner coefficients for each lmn
    # Note that almost the entire compute is concentrated at the highest J
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = shapes.LN_j(L, j, N, lam, multiresolution)
        spmd_iter = spmd if N == Nj else False
        temp = wigner.forward_transform_jax(f_wav[j - J_min], precomps[2][j-J_min], Lj, Nj, sampling, reality, nside)
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
    return spherical.inverse_transform_jax(flm, precomps[0], L, sampling, reality, spin, nside)


def analysis(
    f: jnp.ndarray,
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
    filters: Tuple[jnp.ndarray] = None,
    spmd: bool = False,
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

        spin0 (int, optional): Spin (integer) of output signal. Defaults to 0.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss", "dh", "healpix"}. Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required if sampling="healpix".  Defaults
            to None.

        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

        filters (Tuple[jnp.ndarray], optional): Precomputed wavelet filters. Defaults to None.

        spmd (bool, optional): Whether to map compute over multiple devices. Currently this
            only maps over all available devices, and is only valid for JAX implementations.
            Defaults to False.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

    Returns:
        f_wav (jnp.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (jnp.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.
    """
    if precomps == None:
        raise ValueError("Must provide precomputed kernels for this transform!")

    J = shapes.j_max(L, lam)
    Ls = shapes.scal_bandlimit(L, J_min, lam, multiresolution)

    f_wav_lmn = shapes.construct_flmn_jax(L, N, J_min, lam, multiresolution)
    f_wav = shapes.construct_f_jax(
        L, N, J_min, lam, sampling, nside, multiresolution
    )

    wav_lm = jnp.einsum(
        "jln, l->jln",
        jnp.conj(filters[0]),
        8 * jnp.pi**2 / (2 * jnp.arange(L) + 1),
        optimize=True,
    )

    flm = spherical.forward_transform_jax(f, precomps[0], L, sampling, reality, spin, nside)
    # Project all wigner coefficients for each lmn onto wavelet coefficients
    # Note that almost the entire compute is concentrated at the highest J
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = shapes.LN_j(L, j, N, lam, multiresolution)
        spmd_iter = spmd if N == Nj else False
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

        f_wav[j-J_min] = wigner.inverse_transform_jax(f_wav_lmn[j - J_min], precomps[2][j - J_min], Lj, Nj, sampling, reality, nside)

    # Project all harmonic coefficients for each lm onto scaling coefficients
    phi = filters[1][:Ls] * jnp.sqrt(4 * jnp.pi / (2 * jnp.arange(Ls) + 1))
    temp = jnp.einsum(
        "lm,l->lm", flm[:Ls, L - Ls : L - 1 + Ls], phi, optimize=True
    )
    # Handle edge case
    if Ls == 1:
        f_scal = temp * jnp.sqrt(1 / (4 * jnp.pi))
    else:
        f_scal = spherical.inverse_transform_jax(temp, precomps[1], Ls, sampling, reality, spin, nside)
    return f_wav, f_scal
