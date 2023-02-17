from jax import jit, config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
from s2wav.utils import shapes
from s2wav.filter_factory import filters
import s2fft
from functools import partial
from typing import Tuple

@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
def analysis_transform_jax(
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
    filters: Tuple[jnp.ndarray, jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    Returns:
        f_wav (jnp.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (jnp.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.
    """
    J = shapes.j_max(L, lam)
    Ls = shapes.scal_bandlimit(L, J_min, lam, multiresolution)

    f_scal_lm = shapes.construct_flm_jax(L, J_min, lam, multiresolution)
    f_wav_lmn = shapes.construct_flmn_jax(L, N, J_min, lam, multiresolution)
    f_wav = shapes.construct_f_jax(L, N, J_min, lam, sampling, multiresolution)

    wav_lm = jnp.einsum(
        "jln, l->jln", jnp.conj(filters[0]), 8 * jnp.pi**2 / (2 * jnp.arange(L) + 1),
        optimize = True
    )

    flm = s2fft.forward_jax(f, L, spin, nside, sampling, reality)
    
    # Project all wigner coefficients for each lmn onto wavelet coefficients
    # Note that almost the entire compute is concentrated at the highest J
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = shapes.LN_j(L, j, N, lam, multiresolution)
        f_wav_lmn[j - J_min] = f_wav_lmn[j - J_min].at[::2, L0j:].set(jnp.einsum(
            "lm,ln->nlm",
            flm[L0j:Lj, L - Lj : L - 1 + Lj],
            wav_lm[j, L0j:Lj, L - Nj : L - 1 + Nj : 2],
            optimize = True
        ))
        f_wav[j - J_min] = s2fft.wigner.inverse_jax(
            f_wav_lmn[j - J_min], Lj, Nj, nside, sampling, reality, L_lower=L0j
        )

    # Project all harmonic coefficients for each lm onto scaling coefficients
    phi = filters[1][:Ls] * jnp.sqrt(4 * jnp.pi / (2 * jnp.arange(Ls) + 1))
    f_scal_lm = jnp.einsum("lm,l->lm", flm[:Ls, L - Ls : L - 1 + Ls], phi, optimize=True)

    return f_wav, s2fft.inverse_jax(
        f_scal_lm, Ls, spin, nside, sampling, reality
    )

if __name__ == "__main__":
    L = 8
    N = 3 
    f = jnp.zeros((L, 2*L-1), dtype=jnp.complex128)

    # Generate the directional wavelet kernels
    filter = filters.filters_directional_vectorised(L, N) 
    print(analysis_transform_jax(f, L, N, filters=filter))