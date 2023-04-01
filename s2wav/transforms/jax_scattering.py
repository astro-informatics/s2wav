from jax import config, jit
from functools import partial

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.lax as lax
from typing import List, Tuple
from s2wav.transforms import jax_wavelets as wavelets
from s2wav.utils import shapes
import s2fft


def scatter(
    f: jnp.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    nlayers: int = None,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    filters: Tuple[jnp.ndarray] = None,
    multiresolution: bool = False,
    spmd: bool = False,
    precomps: List[List[jnp.ndarray]] = None,
) -> List[jnp.ndarray]:

    if precomps == None:
        precomps = wavelets.generate_wigner_precomputes(
            L, N, J_min, lam, sampling, nside, False, reality, multiresolution
        )

    scattering_coefficients = []
    J = shapes.j_max(L, lam)

    if nlayers is None:
        nlayers = J - J_min

    # Weight filters a priori
    wav_lm = jnp.einsum(
        "jln, l->jln",
        jnp.conj(filters[0]),
        8 * jnp.pi**2 / (2 * jnp.arange(L) + 1),
        optimize=True,
    )
    scal_l = filters[1] * jnp.sqrt(4 * jnp.pi / (2 * jnp.arange(L) + 1))

    # Perform the first wavelet transform for all scales and directions
    f_wav, f_scal = wavelets.analysis(
        f,
        L,
        N,
        J_min,
        lam,
        sampling=sampling,
        nside=nside,
        reality=reality,
        multiresolution=multiresolution,
        filters=filters,
        spmd=spmd,
        precomps=precomps,
        scattering=True,
    )
    scattering_coefficients.append(f_scal)

    # Perform the subsequent wavelet transforms for only J-1th scale.
    j_iter = J_min
    for layer in range(nlayers):
        for j in range(j_iter, J + 1):
            _, Nj, _ = shapes.LN_j(L, j - layer - 1, N, lam, multiresolution)
            wavelet_coefficients = []
            for n in range(2 * Nj - 1):
                temp, f_scal = _analysis_scattering(
                    f_wav[j - J_min][n],
                    L,
                    j - layer,
                    J_min,
                    lam,
                    sampling=sampling,
                    nside=nside,
                    reality=reality,
                    multiresolution=multiresolution,
                    filters=(wav_lm[j - J_min - layer - 1], scal_l),
                    precomps=precomps[j - J_min - layer - 1],
                )
                wavelet_coefficients.append(temp[0])
                scattering_coefficients.append(f_scal)

            f_wav[j - J_min] = jnp.array(wavelet_coefficients)

        j_iter += 1

    return jnp.array(scattering_coefficients)


def _analysis_scattering(
    f: jnp.ndarray,
    Lin: int,
    j: int,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    spin0: int = 0,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    multiresolution: bool = False,
    filters: Tuple[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    L = shapes.wav_j_bandlimit(Lin, j, lam, multiresolution)
    Ls = shapes.scal_bandlimit(L, J_min, lam, multiresolution)
    flm = s2fft.forward_jax(f, L, spin, nside, sampling, reality)

    # Compute the scaling coefficients as usual.
    phi = jnp.einsum(
        "lm,l->lm",
        flm[:Ls, L - Ls : L - 1 + Ls],
        filters[1][:Ls],
        optimize=True,
    )

    # Handle edge case
    if Ls == 1:
        f_scal = phi * jnp.sqrt(1 / (4 * jnp.pi))
    else:
        f_scal = s2fft.inverse_jax(phi, Ls, spin, nside, sampling, reality)

    # Get shapes for scale j - 1.
    Lj, Nj, L0j = shapes.LN_j(Lin, j - 1, 1, lam, multiresolution)
    if j == J_min:
        return (
            jnp.zeros((2 * Nj - 1, Lj, 2 * Lj - 1)),
            jnp.real(f_scal) if reality else f_scal,
        )

    f_wav_lmn = shapes.construct_flmn_jax(
        Lj, Nj, J_min, lam, multiresolution, True
    )

    # Only compute the wavelet coefficients for descending by 1.
    f_wav_lmn = f_wav_lmn.at[::2, L0j:].set(
        jnp.einsum(
            "lm,ln->nlm",
            flm[L0j:Lj, L - Lj : L - 1 + Lj],
            filters[0][L0j:Lj, L - Nj : L - 1 + Nj : 2],
            optimize=True,
        )
    )
    f_wav = s2fft.wigner.inverse_jax(
        f_wav_lmn,
        Lj,
        Nj,
        nside,
        sampling,
        reality,
        precomps,
        spmd=False,
        L_lower=L0j,
    )

    return jnp.abs(f_wav), jnp.real(f_scal) if reality else f_scal


# ---------------------------------------------------------------
# Apr 1 2023
# ---------------------------------------------------------------


@partial(jit, static_argnums=(1, 2, 3))
def scatter_new(
    flm: jnp.ndarray,
    L: int,
    N: int,
    reality: bool = False,
    filters: Tuple[jnp.ndarray] = None,
    precomps: List[List[jnp.ndarray]] = None,
) -> List[jnp.ndarray]:

    if precomps == None:
        precomps = wavelets.generate_wigner_precomputes(
            L, N, 0, 2, "mw", None, False, reality, True
        )

    J = shapes.j_max(L, 2)

    W, S = wavelets.flm_to_analysis_full(
        flm,
        L,
        N,
        reality=reality,
        multiresolution=True,
        filters=filters,
        precomps=precomps,
    )

    scattering_coefficients = []
    scattering_coefficients.append(S)
    Njjprime = []
    for j in range(J + 1):
        Lj = shapes.wav_j_bandlimit(L, j, 2.0, True)
        Njjprime_for_j = []
        M_lm = jnp.zeros((2 * N - 1, Lj, 2 * Lj - 1), dtype=jnp.complex128)

        def harmonic_step_for_j(n, args):
            M_lm = args
            M_lm = M_lm.at[n].add(
                s2fft.forward_jax(
                    jnp.abs(W[j][n]),
                    Lj,
                    0,
                    reality=reality,
                )
            )
            return M_lm

        M_lm = lax.fori_loop(0, 2 * N - 1, harmonic_step_for_j, M_lm)

        for n in range(2 * N - 1):
            val, S = wavelets.flm_to_analysis_full(
                M_lm[n],
                Lj,
                N,
                J_max=j - 1,
                reality=reality,
                multiresolution=True,
                filters=(
                    filters[0][: j + 1, :Lj, L - Lj : L - 1 + Lj],
                    filters[1],
                ),
                precomps=precomps[:j],
            )
            scattering_coefficients.append(S)
            Njjprime_for_j.append(val)
        Njjprime.append(Njjprime_for_j)

    # Reorder and flatten Njjprime, convert to JAX arrays for C01/C11
    Njjprime_flat = []
    for j1 in range(J):
        Njjprime_flat_for_j2 = []
        for j2 in range(j1 + 1, J + 1):
            for n2 in range(2 * N - 1):
                for n1 in range(2 * N - 1):
                    Njjprime_flat_for_j2.append(Njjprime[j2][n2][j1][n1])
        Njjprime_flat.append(jnp.array(Njjprime_flat_for_j2))

    # Now should be indexed by scale up to J and we can perform 2nd layer
    for j in range(J):
        Lj = shapes.wav_j_bandlimit(L, j, 2.0, True)
        quads = s2fft.utils.quadrature_jax.quad_weights(Lj)
        S = jnp.einsum(
            "itp,t->i", jnp.abs(Njjprime_flat[j]), quads, optimize=True
        )

        scattering_coefficients.append(S)

    return jnp.concatenate(scattering_coefficients, axis=None)
