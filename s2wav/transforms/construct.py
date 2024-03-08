import jax.numpy as jnp
from typing import  List
import s2fft
from s2fft.precompute_transforms.construct import (
    wigner_kernel_jax,
    spin_spherical_kernel_jax,
)
from s2wav import samples

def generate_full_precomputes(
    L: int,
    N: int,
    J_min: int = 0,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
    reality: bool = False,
    nospherical: bool = False,
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

        nospherical (bool, optional): Whether to only compute Wigner precomputes.
            Defaults to False.

    Returns:
        List[jnp.ndarray]: Precomputed recursion arrays for underlying Wigner transforms.
    """
    precomps = []
    J = samples.j_max(L, lam)
    for j in range(J_min, J):
        Lj, Nj, L0j = samples.LN_j(L, j, N, lam, True)
        precomps.append(wigner_kernel_jax(Lj, Nj, reality, sampling, nside, forward))
    Ls = samples.scal_bandlimit(L, J_min, lam, True)
    if nospherical:
        return [], [], precomps
    precompute_scaling = spin_spherical_kernel_jax(
        Ls, 0, reality, sampling, nside, forward
    )
    precompute_full = spin_spherical_kernel_jax(
        L, 0, reality, sampling, nside, not forward
    )
    return precompute_full, precompute_scaling, precomps

def generate_wigner_precomputes(
    L: int,
    N: int,
    J_min: int = 0,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    forward: bool = False,
    reality: bool = False
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

    Returns:
        List[jnp.ndarray]: Precomputed recursion arrays for underlying Wigner transforms.
    """
    precomps = []
    J = samples.j_max(L, lam)
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = samples.LN_j(L, j, N, lam, True)
        precomps.append(
            s2fft.generate_precomputes_wigner_jax(
                Lj, Nj, sampling, nside, forward, reality, L0j
            )
        )
    return precomps