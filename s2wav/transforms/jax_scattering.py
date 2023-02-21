from jax import config

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
    r"""Computes the scattering transform for descending by one nodes alone.

    Following equations outlined in section 3.2 of [1], recursively compute wavelet
    transform each time passed through the activation function (in this case the absolute
    value, 'modulus' operator) and store the scaling coefficients.

    Args:
        f (np.ndarray): Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

        L (int): Harmonic bandlimit.

        N (int, optional): Upper azimuthal band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.

        nlayers (int, optional): Total number of scattering layers. Defaults to None, in
            which case all paths which descend by 1 are included.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss", "dh",
            "healpix"}. Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required if
            sampling="healpix".  Defaults to None.

        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.

        filters (Tuple[jnp.ndarray], optional): Precomputed wavelet filters. Defaults to None.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

        spmd (bool, optional): Whether to map compute over multiple devices. Currently this
            only maps over all available devices, and is only valid for JAX implementations.
            Defaults to False.

        precomps (List[jnp.ndarray]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.


    Raises:
        ValueError: Number of layers is larger than the number of available wavelet scales.

        NotImplementedError: Filters not provided, and functionality to compute these in
            JAX is not yet implemented.

    Returns:
        List[np.ndarray]: List of scattering coefficients. Dimensionality of each scattering
            coefficien will depend on the selection of hyperparameters. In the most
            typical case (J_min = 0), each scattering coefficient is a single scalar value.

    Notes:
        [1] McEwen et al, Scattering networks on the sphere for scalable and
            rotationally equivariant spherical CNNs, ICLR 2022.
    """
    if precomps == None:
        precomps = wavelets.generate_wigner_precomputes(
            L, N, J_min, lam, sampling, nside, False, reality, multiresolution
        )

    scattering_coefficients = []
    J = shapes.j_max(L, lam)

    if nlayers is None:
        nlayers = J - J_min
    if nlayers > J - J_min:
        raise ValueError(
            f"Number of scattering layers {nlayers} is larger than the number of available wavelet scales {J-J_min}."
        )
    if filters == None:
        raise ValueError("Automatic filter computation not yet implemented!")

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
    r"""Wavelet analysis from pixel space to wavelet space for complex signals.

    Args:
        f (np.ndarray): Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

        L (int): Harmonic bandlimit.

        j (int): Wavelet scale.

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
        f_wav (np.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (np.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.
    """
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
