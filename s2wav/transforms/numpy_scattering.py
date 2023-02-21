import numpy as np
from typing import List, Tuple
from s2wav.transforms import numpy_wavelets as wavelets
from s2wav.filter_factory import filters
from s2wav.utils import shapes
from s2fft import base_transforms as base


def scatter(
    f: np.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    nlayers: int = None,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    multiresolution: bool = False,
    filter: Tuple[np.ndarray] = None,
) -> List[np.ndarray]:
    """Computes the scattering transform for descending by one nodes alone.

    Following equations outlined in section 3.2 of [1], recursively compute wavelet
    transform each time passed through the activation function (in this case the absolute
    value, 'modulus' operator) and store the scaling coefficients.

    Args:
        f (np.ndarray): _description_
        L (int): _description_
        N (int, optional): _description_. Defaults to 1.
        J_min (int, optional): _description_. Defaults to 1.
        lam (float, optional): _description_. Defaults to 2.0.
        nlayers (int, optional): _description_. Defaults to None.
        sampling (str, optional): _description_. Defaults to "mw".
        nside (int, optional): _description_. Defaults to None.
        reality (bool, optional): _description_. Defaults to False.
        multiresolution (bool, optional): _description_. Defaults to False.
        filters (Tuple[jnp.ndarray], optional): Precomputed wavelet filters. Defaults to None.

    Raises:
        ValueError: Number of layers is larger than the number of available wavelet scales.

    Returns:
        List[np.ndarray]: _description_

    Notes:
        [1] McEwen et al, Scattering networks on the sphere for scalable and
            rotationally equivariant spherical CNNs, ICLR 2022.
    """
    scattering_coefficients = []
    J = shapes.j_max(L, lam)

    if nlayers is None:
        nlayers = J - J_min
    if nlayers > J - J_min:
        raise ValueError(
            f"Number of scattering layers {nlayers} is larger than the number of available wavelet scales {J-J_min}."
        )

    if filter == None:
        wav_lm, scal_l = filters.filters_directional_vectorised(
            L, N, J_min, lam, 0, 0
        )
        wav_lm = np.einsum(
            "jln, l->jln",
            np.conj(wav_lm),
            8 * np.pi**2 / (2 * np.arange(L) + 1),
        )

    # Perform the first wavelet transform for all scales
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
                temp, f_scal = wavelets._analysis_scattering(
                    f_wav[j - J_min][n],
                    L,
                    j - layer,
                    1,
                    J_min,
                    lam,
                    sampling=sampling,
                    nside=nside,
                    reality=reality,
                    multiresolution=multiresolution,
                    filters=(wav_lm[j - J_min - layer - 1], scal_l),
                )
                wavelet_coefficients.append(temp[0])
                scattering_coefficients.append(f_scal)
            f_wav[j - J_min] = np.array(temp)

        j_iter += 1

    return np.array(scattering_coefficients)


def _analysis_scattering(
    f: np.ndarray,
    Lin: int,
    j: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    spin0: int = 0,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    multiresolution: bool = False,
    filters: Tuple[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns:
        f_wav (np.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (np.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.
    """
    L = shapes.wav_j_bandlimit(Lin, j, lam, multiresolution)
    Ls = shapes.scal_bandlimit(L, J_min, lam, multiresolution)
    flm = base.spherical.forward(f, L, spin, sampling, nside, reality)

    # Compute the scaling coefficients as usual.
    phi = filters[1][:Ls] * np.sqrt(4 * np.pi / (2 * np.arange(Ls) + 1))
    f_scal = base.spherical.inverse(
        np.einsum("lm,l->lm", flm[:Ls, L - Ls : L - 1 + Ls], phi),
        Ls,
        spin,
        sampling,
        nside,
        reality,
    )
    # Get shapes for scale j - 1.
    Lj, Nj, L0j = shapes.LN_j(Lin, j - 1, 1, lam, multiresolution)
    if j == J_min:
        return np.zeros((2 * Nj - 1, Lj, 2 * Lj - 1)), f_scal

    f_wav_lmn = shapes.construct_flmn(Lj, Nj, J_min, lam, multiresolution, True)

    # Only compute the wavelet coefficients for descending by 1.
    f_wav_lmn[::2, L0j:] = np.einsum(
        "lm,ln->nlm",
        flm[L0j:Lj, L - Lj : L - 1 + Lj],
        filters[0][L0j:Lj, L - Nj : L - 1 + Nj : 2],
    )
    f_wav = base.wigner.inverse(
        f_wav_lmn, Lj, Nj, L0j, sampling, reality, nside
    )
    return np.abs(f_wav), f_scal
