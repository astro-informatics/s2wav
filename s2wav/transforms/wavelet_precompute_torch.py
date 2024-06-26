import torch
from typing import Tuple, List
from s2fft.precompute_transforms import wigner, spherical
from s2wav import samples


def synthesis(
    f_wav: torch.tensor,
    f_scal: torch.tensor,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    filters: Tuple[torch.tensor] = None,
    precomps: List[List[torch.tensor]] = None,
) -> torch.tensor:
    r"""Computes the synthesis directional wavelet transform [1,2].
    Specifically, this transform synthesises the signal :math:`_{s}f(\omega) \in \mathbb{S}^2`
    by summing the contributions from wavelet and scaling coefficients in harmonic space,
    see equation 27 from `[2] <https://arxiv.org/pdf/1509.06749.pdf>`_.
    Args:
        f_wav (torch.tensor): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (torch.tensor): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.

        L (int): Harmonic bandlimit.

        N (int, optional): Upper azimuthal band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 1.

        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.

        spin (int, optional): Spin (integer) of input signal. Defaults to 0.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss", "dh",
            "healpix"}. Defaults to "mw".

        nside (int, optional): HEALPix Nside resolution parameter.  Only required if
            sampling="healpix".  Defaults to None.

        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.

        filters (Tuple[torch.tensor], optional): Precomputed wavelet filters. Defaults to None.

        precomps (List[torch.tensor]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

    Raises:
        AssertionError: Shape of wavelet/scaling coefficients incorrect.

    Returns:
        torch.tensor: Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

    Notes:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
        [2] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint arXiv:1509.06749 (2015).
    """
    if precomps == None:
        raise ValueError("Must provide precomputed kernels for this transform!")

    J = samples.j_max(L, lam)
    Ls = samples.scal_bandlimit(L, J_min, lam, True)
    flm = torch.zeros((L, 2 * L - 1), dtype=torch.complex128)
    f_scal_lm = spherical.forward_transform_torch(
        f_scal, precomps[1], Ls, sampling, reality, spin, nside
    )

    # Sum the all wavelet wigner coefficients for each lmn
    # Note that almost the entire compute is concentrated at the highest two scales.
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = samples.LN_j(L, j, N, lam, True)
        shift = 0 if j < J else -1
        temp = wigner.forward_transform_torch(
            f_wav[j - J_min],
            precomps[2][j - J_min + shift],
            Lj,
            Nj,
            sampling,
            reality,
            nside,
        )
        flm[L0j:Lj, L - Lj : L - 1 + Lj] += torch.einsum(
            "ln,nlm->lm",
            filters[0][j, L0j:Lj, L - Nj : L - 1 + Nj : 2],
            temp[::2, L0j:, :],
        )

    # Sum the all scaling harmonic coefficients for each lm
    phi = filters[1][:Ls] * torch.sqrt(
        4 * torch.pi / (2 * torch.arange(Ls, dtype=torch.float64) + 1)
    )
    flm[:Ls, L - Ls : L - 1 + Ls] += torch.einsum("lm,l->lm", f_scal_lm, phi)
    return spherical.inverse_transform_torch(
        flm, precomps[0], L, sampling, reality, spin, nside
    )


def analysis(
    f: torch.tensor,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    filters: Tuple[torch.tensor] = None,
    precomps: List[List[torch.tensor]] = None,
) -> Tuple[torch.tensor]:
    r"""Wavelet analysis from pixel space to wavelet space for complex signals.

    Args:
        f (torch.tensor): Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

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

        filters (Tuple[torch.tensor], optional): Precomputed wavelet filters. Defaults to None.

        precomps (List[torch.tensor]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

    Returns:
        f_wav (torch.tensor): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (torch.tensor): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.
    """
    if precomps == None:
        raise ValueError("Must provide precomputed kernels for this transform!")

    J = samples.j_max(L, lam)
    Ls = samples.scal_bandlimit(L, J_min, lam, True)

    f_wav_lmn = samples.construct_flmn_torch(L, N, J_min, J, lam, True)
    f_wav = samples.construct_f_torch(L, J_min, J, lam)

    wav_lm = torch.einsum(
        "jln, l->jln",
        torch.conj(filters[0]),
        8 * torch.pi**2 / (2 * torch.arange(L, dtype=torch.float64) + 1),
    )

    flm = spherical.forward_transform_torch(
        f, precomps[0], L, sampling, reality, spin, nside
    )
    # Project all wigner coefficients for each lmn onto wavelet coefficients
    # Note that almost the entire compute is concentrated at the highest J
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = samples.LN_j(L, j, N, lam, True)
        f_wav_lmn[j - J_min][::2, L0j:] += torch.einsum(
            "lm,ln->nlm",
            flm[L0j:Lj, L - Lj : L - 1 + Lj],
            wav_lm[j, L0j:Lj, L - Nj : L - 1 + Nj : 2],
        )

        shift = 0 if j < J else -1
        f_wav[j - J_min] = wigner.inverse_transform_torch(
            f_wav_lmn[j - J_min],
            precomps[2][j - J_min + shift],
            Lj,
            Nj,
            sampling,
            reality,
            nside,
        )

    # Project all harmonic coefficients for each lm onto scaling coefficients
    phi = filters[1][:Ls] * torch.sqrt(
        4 * torch.pi / (2 * torch.arange(Ls, dtype=torch.float64) + 1)
    )
    temp = torch.einsum("lm,l->lm", flm[:Ls, L - Ls : L - 1 + Ls], phi)

    # Handle edge case
    if Ls == 1:
        f_scal = temp * torch.sqrt(
            torch.tensor(1 / (4 * torch.pi), dtype=torch.float64)
        )
    else:
        f_scal = spherical.inverse_transform_torch(
            temp, precomps[1], Ls, sampling, reality, spin, nside
        )
    return f_wav, f_scal


def flm_to_analysis(
    flm: torch.tensor,
    L: int,
    N: int = 1,
    J_min: int = 0,
    J_max: int = None,
    lam: float = 2.0,
    sampling: str = "mw",
    nside: int = None,
    reality: bool = False,
    filters: Tuple[torch.tensor] = None,
    precomps: List[List[torch.tensor]] = None,
) -> Tuple[torch.tensor]:
    r"""Wavelet analysis from pixel space to wavelet space for complex signals.

    Args:
        f (torch.tensor): Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

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

        filters (Tuple[torch.tensor], optional): Precomputed wavelet filters. Defaults to None.

        precomps (List[torch.tensor]): Precomputed list of recursion coefficients. At most
            of length :math:`L^2`, which is a minimal memory overhead.

    Returns:
        f_wav (torch.tensor): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (torch.tensor): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.
    """
    if precomps == None:
        raise ValueError("Must provide precomputed kernels for this transform!")

    J = J_max if J_max is not None else samples.j_max(L, lam)

    f_wav_lmn = samples.construct_flmn_torch(L, N, J_min, J, lam, True)
    f_wav = samples.construct_f_torch(L, J_min, J, lam)

    wav_lm = torch.einsum(
        "jln, l->jln",
        torch.conj(filters),
        8 * torch.pi**2 / (2 * torch.arange(L, dtype=torch.float64) + 1),
    )

    # Project all wigner coefficients for each lmn onto wavelet coefficients
    # Note that almost the entire compute is concentrated at the highest J
    for j in range(J_min, J + 1):
        Lj, Nj, L0j = samples.LN_j(L, j, N, lam, True)
        f_wav_lmn[j - J_min][::2, L0j:] += torch.einsum(
            "lm,ln->nlm",
            flm[L0j:Lj, L - Lj : L - 1 + Lj],
            wav_lm[j, L0j:Lj, L - Nj : L - 1 + Nj : 2],
        )
        shift = 0 if j < J else -1
        f_wav[j - J_min] = wigner.inverse_transform_torch(
            f_wav_lmn[j - J_min],
            precomps[2][j - J_min + shift],
            Lj,
            Nj,
            sampling,
            reality,
            nside,
        )

    return f_wav
