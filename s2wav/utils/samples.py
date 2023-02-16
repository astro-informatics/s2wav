import numpy as np


def nphi(L: int, sampling: str = "mw") -> int:
    """Computes the number of :math:`\phi`.

    Args:
        L (int): Upper harmonic band-limit.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}. Defaults to "mw".

    Raise:
        ValueError: Sampling scheme not recognised.

    Returns:
        int: Number of phi samples.
    """
    if sampling.lower() == "mw":
        return 2 * L - 1
    elif sampling.lower() == "mwss":
        return 2 * L
    else:
        raise ValueError(f"Sampling scheme {sampling} not supported.")


def ntheta(L: int, sampling: str = "mw") -> int:
    r"""Computes the number of :math:`\theta` samples.

    Args:
        L (int): Harmonic band-limit.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}. Defaults to "mw".

    Raise:
        ValueError: Sampling scheme not recognised.

    Returns:
        int: Number of theta samples.

    Note:
        Computes the number of samples :math:`\in [0, \pi]`, *not* over extended domain.
    """
    if sampling.lower() == "mw":
        return L
    elif sampling.lower() == "mwss":
        return L + 1
    else:
        raise ValueError(f"Sampling scheme {sampling} not supported.")


def j_bandlimit(j: int, lam: float = 2.0, kernel: str = "s2dw") -> int:
    r"""Computes the band-limit of a specific wavelet scale.

    Args:
        j (int): Wavelet scale to consider.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

        kernel (str, optional): The wavelet type from {"s2dw"}. Defaults to "s2dw".

    Raises:
        ValueError: Kernel type not supported.

    Returns:
        int: Harmonic bandlimit for scale :math:`j` with scaling :math:`\lambda`.
    """
    if kernel.lower() == "s2dw":
        return np.ceil(lam ** (j + 1))
    else:
        raise ValueError(f"Kernel type {kernel} not supported!")


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


def n_px(L: int, sampling: str = "mw") -> int:
    r"""Returns the number of spherical pixels for a given sampling scheme.

    Args:
        L (int): Harmonic band-limit.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}. Defaults to "mw".

    Returns:
        int: Total number of pixels.
    """
    return nphi(L, sampling) * ntheta(L, sampling)


def n_lm(L: int) -> int:
    """Returns the number of harmonic coefficients at bandlimit L.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        int: Total number of harmonic coefficients.
    """
    return L * L


def n_lm_scal(
    L: int,
    J_min: int = 0,
    lam: float = 2.0,
    kernel: str = "s2dw",
    upsample: bool = True,
) -> int:
    r"""Computes the total number of harmonic coefficients for scaling kernels :math:`\Phi_{\el m}`

    Args:
        L (int): Harmonic band-limit.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

        kernel (str, optional): The wavelet type from {"s2dw"}. Defaults to "s2dw".

        upsample (bool, optional): Whether to store the scales at :math:`j_{\text{max}}` resolution
            or its own resolution. Defaults to True.

    Returns:
        int: Number of harmonic coefficients for scaling kernels :math:`\Phi_{\el m}`.
    """
    if upsample == True:
        bandlimit = L
    else:
        bandlimit = min(j_bandlimit(J_min - 1, lam, kernel), L)

    return bandlimit * bandlimit


def n_lmn_wav(
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    kernel: str = "s2dw",
    storage: str = "padded",
    reality: bool = False,
    upsample: bool = True,
) -> int:
    r"""Computes the total number of Wigner coefficients for directional wavelet kernels :math:`\Psi^j_{\el n}`.

    Calls upon functions originally defined in so3, which have been defined explicitly for the case needed here.

    Args:
        L (int): Harmonic band-limit.

        N (int, optional): Upper azimuthal band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

        kernel (str, optional): The wavelet type from {"s2dw"}. Defaults to "s2dw".

        storage (str, optional): The type of storage from {"padded","compact"}. Defaults to "padded".

        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.

        upsample (bool, optional): Whether to store the scales at :math:`j_{\text{max}}` resolution
            or its own resolution. Defaults to True.

    Raises:
        ValueError: Storage method not recognised.

    Returns:
        int: Total number of Wigner space wavelet coefficients :math:`\Psi^j_{\el n}`.
    """

    J = j_max(L, lam)
    el = L
    total = 0

    for j in range(J_min, J + 1):
        if not upsample:
            el = min(j_bandlimit(j, lam, kernel), L)
            N = min(N, el)

        if storage.lower() == "padded":
            if reality:
                sampling_flmn_size = N * el * el
            else:
                sampling_flmn_size = (2 * N - 1) * el * el
        elif storage.lower() == "compact":
            if reality:
                sampling_flmn_size = (
                    N * (6 * el * el - (N - 1) * (2 * N - 1)) / 6
                )
            else:
                sampling_flmn_size = (
                    (2 * N - 1) * (3 * el * el - N * (N - 1)) / 3
                )
        else:
            raise ValueError(f"Storage method {storage} not recognised.")

        total += sampling_flmn_size

    return total


def n_gamma(N: int, steerable: bool = False) -> int:
    r"""Computes the number of :math:`\gamma` samples for a given sampling scheme

    Args:
        N (int): Upper orientational band-limit. Only flmn with :math:`n < N` will be stored.

        steerable (bool, optional): Indicates that the signal is steerable. Defaults to False.

    Returns:
        int: Number of :math:`\gamma` samples.
    """
    if steerable:
        return N
    else:
        return 2 * N - 1


def n_scal(
    L: int,
    J_min: int = 0,
    lam: float = 2.0,
    kernel: str = "s2dw",
    sampling: str = "mw",
    upsample: bool = True,
) -> int:
    r"""Computes the number of pixel-space samples for scaling kernels :math:`\Phi`.

    Args:
        L (int): Harmonic band-limit.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

        kernel (str, optional): The wavelet type from {"s2dw"}. Defaults to "s2dw".

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}. Defaults to "mw".

        upsample (bool, optional): Whether to store the scales at :math:`j_{\text{max}}` resolution
            or its own resolution. Defaults to True.

    Returns:
       int: Total number of pixel-space samples for scaling kernels :math:`\Phi`.
    """
    if upsample == True:
        el = L
    else:
        el = min(j_bandlimit(J_min - 1, lam, kernel), L)
    return nphi(el, sampling) * ntheta(el, sampling)


def n_wav(
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    kernel: str = "s2dw",
    sampling: str = "mw",
    steerable: bool = False,
    upsample: bool = True,
) -> int:
    r"""Computes the total number of pixel-space samples for directional wavelet kernels :math:`\Psi`.

    Calls upon functions originally defined in so3, which have been defined explicitly for the case needed here.

    Args:
        L (int): Harmonic band-limit.

        N (int, optional): Upper orientational band-limit. Only flmn with :math:`n < N` will be stored.
            Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

        kernel (str, optional): The wavelet type from {"s2dw"}. Defaults to "s2dw".

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}. Defaults to "mw".

        steerable (bool, optional): Indicates that the signal is steerable. Defaults to False.

        upsample (bool, optional): Whether to store the scales at :math:`j_{\text{max}}` resolution
            or its own resolution. Defaults to True.

    Returns:
        int: Total number of pixel-space samples for :math:`\Psi`.
    """
    J = j_max(L, lam)
    el = L

    total = 0

    for j in range(J_min, J + 1):
        if not upsample:
            el = min(j_bandlimit(j, lam, kernel), L)

        if sampling.lower() == "mw":
            sampling_nalpha = 2 * el - 1
            sampling_nbeta = el
        else:
            sampling_nalpha = 2 * el
            sampling_nbeta = el + 1

        if steerable:
            sampling_ngamma = N
        else:
            sampling_ngamma = 2 * N - 1

        sampling_f_size = sampling_nalpha * sampling_nbeta * sampling_ngamma
        total += sampling_f_size
    return total


def n_wav_j(
    j: int,
    L: int,
    N: int = 1,
    lam: float = 2.0,
    kernel: str = "s2dw",
    sampling: str = "mw",
    steerable: bool = False,
    upsample: bool = True,
) -> int:
    r"""Number of directional wavelet pixel-space coefficients for a specific scale :math:`j`.

    Calls upon functions originally defined in so3, which have been defined explicitly for the case needed here.

    Args:
        j (int): Wavelet scale under consideration.

        L (int): Harmonic band-limit.

        N (int, optional): Upper orientational band-limit. Only flmn with :math:`n < N` will be stored.
            Defaults to 1.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

        kernel (str, optional): The wavelet type from {"s2dw"}. Defaults to "s2dw".

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}. Defaults to "mw".

        steerable (bool, optional): Indicates that the signal is steerable. Defaults to False.

        upsample (bool, optional): Whether to store the scales at :math:`j_{\text{max}}` resolution
            or its own resolution. Defaults to True.

    Returns:
        int: The number of wavelet pixel-space coefficients at scale :math:`j`.
    """
    el = L
    if not upsample:
        el = min(j_bandlimit(j, lam, kernel), L)

    if sampling.lower() == "mw":
        sampling_nalpha = 2 * el - 1
        sampling_nbeta = el
    else:
        sampling_nalpha = 2 * el
        sampling_nbeta = el + 1

    if steerable:
        sampling_ngamma = N
    else:
        sampling_ngamma = 2 * N - 1

    sampling_f_size = sampling_nalpha * sampling_nbeta * sampling_ngamma

    return sampling_f_size


def elm2ind(el: int, m: int) -> int:
    r"""Convert from spherical harmonic 2D indexing of :math:`(\ell,m)` to 1D index.
    1D index is defined by `ell**2 + ell + m`.

    Warning:
        Note that 1D storage of spherical harmonic coefficients is *not* the default.

    Args:
        el (int): Harmonic degree :math:`\ell`.

        m (int): Harmonic order :math:`m`.

    Returns:
        int: Corresponding 1D index value.
    """

    return el**2 + el + m


def elmn2ind(el: int, m: int, n: int, L: int, N: int = 1) -> int:
    r"""Convert from Wigner space 3D indexing of :math:`(\ell,m, n)` to 1D index.

    Args:
        el (int): Harmonic degree :math:`\ell`.

        m (int): Harmonic order :math:`m`.

        n (int): Directional order :math:`n`.

        L (int): Harmonic band-limit.

        N (int, optional): Number of Fourier coefficients for tangent plane rotations (i.e. directionality). Defaults to 1.

    Returns:
        int: Corresponding 1D index in Wigner space.
    """
    n_offset = (N - 1 + n) * L * L
    el_offset = el * el
    return n_offset + el_offset + el + m
