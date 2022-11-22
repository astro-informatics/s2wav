import numpy as np


def sampling_mw_ss_nphi(L: int) -> int:
    """Computes the number of phi samples for McEwen and Wiaux symmetric sampling.

    Args:
        L (int): Upper harmonic band-limit.

    Returns:
        nphi (int): Number of phi samples.
    """
    return 2 * L


def sampling_mw_nphi(L: int) -> int:
    """Computes the number of phi samples for McEwen and Wiaux sampling.

    Args:
        L (int): Upper harmonic band-limit.

    Returns:
        nphi (int): Number of phi samples.
    """
    return 2 * L - 1


def sampling_mw_ss_ntheta(L: int) -> int:
    r"""Computes the number of theta samples for McEwen and Wiaux symmetric sampling.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        ntheta (int): Number of theta samples.

    Note:
        Computes the number of samples :math:`\in [0, \pi]`, *not* over extended domain.
    """
    return L + 1


def sampling_mw_ntheta(L: int) -> int:
    r"""Computes the number of theta samples for McEwen and Wiaux sampling.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        ntheta (int): Number of theta samples.

    Note:
        Computes the number of samples :math:`\in [0, \pi]`, *not* over extended domain.
    """
    return L


def bandlimit(kernel: str, lam: float, j: int) -> int:
    r"""Computes the band-limit of a specific wavelet scale.

    Args:
        kernel (str): The wavelet type from {"s2dw"}.

        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.

        j (int): Wavelet scale.

    Raises:
        ValueError: Kernel type not supported.

    Returns:
        (int): Harmonic bandlimit for scale :math:`j` with scaling :math:`\lambda`.
    """
    if kernel.lower() == "s2dw":
        return np.ceil(lam ** (j + 1))
    else:
        raise ValueError(f"Kernel type {kernel} not supported!")


def L0(kernel: str, lam: float, j: int) -> int:
    """Computes the minimum harmonic index supported by the given wavelet scale.

    Args:
        kernel (str): The wavelet type from {"s2dw"}.

        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.

        j (int): Wavelet scale.

    Raises:
        ValueError: Kernel type not supported.

    Returns:
        (int): The minimum harmonic multipole :math:`el` which is supported by a given wavelet scale.
    """
    if kernel.lower() == "s2dw":
        return np.ceil(lam ** (j - 1))
    else:
        raise ValueError(f"Kernel type {kernel} not supported!")


def j_max(L: int, lam: float) -> int:
    """Computes needlet maximum level required to ensure exact reconstruction.

    Args:
        L (int): Harmonic band-limit.

        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.

    Returns:
        (int): The maximum wavelet scale used.
    """
    return int(np.ceil(np.log(L) / np.log(lam)))


def n_phi(sampling_scheme: str, L: int) -> int:
    r"""Returns the number of :math:`\phi` samples.

    Args:
        sampling_scheme (str): Spherical sampling scheme from {"mw","mwss"}.

        L (int): Upper harmonic band-limit.

    Returns:
        (int): Total number of :math:`\phi` samples.
    """
    if sampling_scheme.lower() == "mwss":
        return sampling_mw_ss_nphi(L)
    elif sampling_scheme.lower() == "mw":
        return sampling_mw_nphi(L)


def n_theta(sampling_scheme: str, L: int) -> int:
    r"""Returns the number of :math:`\theta` samples.

    Args:
        sampling_scheme (str): Spherical sampling scheme from {"mw","mwss"}.

        L (int): Harmonic band-limit.

    Returns:
        (int): Total number of :math:`\theta` samples.
    """
    if sampling_scheme.lower() == "mwss":
        return sampling_mw_ss_ntheta(L)
    elif sampling_scheme.lower() == "mw":
        return sampling_mw_ntheta(L)


def n_px(sampling_scheme: str, L: int) -> int:
    r"""Returns the number of spherical pixels for a given sampling scheme.

    Args:
        sampling_scheme (str): Spherical sampling scheme from {"mw","mwss"}.

        L (int): Harmonic band-limit.

    Returns:
        (int): Total number of pixels.
    """
    return n_phi(sampling_scheme, L) * n_theta(sampling_scheme, L)


def n_lm(L: int) -> int:
    """Returns the number of harmonic coefficients at bandlimit L.

    Args:
        L (int): Harmonic band-limit.

    Returns:
        (int): Total number of harmonic coefficients.
    """
    return L * L


def n_lm_scal(upsample: bool, L: int, kernel: str, J_min: int, lam: float) -> int:
    r"""Computes the total number of harmonic coefficients for scaling kernels :math:`\Phi_{\el m}`

    Args:
        upsample (bool): Whether to store the scales at :math:`j_{\text{max}}` resolution or its own resolution.

        L (int): Harmonic band-limit.

        kernel (str): The wavelet type from {"s2dw"}.

        J_min (int): Lowest frequency wavelet scale to be used.

        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.

    Returns:
        (int): Number of harmonic coefficients for scaling kernels :math:`\Phi_{\el m}`.
    """
    if upsample == True:
        bandlimit = L
    else:
        bandlimit = min(bandlimit(kernel, J_min - 1, lam, L), L)

    return bandlimit * bandlimit


def n_lmn_wav(
    lam: float,
    L: int,
    J_min: int,
    upsample: bool,
    kernel: str,
    N: int,
    storage: str,
    reality: int,
) -> int:
    r"""Computes the total number of Wigner coefficients for directional wavelet kernels :math:`\Psi^j_{\el n}`.

    Calls upon functions originally defined in so3, which have been defined explicitly for the case needed here.

    Args:
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.

        L (int): Harmonic band-limit.

        J_min (int): Lowest frequency wavelet scale to be used.

        upsample (bool): Whether to store the scales at :math:`j_{\text{max}}` resolution or its own resolution.

        kernel (str): The wavelet type from {"s2dw"}.

        N (int): Upper azimuthal band-limit.

        storage (str): The type of storage from {"padded","compact").

        reality (int): A non-zero value indicates the signal :math:`f \in \mathbb{R}`.

    Returns:
        (int): Total number of Wigner space wavelet coefficients :math:`\Psi^j_{\el n}`.
    """

    J = j_max(lam, L)

    total = 0

    for i in range(J_min, J):
        if not upsample:
            bandlimit = min(bandlimit(kernel, i, lam, L), L)
            L = bandlimit
            N = min(N, bandlimit)

        if storage.lower() == "padded":
            if reality != 0:
                sampling_flmn_size = N * L * L
            else:
                sampling_flmn_size = (2 * N - 1) * L * L
        elif storage.lower() == "compact":
            if reality != 0:
                sampling_flmn_size = N * (6 * L * L - (N - 1) * (2 * N - 1)) / 6
            else:
                sampling_flmn_size = (2 * N - 1) * (3 * L * L - N * (N - 1)) / 3
        else:
            print("Invalid storage method.")

        total += sampling_flmn_size

    return total


def n_gamma(N: int, steerable: bool) -> int:
    r"""Computes the number of :math:`\gamma` samples for a given sampling scheme

    Args:
        N (int): Upper orientational band-limit. Only flmn with :math:`n < N` will be stored.

        steerable (bool): Indicates that the signal is steerable.

    Returns:
        (int): Number of :math:`\gamma` samples.
    """
    if steerable:
        return N
    else:
        return 2 * N - 1


def n_scal(
    upsample: bool, L: int, kernel: str, J_min: int, lam: float, sampling_scheme: str
) -> int:
    r"""Computes the number of pixel-space samples for scaling kernels :math:`\Phi`.

    Args:
        upsample (bool): Whether to store the scales at :math:`j_{\text{max}}` resolution or its own resolution.

        L (int): Harmonic band-limit.

        kernel (str): The wavelet type from {"s2dw"}.

        J_min (int): Lowest frequency wavelet scale to be used.

        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.

        sampling_scheme (str): Spherical sampling scheme from {"mw","mwss"}.

    Returns:
       (int): Total number of pixel-space samples for scaling kernels :math:`\Phi`.
    """
    if upsample == True:
        bandlimit = L
    else:
        bandlimit = min(bandlimit(kernel, J_min - 1, lam, L), L)
    return n_phi(sampling_scheme, L) * n_theta(sampling_scheme, L)


def n_wav(
    lam: float,
    L: int,
    J_min: int,
    upsample: bool,
    kernel: str,
    sampling_scheme: str,
    steerable: bool,
    N: int,
) -> int:
    r"""Computes the total number of pixel-space samples for directional wavelet kernels :math:`\Psi`.

    Calls upon functions originally defined in so3, which have been defined explicitly for the case needed here.

    Args:
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.

        L (int): Harmonic band-limit.

        J_min (int): Lowest frequency wavelet scale to be used.

        upsample (bool): Whether to store the scales at :math:`j_{\text{max}}` resolution or its own resolution.

        kernel (str): The wavelet type from {"s2dw"}.

        sampling_scheme (str): Spherical sampling scheme from {"mw","mwss"}.

        steerable (bool): Indicates that the signal is steerable.

        N (int): Upper orientational band-limit. Only flmn with :math:`n < N` will be stored.

    Returns:
        (int): Total number of pixel-space samples for :math:`\Psi`.
    """
    J = j_max(lam, L)
    bandlimit = L

    total = 0

    for i in range(J_min, J):
        if not upsample:
            bandlimit = min(bandlimit(kernel, i, lam, L), L)
            L = bandlimit

            if sampling_scheme.lower() == "mw":
                sampling_nalpha = 2 * L - 1
                sampling_nbeta = L
            else:
                sampling_nalpha = 2 * L
                sampling_nbeta = L + 1

            if steerable:
                sampling_ngamma = N
            else:
                sampling_ngamma = 2 * N - 1

        sampling_f_size = sampling_nalpha * sampling_nbeta * sampling_ngamma
        total += sampling_f_size
    return total


def n_wav_j(
    upsample: bool,
    L: int,
    kernel: str,
    j: int,
    lam: float,
    sampling_scheme: str,
    steerable: bool,
    N: int,
) -> int:
    r"""Number of directional wavelet pixel-space coefficients for a specific scale :math:`j`.

    Calls upon functions originally defined in so3, which have been defined explicitly for the case needed here.

    Args:
        upsample (bool): Whether to store the scales at :math:`j_{\text{max}}` resolution or its own resolution.

        L (int): Harmonic band-limit.

        kernel (str): The wavelet type from {"s2dw"}.

        j (int): Wavelet scale under consideration.

        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.

        sampling_scheme (str): Spherical sampling scheme from {"mw","mwss"}.

        steerable (bool): Indicates that the signal is steerable.

        N (int): Upper orientational band-limit. Only flmn with :math:`n < N` will be stored.

    Returns:
        (int): The number of wavelet pixel-space coefficients at scale :math:`j`.
    """
    if not upsample:
        L = min(bandlimit(kernel, j, lam, L), L)

    if sampling_scheme.lower() == "mw":
        sampling_nalpha = 2 * L - 1
        sampling_nbeta = L
    else:
        sampling_nalpha = 2 * L
        sampling_nbeta = L + 1

    if steerable:
        sampling_ngamma = N
    else:
        sampling_ngamma = 2 * N - 1

    sampling_f_size = sampling_nalpha * sampling_nbeta * sampling_ngamma

    return sampling_f_size

def elm2ind(el: int, m: int) -> int:
    """Convert from spherical harmonic 2D indexing of :math:`(\ell,m)` to 1D index.
    1D index is defined by `el**2 + el + m`.

    Warning:
        Note that 1D storage of spherical harmonic coefficients is *not* the default.

    Args:
        el (int): Harmonic degree :math:`\el`.

        m (int): Harmonic order :math:`m`.

    Returns:
        int: Corresponding 1D index value.
    """

    return el**2 + el + m

def elmn2ind(el: int, m: int, n: int, L: int, N: int = 1) -> int:
    """Convert from Wigner space 3D indexing of :math:`(\ell,m, n)` to 1D index.
    
    Args:
        el (int): Harmonic degree :math:`\ell`.

        m (int): Harmonic order :math:`m`.

        n (int): Directional order :math:`n`.

        L (int): Harmonic band-limit.

        N (int, optional): Number of Fourier coefficients for tangent plane rotations (i.e. directionality). Defaults to 1.

    Returns:
        (int): Corresponding 1D index in Wigner space.
    """
    n_offset = (N - 1 + n) * L * L
    el_offset = el * el
    return n_offset + el_offset + el + m
