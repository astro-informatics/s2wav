import numpy as np
from typing import Tuple
from s2wav import samples, filters, shapes
import so3
import pyssht as ssht


def analysis_transform(
    f: np.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    spin0: int = 0,
    sampling: str = "mw",
    kernel: str = "s2dw",
    reality: bool = False,
    multiresolution: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Wavelet analysis from pixel space to wavelet space for complex signals.

    Args:
        f (np.ndarray): Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

        L (int): Harmonic bandlimit.

        N (int, optional): Upper azimuthal band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

        spin (int, optional): Spin (integer) of input signal. Defaults to 0.

        spin0 (int, optional): Spin (integer) of output signal. Defaults to 0.

        sampling (str, optional): Spherical sampling scheme from {"mw","mwss"}. Defaults to "mw".

        kernel (str, optional): The wavelet type from {"s2dw"}. Defaults to "s2dw".

        reality (bool, optional): Whether :math:`f \in \mathbb{R}`, if True exploits
            conjugate symmetry of harmonic coefficients. Defaults to False.

        multiresolution (bool, optional): Whether to store the scales at :math:`j_{\text{max}}`
            resolution or its own resolution. Defaults to False.

    Returns:
        f_wav (np.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (np.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.
    """
    flm = ssht_to_s2wav(ssht.forward(f, L, Reality=reality), L)
    J = samples.j_max(L, lam)

    f_wav_lmn = np.zeros(shapes.flmn_wav(L, N, J_min, lam), dtype=np.complex128)
    f_scal_lm = np.zeros(shapes.flm_scal(L), dtype=np.complex128)

    wav_lm, scal_l = filters.filters_directional(L, N, J_min, lam, spin, spin0)
    for j in range(J_min, J + 1):
        for n in range(-N + 1, N, 2):
            for el in range(max(abs(spin), abs(n)), L):
                if el != 0:
                    psi = np.conj(wav_lm[j, el, L - 1 + n])
                    psi *= 8 * np.pi**2 / (2 * el + 1)
                    for m in range(-el, el + 1):
                        f_wav_lmn[j, N - 1 + n, el, L - 1 + m] = (
                            flm[el, L - 1 + m] * psi
                        )

    for el in range(abs(spin), L):
        phi = np.sqrt(4.0 * np.pi / (2 * el + 1)) * scal_l[el]
        for m in range(-el, el + 1):
            f_scal_lm[el, L - 1 + m] = flm[el, L - 1 + m] * phi

    params = so3.create_parameter_dict(L=L, N=N)
    f_wav = np.zeros(shapes.f_wav(L, N, J_min, lam), dtype=np.complex128)

    for j in range(J_min, J + 1):
        params.L0 = samples.L0(j, lam, kernel)
        temp = so3.inverse(s2wav_to_so3(f_wav_lmn[j, ...], L, N), params)
        f_wav[j, ...] = temp.reshape(2 * N - 1, L, 2 * L - 1)

    f_scal = ssht.inverse(s2wav_to_ssht(f_scal_lm, L), L)

    return f_wav, f_scal


def s2wav_to_ssht(flm, L):
    """Temporary function to convert flm from ssht to s2wav indexing"""
    flm_out = np.zeros(L * L, dtype=np.complex128)
    ind = 0
    for el in range(L):
        for m in range(-el, el + 1):
            flm_out[ind] = flm[el, L - 1 + m]
            ind += 1
    return flm_out


def ssht_to_s2wav(flm, L):
    """Temporary function to convert flm from ssht to s2wav indexing"""
    flm_out = np.zeros((L, 2 * L - 1), dtype=np.complex128)
    ind = 0
    for el in range(L):
        for m in range(-el, el + 1):
            flm_out[el, L - 1 + m] = flm[ind]
            ind += 1
    return flm_out


def s2wav_to_so3(flmn, L, N):
    """Temporary function to convert flmn from so3 to s2wav indexing"""
    flmn_out = np.zeros((2 * N - 1, L * L), dtype=np.complex128)
    for n in range(-N + 1, N):
        ind = 0
        for el in range(L):
            for m in range(-el, el + 1):
                flmn_out[N - 1 + n, ind] = flmn[N - 1 + n, el, L - 1 + m]
                ind += 1
    return flmn_out.flatten("C")
