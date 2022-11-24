import numpy as np
from s2wav import samples, filters, shapes

# TODO: Switch to S2FFT
import pyssht as ssht
import so3


def synthesis_transform(
    f_wav: np.ndarray,
    f_scal: np.ndarray,
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
) -> np.ndarray:
    r"""Computes the synthesis directional wavelet transform [1,2].

    Specifically, this transform synthesises the signal :math:`_{s}f(\omega) \in \mathbb{S}^2` by summing the contributions from wavelet and scaling coefficients in harmonic space, see equation 27 from `[2] <https://arxiv.org/pdf/1509.06749.pdf>`_.

    Args:
        f_wav (np.ndarray): Array of wavelet pixel-space coefficients
            with shape :math:`[n_{J}, 2N-1, n_{\theta}, n_{\phi}]`.

        f_scal (np.ndarray): Array of scaling pixel-space coefficients
            with shape :math:`[n_{\theta}, n_{\phi}]`.

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
        np.ndarray: Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

    Notes:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
        [2] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint arXiv:1509.06749 (2015).
    """
    assert f_wav.shape == shapes.f_wav(L, N, J_min, lam, sampling)
    assert f_scal.shape == shapes.f_scal(L, sampling)

    J = samples.j_max(L, lam)
    flmn_shape = shapes.flmn_wav(L, N, J_min, lam)
    params = so3.create_parameter_dict(L=L, N=N)

    # Convert scaling/wavelet coefficients from pixel-space
    # to harmonic/Wigner space.
    f_scal_lm = ssht.forward(f_scal, L)
    f_wav_lmn = np.zeros(flmn_shape, dtype=np.complex128)

    for j in range(J_min, J + 1):
        params.L0 = samples.L0(j, lam, kernel)
        temp = so3.forward(f_wav[j, ...].flatten("C"), params)
        f_wav_lmn[j, ...] = temp.reshape(flmn_shape[1], flmn_shape[2])

    # Generate the directional wavelet kernels
    flm = np.zeros(L * L, dtype=np.complex128)
    wav_lm, scal_l = filters.filters_directional(L, N, J_min, lam, spin, spin0)

    # Sum the all scaling harmonic coefficients for each lm
    for j in range(J_min, J + 1):
        for n in range(-N + 1, N, 2):
            for el in range(max(abs(spin), abs(n)), L):
                if el != 0:
                    lm_ind = samples.elm2ind(el, n)
                    psi = wav_lm[j, lm_ind]
                    for m in range(-el, el + 1):
                        lm_ind = samples.elm2ind(el, m)
                        flm[lm_ind] += f_wav_lmn[j, N - 1 + n, lm_ind] * psi

    # Sum the all scaling harmonic coefficients for each lm
    for el in range(np.abs(spin), L):
        phi = np.sqrt(4 * np.pi / (2 * el + 1)) * scal_l[el]
        for m in range(-el, el + 1):
            lm_ind = samples.elm2ind(el, m)
            flm[lm_ind] += f_scal_lm[lm_ind] * phi

    return ssht.inverse(flm, L, Reality=reality)
