import numpy as np
from typing import Tuple
from s2wav import samples

def analysis_transform(flm: np.ndarray,
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    spin0: int = 0,
    sampling: str = "mw",
    kernel: str = "s2dw",
    reality: bool = False,
    multiresolution: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Computes the directional analysis transform.

    Insert equation here?

    Args:
        flm (np.ndarray): Signal :math:`f` on the sphere with shape :math:`[n_{\theta}, n_{\phi}]`.

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

    J = samples.j_max(L, lam)

    for j in range(J_min, J + 1):
        for n in range(-N + 1, N + 1, 2):
            for el in range(max(abs(spin), abs(n)), L + 1):
                if el != 0: #Saw it was done this way in synthesis transform, is it correct here too?
                    psi = 8 * np.pi * np.pi / (2 * el + 1) * conj(wav_lm[j, el, L - 1 + n])
                    for m in range (-el, el + 1):
                        flm_wav_lmn[j, N - 1 + n, el, L - 1 + m] = flm[el, L - 1 + m] * psi
    
    for el in range(abs(spin), L +1):
        phi = np.sqrt(4.0 * np.pi / (2 * el + 1)) * scal_l[el]
        for m in range (-el, el + 1):
            f_scal[el, L - 1 + m] = flm[el, L - 1 + m] * phi
    
    return f_wav, f_scal

