import numpy as np
from s2wav import tiling, samples


def k_lam(L: int, lam: float = 2.0, quad_iters: int = 300) -> float:
    r"""Compute function :math:`k_{\lambda}` used as a wavelet generating function.

    Specifically, this function is derived in [1] and is given by

    .. math::

        k_{\lambda} \equiv \frac{ \int_t^1 \frac{\text{d}t^{\prime}}{t^{\prime}} s_{\lambda}^2(t^{\prime})}{ \int_{\frac{1}{\lambda}}^1 \frac{\text{d}t^{\prime}}{t^{\prime}} s_{\lambda}^2(t^{\prime})},

    where the integrand is defined to be

    .. math::

        s_{\lambda} \equiv s \Big ( \frac{2\lambda}{\lambda - 1}(t-\frac{1}{\lambda}) - 1 \Big ),

    for infinitely differentiable Cauchy-Schwartz function :math:`s(t) \in C^{\infty}`.

    Args:
        L (int): Harmonic band-limit.

        lam (float, optional): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
            Note that :math:`\lambda = 2` indicates dyadic wavelets. Defaults to 2.

        quad_iters (int, optional): Total number of iterations for quadrature integration. Defaults to 300.

    Returns:
        (np.ndarray): Value of :math:`k_{\lambda}` computed for values between :math:`\frac{1}{\lambda}` and 1, parametrised by :math:`\el` as required to compute the axisymmetric filters in :func:`~tiling_axisym`.

    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
    """

    J = samples.j_max(L, lam)

    normalisation = tiling.part_scaling_fn(1 / lam, 1.0, quad_iters, lam)
    k = np.zeros((J + 2) * L)

    for j in range(J + 2):
        for l in range(L):
            if l < lam ** (j - 1):
                k[l + j * L] = 1
            elif l > lam**j:
                k[l + j * L] = 0
            else:
                k[l + j * L] = (
                    tiling.part_scaling_fn(l / lam**j, 1.0, quad_iters, lam)
                    / normalisation
                )

    return k
