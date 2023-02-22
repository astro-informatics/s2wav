from jax import jit, config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from s2wav.utils.shapes import j_max
from functools import partial


def tiling_integrand(t: float, lam: float = 2.0) -> float:
    r"""Tiling integrand for scale-discretised wavelets `[1] <https://arxiv.org/pdf/1211.1680.pdf>`_.

    Intermediate step used to compute the wavelet and scaling function generating
    functions. One of the basic mathematical functions needed to carry out the tiling of
    the harmonic space.

    Args:
        t (float): Real argument over which we integrate.

        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales.Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.

    Returns:
        float: Value of tiling integrand for given :math:`t` and scaling factor.

    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on
            the sphere", A&A, vol. 558, p. A128, 2013.
    """
    s_arg = (t - (1.0 / lam)) * (2.0 * lam / (lam - 1.0)) - 1.0

    integrand = np.exp(-2.0 / (1.0 - s_arg**2.0)) / t

    return integrand


def part_scaling_fn(a: float, b: float, n: int, lam: float = 2.0) -> float:
    r"""Computes integral used to calculate smoothly decreasing function :math:`k_{\lambda}`.

    Intermediate step used to compute the wavelet and scaling function generating
    functions. Uses the trapezium method to integrate :func:`~tiling_integrand` in the
    limits from :math:`a \rightarrow b` with scaling parameter :math:`\lambda`. One of
    the basic mathematical functions needed to carry out the tiling of the harmonic
    space.

    Args:
        a (float): Lower limit of the numerical integration.

        b (float): Upper limit of the numerical integration.

        n (int): Number of steps to be performed during integration.

        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales.Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.

    Returns:
        float: Integral of the tiling integrand from :math:`a \rightarrow b`.
    """
    sum = 0.0
    h = (b - a) / n

    if a == b:
        return 0

    for i in range(n):
        if a + i * h not in [1 / lam, 1.0] and a + (i + 1) * h not in [
            1 / lam,
            1.0,
        ]:
            f1 = tiling_integrand(a + i * h, lam)
            f2 = tiling_integrand(a + (i + 1) * h, lam)

            sum += ((f1 + f2) * h) / 2

    return sum


def k_lam(L: int, lam: float = 2.0, quad_iters: int = 300) -> float:
    r"""Compute function :math:`k_{\lambda}` used as a wavelet generating function.

    Specifically, this function is derived in [1] and is given by

    .. math::

        k_{\lambda} \equiv \frac{ \int_t^1 \frac{\text{d}t^{\prime}}{t^{\prime}}
        s_{\lambda}^2(t^{\prime})}{ \int_{\frac{1}{\lambda}}^1
        \frac{\text{d}t^{\prime}}{t^{\prime}} s_{\lambda}^2(t^{\prime})},

    where the integrand is defined to be

    .. math::

        s_{\lambda} \equiv s \Big ( \frac{2\lambda}{\lambda - 1}(t-\frac{1}{\lambda})
        - 1 \Big ),

    for infinitely differentiable Cauchy-Schwartz function :math:`s(t) \in C^{\infty}`.

    Args:
        L (int): Harmonic band-limit.

        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.

        quad_iters (int, optional): Total number of iterations for quadrature
            integration. Defaults to 300.

    Returns:
        (np.ndarray): Value of :math:`k_{\lambda}` computed for values between
            :math:`\frac{1}{\lambda}` and 1, parametrised by :math:`\el` as required to
            compute the axisymmetric filters in :func:`~tiling_axisym`.

    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the
            sphere", A&A, vol. 558, p. A128, 2013.
    """

    J = j_max(L, lam)

    normalisation = part_scaling_fn(1.0 / lam, 1.0, quad_iters, lam)
    k = np.zeros((J + 2, L))

    for j in range(J + 2):
        for l in range(L):
            if l < lam ** (j - 1):
                k[j, l] = 1
            elif l > lam**j:
                k[j, l] = 0
            else:
                k[j, l] = (
                    part_scaling_fn(l / lam**j, 1.0, quad_iters, lam) / normalisation
                )

    return k


@partial(jit, static_argnums=(2, 3))  # not sure
def part_scaling_fn_jax(a: float, b: float, n: int, lam: float = 2.0) -> float:
    r"""JAX version of part_scaling_fn. Computes integral used to calculate smoothly decreasing function :math:`k_{\lambda}`.

    Intermediate step used to compute the wavelet and scaling function generating
    functions. Uses the trapezium method to integrate :func:`~tiling_integrand` in the
    limits from :math:`a \rightarrow b` with scaling parameter :math:`\lambda`. One of
    the basic mathematical functions needed to carry out the tiling of the harmonic
    space.

    Args:
        a (float): Lower limit of the numerical integration.

        b (float): Upper limit of the numerical integration.

        n (int): Number of steps to be performed during integration.

        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales.Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.

    Returns:
        float: Integral of the tiling integrand from :math:`a \rightarrow b`.
    """

    h = (b - a) / n

    x = jnp.linspace(a, b, num=n + 1)
    s_arg = (x - (1.0 / lam)) * (2.0 * lam / (lam - 1.0)) - 1.0
    # s_arg = jnp.where((x > 1./lam) & (x < 1.), s_arg, jnp.zeros(n+1))
    # integrand = jnp.exp(-2.0 / (1.0 - s_arg**2.0)) / x
    # value = jnp.where((x[:-1]>1/lam) & (x[:-1] < 1) & (x[1:]>1/lam) & (x[1:] < 1), integrand[:-1]+integrand[1:], jnp.zeros(n))
    value = jnp.where(
        (x[:-1] == 1.0 / lam) | (x[:-1] == 1.0) | (x[1:] == 1.0 / lam) | (x[1:] == 1.0),
        jnp.zeros(n),
        (jnp.exp(-2.0 / (1.0 - jnp.square(s_arg))) / x)[:-1]
        + (jnp.exp(-2.0 / (1.0 - jnp.square(s_arg))) / x)[1:],
    )

    return jnp.sum(value * h / 2)


@partial(jit, static_argnums=(0, 1, 2))
def k_lam_jax(L: int, lam: float = 2.0, quad_iters: int = 300) -> float:
    r"""JAX version of k_lam. Compute function :math:`k_{\lambda}` used as a wavelet generating function.

    Specifically, this function is derived in [1] and is given by

    .. math::

        k_{\lambda} \equiv \frac{ \int_t^1 \frac{\text{d}t^{\prime}}{t^{\prime}}
        s_{\lambda}^2(t^{\prime})}{ \int_{\frac{1}{\lambda}}^1
        \frac{\text{d}t^{\prime}}{t^{\prime}} s_{\lambda}^2(t^{\prime})},

    where the integrand is defined to be

    .. math::

        s_{\lambda} \equiv s \Big ( \frac{2\lambda}{\lambda - 1}(t-\frac{1}{\lambda})
        - 1 \Big ),

    for infinitely differentiable Cauchy-Schwartz function :math:`s(t) \in C^{\infty}`.

    Args:
        L (int): Harmonic band-limit.

        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.

        quad_iters (int, optional): Total number of iterations for quadrature
            integration. Defaults to 300.

    Returns:
        (np.ndarray): Value of :math:`k_{\lambda}` computed for values between
            :math:`\frac{1}{\lambda}` and 1, parametrised by :math:`\el` as required to
            compute the axisymmetric filters in :func:`~tiling_axisym`.

    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the
            sphere", A&A, vol. 558, p. A128, 2013.
    """

    J = j_max(L, lam)

    normalisation = part_scaling_fn(1.0 / lam, 1.0, quad_iters, lam)
    k = jnp.zeros((J + 2, L))

    for j in range(J + 2):
        for l in range(L):
            if l < lam ** (j - 1):
                k = k.at[j, l].set(1.0)
            elif l > lam**j:
                k = k.at[j, l].set(0.0)
            else:
                k = k.at[j, l].set(
                    part_scaling_fn(l / lam**j, 1.0, quad_iters, lam) / normalisation
                )

    return k