from jax import jit
import jax.numpy as jnp
import numpy as np
from typing import Tuple
from functools import partial
from s2wav import samples


def filters_axisym(
    L: int, J_min: int = 0, lam: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Computes wavelet kernels :math:`\Psi^j_{\ell m}` and scaling kernel 
        :math:`\Phi_{\ell m}` in harmonic space.

    Specifically, these kernels are derived in `[1] <https://arxiv.org/pdf/1211.1680.pdf>`_, 
    where the wavelet kernels are defined (15) for scale :math:`j` to be

    .. math::

        \Psi^j_{\ell m} \equiv \sqrt{\frac{2\ell+1}{4\pi}} \kappa_{\lambda}(\frac{\ell}{\lambda^j})\delta_{m0},

    where :math:`\kappa_{\lambda} = \sqrt{k_{\lambda}(t/\lambda) - k_{\lambda}(t)}` for :math:`k_{\lambda}` 
    given in :func:`~k_lam`. Similarly, the scaling kernel is defined (16) as

    .. math::

        \Phi_{\ell m} \equiv \sqrt{\frac{2\ell+1}{4\pi}} \nu_{\lambda} (\frac{\ell}{\lambda^{J_0}})\delta_{m0},

    where :math:`\nu_{\lambda} = \sqrt{k_{\lambda}(t)}` for :math:`k_{\lambda}` given in :func:`~k_lam`. 
    Notice that :math:`\delta_{m0}` enforces that these kernels are axisymmetric, i.e. coefficients 
    for :math:`m \not = \ell` are zero. In this implementation the normalisation constant has been 
    omitted as it is nulled in subsequent functions.

    Args:
        L (int): Harmonic band-limit.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between 
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic 
            wavelets. Defaults to 2.

    Raises:
        ValueError: J_min is negative or greater than J.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Unnormalised wavelet kernels :math:`\Psi^j_{\ell m}` 
        with shape :math:`[(J+1)L]`, and scaling kernel :math:`\Phi_{\el m}` with shape 
        :math:`[L]` in harmonic space.

    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
    """
    J = samples.j_max(L, lam)

    if J_min >= J or J_min < 0:
        raise ValueError(
            "J_min must be non-negative and less than J= "
            + str(J)
            + " for given L and lam."
        )

    previoustemp = 0.0
    k = k_lam(L, lam)
    psi = np.zeros((J + 1, L), np.float64)
    phi = np.zeros(L, np.float64)
    for l in range(L):
        phi[l] = np.sqrt(k[J_min, l])

    for j in range(J_min, J + 1):
        for l in range(L):
            diff = k[j + 1, l] - k[j, l]
            if diff < 0:
                psi[j, l] = previoustemp
            else:
                temp = np.sqrt(diff)
                psi[j, l] = temp
            previoustemp = temp

    return psi, phi


def filters_directional(
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    spin0: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Generates the harmonic coefficients for the directional tiling wavelets.

    This implementation is based on equation 36 in the wavelet computation paper 
    `[1] <https://arxiv.org/pdf/1509.06749.pdf>`_.

    Args:
        L (int): Harmonic band-limit.

        N (int, optional): Upper azimuthal band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between 
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic 
            wavelets. Defaults to 2.

        spin (int, optional): Spin (integer) to perform the transform. Defaults to 0.

        spin0 (int, optional): Spin number the wavelet was lowered from. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of wavelet and scaling kernels 
            (:math:`\Psi^j_{\ell n}`, :math:`\Phi_{\ell m}`)

    Notes:
        [1] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint arXiv:1509.06749 (2015).
    """
    J = samples.j_max(L, lam)
    el_min = max(abs(spin), abs(spin0))

    phi = np.zeros(L, dtype=np.float64)
    psi = np.zeros((J + 1, L, 2 * L - 1), dtype=np.complex128)

    kappa, kappa0 = filters_axisym(L, J_min, lam)
    s_elm = tiling_direction(L, N)

    for el in range(el_min, L):
        if kappa0[el] != 0:
            phi[el] = np.sqrt((2 * el + 1) / (4.0 * np.pi)) * kappa0[el]
            if spin0 != 0:
                phi[el] *= spin_normalization(el, spin0) * (-1) ** spin0

    for j in range(J_min, J + 1):
        for el in range(el_min, L):
            if kappa[j, el] != 0:
                for m in range(-el, el + 1):
                    if s_elm[el, L - 1 + m] != 0:
                        psi[j, el, L - 1 + m] = (
                            np.sqrt((2 * el + 1) / (8.0 * np.pi * np.pi))
                            * kappa[j, el]
                            * s_elm[el, L - 1 + m]
                        )
                        if spin0 != 0:
                            psi[j, el, L - 1 + m] *= (
                                spin_normalization(el, spin0) * (-1) ** spin0
                            )

    return psi, phi


def filters_axisym_vectorised(
    L: int, J_min: int = 0, lam: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Vectorised version of :func:`~filters_axisym`.

    Args:
        L (int): Harmonic band-limit.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.

    Raises:
        ValueError: J_min is negative or greater than J.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Unnormalised wavelet kernels :math:`\Psi^j_{\ell m}`
        with shape :math:`[(J+1)L], and scaling kernel :math:`\Phi_{\ell m}` with shape
        :math:`[L]` in harmonic space.
    """
    J = samples.j_max(L, lam)

    if J_min >= J or J_min < 0:
        raise ValueError(
            "J_min must be non-negative and less than J= "
            + str(J)
            + " for given L and lam."
        )

    k = k_lam(L, lam)
    diff = (np.roll(k, -1, axis=0) - k)[:-1]
    diff[diff < 0] = 0
    return np.sqrt(diff), np.sqrt(k[J_min])


def filters_directional_vectorised(
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    spin0: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Vectorised version of :func:`~filters_directional`.

    Args:
        L (int): Harmonic band-limit.

        N (int, optional): Upper azimuthal band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        spin (int, optional): Spin (integer) to perform the transform. Defaults to 0.

        spin0 (int, optional): Spin number the wavelet was lowered from. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of wavelet and scaling kernels 
            (:math:`\Psi^j_{\ell n}`, :math:`\Phi_{\ell m}`).
    """
    el_min = max(abs(spin), abs(spin0))

    spin_norms = (
        (-1) ** spin0 * spin_normalization_vectorised(np.arange(L), spin0)
        if spin0 != 0
        else 1
    )

    kappa, kappa0 = filters_axisym_vectorised(L, J_min, lam)
    s_elm = tiling_direction(L, N)

    kappa0 *= np.sqrt((2 * np.arange(L) + 1) / (4.0 * np.pi))
    kappa0 = kappa0 * spin_norms if spin0 != 0 else kappa0

    kappa *= np.sqrt((2 * np.arange(L) + 1) / 8.0) / np.pi
    kappa = np.einsum("ij,jk->ijk", kappa, s_elm)
    kappa = np.einsum("ijk,j->ijk", kappa, spin_norms) if spin0 != 0 else kappa

    kappa0[:el_min] = 0
    kappa[:, :el_min, :] = 0
    return kappa, kappa0


@partial(jit, static_argnums=(0, 1, 2))
def filters_axisym_jax(
    L: int, J_min: int = 0, lam: float = 2.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""JAX version of :func:`~filters_axisym_vectorised`.

    Args:
        L (int): Harmonic band-limit.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor
            between consecutive wavelet scales. Note that :math:`\lambda = 2` indicates
            dyadic wavelets. Defaults to 2.

    Raises:
        ValueError: J_min is negative or greater than J.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Unnormalised wavelet kernels :math:`\Psi^j_{\ell m}`
        with shape :math:`[(J+1)L], and scaling kernel :math:`\Phi_{\ell m}` with shape
        :math:`[L]` in harmonic space.
    """
    J = samples.j_max(L, lam)

    if J_min >= J or J_min < 0:
        raise ValueError(
            "J_min must be non-negative and less than J= "
            + str(J)
            + " for given L and lam."
        )

    k = k_lam_jax(L, lam)
    diff = (jnp.roll(k, -1, axis=0) - k)[:-1]
    diff = jnp.where(diff < 0, jnp.zeros((J + 1, L)), diff)
    return jnp.sqrt(diff), jnp.sqrt(k[J_min])


@partial(jit, static_argnums=(0, 1, 2, 3, 4, 5))
def filters_directional_jax(
    L: int,
    N: int = 1,
    J_min: int = 0,
    lam: float = 2.0,
    spin: int = 0,
    spin0: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""JAX version of :func:`~filters_directional`.

    Args:
        L (int): Harmonic band-limit.

        N (int, optional): Upper azimuthal band-limit. Defaults to 1.

        J_min (int, optional): Lowest frequency wavelet scale to be used. Defaults to 0.

        lam (float, optional): Wavelet parameter which determines the scale factor between
            consecutive wavelet scales. Note that :math:`\lambda = 2` indicates dyadic
            wavelets. Defaults to 2.

        spin (int, optional): Spin (integer) to perform the transform. Defaults to 0.

        spin0 (int, optional): Spin number the wavelet was lowered from. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of wavelet and scaling kernels 
            (:math:`\Psi^j_{\ell n}`, :math:`\Phi_{\ell m}`).
    """
    el_min = max(abs(spin), abs(spin0))

    spin_norms = (
        (-1) ** spin0 * spin_normalization_jax(np.arange(L), spin0)
        if spin0 != 0
        else 1
    )

    kappa, kappa0 = filters_axisym_jax(L, J_min, lam)
    s_elm = tiling_direction_jax(L, N)

    kappa0 *= jnp.sqrt((2 * jnp.arange(L) + 1) / (4.0 * jnp.pi))
    kappa0 = kappa0 * spin_norms if spin0 != 0 else kappa0

    kappa *= jnp.sqrt((2 * jnp.arange(L) + 1) / 8.0) / np.pi
    kappa = jnp.einsum("ij,jk->ijk", kappa, s_elm, optimize=True)
    kappa = (
        jnp.einsum("ijk,j->ijk", kappa, spin_norms, optimize=True)
        if spin0 != 0
        else kappa
    )

    kappa0 = kappa0.at[:el_min].set(0)
    kappa = kappa.at[:, :el_min, :].set(0)

    return kappa, kappa0

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
            :math:`\frac{1}{\lambda}` and 1, parametrised by :math:`\ell` as required to
            compute the axisymmetric filters in :func:`~tiling_axisym`.

    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the
            sphere", A&A, vol. 558, p. A128, 2013.
    """

    J = samples.j_max(L, lam)

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
    r"""JAX version of part_scaling_fn. Computes integral used to calculate smoothly 
        decreasing function :math:`k_{\lambda}`.

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
    value = jnp.where(
        (x[:-1] == 1.0 / lam) | (x[:-1] == 1.0) | (x[1:] == 1.0 / lam) | (x[1:] == 1.0),
        jnp.zeros(n),
        (jnp.exp(-2.0 / (1.0 - jnp.square(s_arg))) / x)[:-1]
        + (jnp.exp(-2.0 / (1.0 - jnp.square(s_arg))) / x)[1:],
    )

    return jnp.sum(value * h / 2)


@partial(jit, static_argnums=(0, 1, 2))
def k_lam_jax(L: int, lam: float = 2.0, quad_iters: int = 300) -> float:
    r"""JAX version of k_lam. Compute function :math:`k_{\lambda}` used as a wavelet 
        generating function.

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
            :math:`\frac{1}{\lambda}` and 1, parametrised by :math:`\ell` as required to
            compute the axisymmetric filters in :func:`~tiling_axisym`.

    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the
            sphere", A&A, vol. 558, p. A128, 2013.
    """

    J = samples.j_max(L, lam)

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

def tiling_direction(L: int, N: int = 1) -> np.ndarray:
    r"""Generates the harmonic coefficients for the directionality component of the
        tiling functions.

    Formally, this function implements the follow equation

    .. math::

        _{s}\eta_{\el m} = \nu \vu \sqrt{\frac{1}{2^{\gamma}} \big ( \binom{\gamma}{
                (\gamma - m)/2} \big )}

    which was first derived in `[1] <https://arxiv.org/pdf/1211.1680.pdf>`_.

    Args:
        L (int): Harmonic band-limit.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

    Returns:
        np.ndarray: Harmonic coefficients of directionality components
            :math:`_{s}\eta_{\el m}`.

    Notes:
        [1] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint
            arXiv:1509.06749 (2015).
    """
    if N % 2:
        nu = 1
    else:
        nu = 1j

    s_elm = np.zeros((L, 2 * L - 1), dtype=np.complex128)

    for el in range(1, L):
        if (N + el) % 2:
            gamma = min(N - 1, el)
        else:
            gamma = min(N - 1, el - 1)

        for m in range(-el, el + 1):
            if abs(m) < N and (N + m) % 2:
                s_elm[el, L - 1 + m] = nu * np.sqrt(
                    (samples.binomial_coefficient(gamma, ((gamma - m) / 2))) / (2**gamma)
                )
            else:
                s_elm[el, L - 1 + m] = 0.0

    return s_elm


def spin_normalization(el: int, spin: int = 0) -> float:
    r"""Computes the normalization factor for spin-lowered wavelets, which is
        :math:`\sqrt{\frac{(\ell+s)!}{(\ell-s)!}}`.

    Args:
        el (int): Harmonic index :math:`\ell`.

        spin (int): Spin of field over which to perform the transform. Defaults to 0.

    Returns:
        float: Normalization factor for spin-lowered wavelets.
    """
    factor = 1.0

    for s in range(-abs(spin) + 1, abs(spin) + 1):
        factor *= el + s

    if spin > 0:
        return np.sqrt(factor)
    else:
        return np.sqrt(1.0 / factor)


def spin_normalization_vectorised(el: np.ndarray, spin: int = 0) -> float:
    r"""Vectorised version of :func:`~spin_normalization`.
    Args:
        el (int): Harmonic index :math:`\ell`.
        spin (int): Spin of field over which to perform the transform. Defaults to 0.
    Returns:
        float: Normalization factor for spin-lowered wavelets.
    """
    factor = np.arange(-abs(spin) + 1, abs(spin) + 1).reshape(1, 2 * abs(spin) + 1)
    factor = el.reshape(len(el), 1).dot(factor)
    return np.sqrt(np.prod(factor, axis=1) ** (np.sign(spin)))


@partial(jit, static_argnums=(0, 1))
def tiling_direction_jax(L: int, N: int = 1) -> np.ndarray:
    r"""JAX version of tiling_direction. Generates the harmonic coefficients for the 
        directionality component of the tiling functions.

    Formally, this function implements the follow equation

    .. math::

        _{s}\eta_{\ell m} = \nu \vu \sqrt{\frac{1}{2^{\gamma}} \big ( \binom{\gamma}{
                (\gamma - m)/2} \big )}

    which was first derived in `[1] <https://arxiv.org/pdf/1211.1680.pdf>`_.

    Args:
        L (int): Harmonic band-limit.

        N (int, optional): Upper orientational band-limit. Defaults to 1.

    Returns:
        np.ndarray: Harmonic coefficients of directionality components
            :math:`_{s}\eta_{\ell m}`.

    Notes:
        [1] J. McEwen et. al., "Directional spin wavelets on the sphere", arXiv preprint
            arXiv:1509.06749 (2015).
    """

    nu = (N % 2 - 1) ** 2 * 1j + (N % 2)

    s_elm = jnp.zeros((L, 2 * L - 1), dtype=np.complex128)

    for el in range(1, L):
        gamma = min(N - 1, el - 1 + (N + el) % 2)

        ms = jnp.arange(-el, el + 1)
        val = nu * jnp.sqrt(
            (samples.binomial_coefficient_jax(gamma, ((gamma - ms) / 2))) / (2**gamma)
        )

        val = jnp.where(
            (ms < N) & (ms > -N) & ((N + ms) % 2 == 1),
            val,
            jnp.zeros(2 * el + 1),
        )
        s_elm = s_elm.at[el, L - 1 - el : L + el].set(val)

    return s_elm


@partial(jit, static_argnums=(1))
def spin_normalization_jax(el: np.ndarray, spin: int = 0) -> float:
    r"""JAX version of :func:`~spin_normalization`.
    Args:
        el (int): Harmonic index :math:`\ell`.
        spin (int): Spin of field over which to perform the transform. Defaults to 0.
    Returns:
        float: Normalization factor for spin-lowered wavelets.
    """
    factor = jnp.arange(-abs(spin) + 1, abs(spin) + 1).reshape(1, 2 * abs(spin) + 1)
    factor = el.reshape(len(el), 1).dot(factor)
    return jnp.sqrt(jnp.prod(factor, axis=1) ** (jnp.sign(spin)))