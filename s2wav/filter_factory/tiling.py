from jax import jit, config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from s2wav.utils.math_functions import binomial_coefficient, binomial_coefficient_jax
from functools import partial

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
                    (binomial_coefficient(gamma, ((gamma - m) / 2)))
                    / (2**gamma)
                )
            else:
                s_elm[el, L - 1 + m] = 0.0

    return s_elm


def spin_normalization(el: int, spin: int = 0) -> float:
    r"""Computes the normalization factor for spin-lowered wavelets, which is
        :math:`\sqrt{\frac{(l+s)!}{(l-s)!}}`.

    Args:
        el (int): Harmonic index :math:`\el`.

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
        el (int): Harmonic index :math:`\el`.
        spin (int): Spin of field over which to perform the transform. Defaults to 0.
    Returns:
        float: Normalization factor for spin-lowered wavelets.
    """
    factor = np.arange(-abs(spin) + 1, abs(spin) + 1).reshape(
        1, 2 * abs(spin) + 1
    )
    factor = el.reshape(len(el), 1).dot(factor)
    return np.sqrt(np.prod(factor, axis=1) ** (np.sign(spin)))

#@partial(jit, static_argnums=(0, 1)) #not sure about which arguments are static here
def tiling_direction_jax(L: int, N: int = 1) -> np.ndarray:
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
    

    nu = (N % 2 -1)**2 * 1j + (N % 2)

    s_elm = jnp.zeros((L, 2 * L - 1), dtype=np.complex128)

    for el in range(1, L):
        gamma = min(N - 1, el - 1 + (N + el) % 2)

        ms = jnp.arange(-el, el+1)
        val = nu * jnp.sqrt(
            (binomial_coefficient_jax(gamma, ((gamma - ms) / 2))) / (2**gamma))
        
        val = jnp.where((ms < N) & (ms > -N) & ((N + ms) % 2 == 1), val, jnp.zeros(2*el+1))
        s_elm = s_elm.at[el, L - 1 + ms[0] : L + ms[-1]].set(val)

    return s_elm


@partial(jit, static_argnums=(1)) #not sure about which arguments are static here
def spin_normalization_jax(el: np.ndarray, spin: int = 0) -> float:
    r"""Vectorised version of :func:`~spin_normalization`.
    Args:
        el (int): Harmonic index :math:`\el`.
        spin (int): Spin of field over which to perform the transform. Defaults to 0.
    Returns:
        float: Normalization factor for spin-lowered wavelets.
    """
    factor = jnp.arange(-abs(spin) + 1, abs(spin) + 1).reshape(
        1, 2 * abs(spin) + 1
    )
    factor = el.reshape(len(el), 1).dot(factor)
    return jnp.sqrt(jnp.prod(factor, axis=1) ** (jnp.sign(spin)))



if __name__ == "__main__":
    gamma = 5
    nu = 1
    ms = jnp.arange(-2,3)
    vals = nu * jnp.sqrt(binomial_coefficient_jax(gamma, ((gamma - ms) / 2))) / (2**gamma)

    for m in ms:
        print(nu * jnp.sqrt(binomial_coefficient_jax(gamma, ((gamma - m) / 2))) / (2**gamma))
        print(vals[m+2])
