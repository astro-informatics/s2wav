from jax import jit, config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.special import loggamma
from jax.scipy.special import gammaln as jax_gammaln
from functools import partial


def binomial_coefficient(n: int, k: int) -> int:
    r"""Computes the binomial coefficient :math:`\binom{n}{k}`.

    Args:
        n (int): Number of elements to choose from.

        k (int): Number of elements to pick.

    Returns:
        (int): Number of possible subsets.
    """
    return np.floor(
        0.5 + np.exp(loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1))
    )

#@partial(jit, static_argnums=(0, 1)) #not sure about which arguments are static here
def binomial_coefficient_jax(n: int, k: int) -> int:
    r"""Computes the binomial coefficient :math:`\binom{n}{k}`.

    Args:
        n (int): Number of elements to choose from.

        k (int): Number of elements to pick.

    Returns:
        (int): Number of possible subsets.
    """
    return jnp.floor(
        0.5 + jnp.exp(jax_gammaln(n + 1) - jax_gammaln(k + 1) - jax_gammaln(n - k + 1))
    )


if __name__ == "__main__":

    gamma = 4
    el = 1
    m = jnp.arange(-el, el + 1)
    print(jnp.sqrt(binomial_coefficient(gamma, ((gamma - m) / 2))) / (2**gamma))