import numpy as np
from s2wav import tiling
from s2wav import helper_functions
from typing import Tuple

def k_lam(L:int, lam:float, quad_iters: int=300) -> float:
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

        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.

        quad_iters (int, optional): Total number of iterations for quadrature integration. Defaults to 300.

    Returns:
        (np.ndarray): Value of :math:`k_{\lambda}` computed for values between :math:`\frac{1}{\lambda}` and 1, parametrised by :math:`\el` as required to compute the axisymmetric filters in :func:`~tiling_axisym`.

    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
    """

    J = helper_functions.j_max(L, lam)

    normalisation = tiling.part_scaling_fn(1/lam, 1.0, quad_iters, lam)
    k = np.zeros((J + 2) * L)
    

    for j in range(J+2):
        for l in range(L):
            if l < lam**(j-1):
                k[l + j * L] = 1
            elif l > lam**j:
                k[l + j * L] = 0
            else:
                k[l + j * L] =  tiling.part_scaling_fn(l/lam**j, 1.0, quad_iters, lam)/ normalisation
            

    return k


def filters_axisym(L: int, lam:float, J_min: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""Computes wavelet kernels :math:`\Psi^j_{\el m}` and scaling kernel :math:`\Phi_{\el m}` in harmonic space. 

    Specifically, these kernels are derived in [1], where the wavelet kernels are defined (15) for scale :math:`j` to be 

    .. math::

        \Psi^j_{\el m} \equiv \sqrt{\frac{2\el+1}{4\pi}} \kappa_{\lambda}(\frac{\el}{\lambda^j})\delta_{m0},

    where :math:`\kappa_{\lambda} = \sqrt{k_{\lambda}(t/\lambda) - k_{\lambda}(t)}` for :math:`k_{\lambda}` given in :func:`~k_lam`. Similarly, the scaling kernel is defined (16) as

    .. math::

        \Phi_{\el m} \equiv \sqrt{\frac{2\el+1}{4\pi}} \nu_{\lambda} (\frac{\el}{\lambda^{J_0}})\delta_{m0},
    
    where :math:`\nu_{\lambda} = \sqrt{k_{\lambda}(t)}` for :math:`k_{\lambda}` given in :func:`~k_lam`. Notice that :math:`\delta_{m0}` enforces that these kernels are axisymmetric, i.e. coefficients for :math:`m \not = \el` are zero. In this implementation the normalisation constant has been omitted as it is nulled in subsequent functions.
    
    Args:
        L (int): Harmonic band-limit.

        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.

        J_min (int): First wavelet scale used.

    Raises:
        ValueError: L is not an integer.

        ValueError: L is a negative integer.

        ValueError: J_min is not an integer.

        ValueError: J_min is negative or greater than J.

    Returns:
        (Tuple[np.ndarray, np.ndarray]): Unnormalised wavelet kernels and scaling kernel in harmonic space.
    
    Note:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
    """
    if not isinstance(L, int):
        raise TypeError("L must be an integer")
    
    if L < 0:
        raise ValueError("L must be non-negative")
    
    if not isinstance(J_min, int):
        raise TypeError("J_min must be an integer")

    J = helper_functions.j_max(L, lam)

    if J_min >= J or J_min <0:
        raise ValueError("J_min must be non-negative and less than J= "+str(J)+" for given L and lam.")

    previoustemp = 0.0
    k = k_lam(L, lam)
    psi = np.zeros((J + 1) * L)
    phi = np.zeros(L)
    for l in range(L):
      phi[l] = np.sqrt(k[l + J_min * L])
    
    for j in range(J_min, J+1):
        for l in range(L):
            diff = k[l + (j + 1) * L] - k[l + j * L] 
            #check if sqrt is defined
            if diff < 0:
              psi[l + j * L] = previoustemp
            else:
              temp = np.sqrt(diff)
              psi[l + j * L] = temp
            previoustemp = temp
          
    return psi, phi
    