import numpy as np
from s2wav import tiling
from s2wav import helper_functions
from typing import Tuple
import pytest

def k_lam(L:int, lam:float, quad_iters: int=300) -> float:
    """Compute function k_lam defined in equation (12) in [1] used to compute scaling function generating function and
        wavelet generating function.

    Args:
        L (int): Harmonic band-limit.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        quad_iters (int): Total number of iterations for quadrature integration. Defaults to 300.
    Returns:
        np.ndarray: value of k_lam computed for values between 1/lam and 1, parametrised by l as required to compute 
        the axisymmetric filters in tiling_axisym()

    Notes:
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


def filters_axisym(L: int, lam:float, J_min: int) -> Tuple[np.ndarray]:
    """Computes wavelet kernels and scaling kernel in harmonic space, according to equations (15), (16) in [1]
        without the normalisation factor.
    
    Args:
        L (int): Harmonic band-limit.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        J_min (int): First wavelet scale used.
    Returns:
        Tuple[np.ndarray]: Unnormalised wavelet kernels and scaling kernel in harmonic space.
    
    Notes:
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
    Psi = np.zeros((J + 1) * L)
    Phi = np.zeros(L)
    for l in range(L):
      Phi[l] = np.sqrt(k[l + J_min * L])
    
    for j in range(J_min, J+1):
        for l in range(L):
            diff = k[l + (j + 1) * L] - k[l + j * L] 
            #check if sqrt is defined
            if diff < 0:
              Psi[l + j * L] = previoustemp
            else:
              temp = np.sqrt(diff)
              Psi[l + j * L] = temp
            previoustemp = temp
          
    return Psi, Phi
    