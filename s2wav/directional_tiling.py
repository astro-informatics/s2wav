import math
from s2wav.helper_functions import j_max
from scipy.special import loggamma
import numpy as np


#Two functions originally defined in s2let_math.c used for tiling.

def logfact(n: int) ->float:
    '''Computes the natural logarithm of an integer factorial.
    
    The engine of this function actually calculates the gamma function,
    for which the real argument is x = n + 1.

    Args:
        n (int): Integer. 
    
    Returns:
        loggamma (float): log(n!).
    '''

    #Fitting constants
    c = [76.18009172947146,
    -86.50532032941677,
    24.01409824083091,
    -1.231739572450155,
    0.1208650973866179e-2,
    -0.5395239384953e-5]

    #This calculates the gamma function, which takes the argument x = n +1.
    x = n + 1.0
    
    #Add up the fit.

    temp = x + 5.5 - (x + 0.5) * math.log(x + 5.5)
    sum = 1.000000000190015
    y = x

    for i in range(0,6):
        y += 1
        sum = sum + c[i] / y

    return (- temp + math.log(2.5066282746310005 * sum / x))


def binomial_coefficient(exact: bool, n: int, k: int) -> int:
    '''Computes the binomial coefficient "n choose k".
    
    Args:
        exact (bool): True for exact computation, False for approximate.
        n (int): Number of elements to choose from.
        k (int): Number of elements to pick.
    
    Returns:
        (int): Number of possible subsets.
    '''
    if not exact:
        #return math.floor(0.5 + math.exp(logfact(n) - logfact(k) - logfact(n-k)))
        return math.floor(0.5 + math.exp(loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)))


#Coding in lines 249 - 429 from s2let_tiling.c


def tiling_direction(N: int, L: int) -> np.ndarray:
    '''Generates the harmonic coefficients for the directionality component of the tiling functions.

    This implementation is based on equation (11) in the wavelet computation paper [1]. 
    
    Args:
        N (int): Upper orientational band-limit. Only flmn with n < N will be stored.
        L (int): Harmonic band-limit.
    
    Returns:
        s_elm (np.ndarray): Harmonic coefficients of directionality components.

    Notes:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
    '''
    if (N % 2):
        nu = 1
    else:
        nu = 1j
    
    #Skip the s_00 component, as it is zero.
    ind = 1

    s_elm = np.zeros(L*L, dtype = np.complex128)

    for el in range(1,L):
        #This if/else replaces the -1^(N+l)
        if ((N + el) % 2):
            gamma = min(N - 1, el)
        else:
            gamma = min(N - 1, el - 1)

        for m in range(-el, el):
            #This if/else takes care of the azimuthal band-limit and replaces the beta factor.
            if abs(m) < N and (N + m) % 2:
                s_elm[ind] = nu * math.sqrt((binomial_coefficient(gamma, int( (gamma-m)/2 ), 1)) / 2 ** gamma)
            else:
                s_elm[ind] = 0.0

            ind += 1
    
    return s_elm
    


def spin_normalization(spin: int, el: int) -> float:
    '''Computes the normalization factor for spin-lowered wavelets, which is sqrt((l+s)!/(l-s)!).
    
    Args:
        spin (int): Spin (integer) to perform the transform.
        el (int): Harmonic index el.
    
    Returns:
        (float): Normalization factor for spin-lowered wavelets.
    '''
    factor = 1

    for s in range(-abs(spin) + 1, abs(spin)):
        factor *= el + s
    
    if spin > 0:
        return math.sqrt(factor)
    else:
        return math.sqrt(1.0 / factor)


def tiling_wavelet(L: int, lam: float, spin: int, original_spin: int, N: int, J_min: int) -> tuple[np.ndarray, np.ndarray]:
    '''Generates the harmonic coefficients for the directional tiling wavelets.
    
    This implementation is based on equation (7) in the wavelet computation paper [1].
    
    Args:
        L (int): Harmonic band-limit.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        spin (int): Spin (integer) to perform the transform.
        original_spin (int): Spin number the wavelet was lowered from.
        N (int): Upper orientational band-limit. Only flmn with n < N will be stored.
        J_min (int): First wavelet scale used.

    Returns:
        (tuple[np.ndarray, np.ndarray]):
            psi (np.ndarray): Harmonic coefficients of directional wavelets.
            phi (np.ndarray): Harmonic coefficients of scaling function.

    Notes:
        [1] B. Leidstedt et. al., "S2LET: A code to perform fast wavelet analysis on the sphere", A&A, vol. 558, p. A128, 2013.
    '''
    J = j_max(L, lam)
    el_min = max(abs(spin), abs(original_spin))

    phi = np.zeros(L, dtype = np.float64)
    psi = np.zeros((J+1) * L * L, dtype = np.complex128) 
    kappa = np.zeros((J+1) * L, dtype = np.float64)
    kappa0 = np.zeros(L, dtype = np.float64)
    s_elm = tiling_direction(N, L)

    for el in range(el_min, L):
        phi[el] = math.sqrt((2 * el + 1) / 4.0 * math.pi * kappa0[el])
        if original_spin != 0:
            phi[el] *= spin_normalization(el, original_spin) * (-1)**original_spin

    for j in range(J_min, J):
        ind = el_min ** 2
        for el in range(el_min, L):
            for m in range(-el, el):
                psi[j * L * L + ind] = math.sqrt((2 * el + 1) / (8.0 * math.pi * math.pi)) * kappa[j * L + el] * s_elm[ind]
                if original_spin != 0:
                    psi[j * L * L + ind] *= spin_normalization(el, original_spin) * (-1)**original_spin


    return psi, phi