# This file translates functions which can be called to check dimensions of vectors from s2let (c).

from cmath import log
from functools import total_ordering
import math
from tkinter import N


#Defining functions originally called from ssht


def sampling_mw_ss_nphi(L: int) -> int:
    '''Computes the number of phi samples for McEwen and Wiaux symmetric sampling.
    
    Args:
        L (int): Upper harmonic band-limit.

    Returns:
        nphi (int): Number of phi samples.   
    '''
    return 2*L



def ampling_mw_nphi(L: int) -> int:
    '''Computes the number of phi samples for McEwen and Wiaux sampling.
    
    Args:
        L (int): Upper harmonic band-limit.

    Returns:
        nphi (int): Number of phi samples.
    '''
    return 2*L-1



def sampling_mw_ss_ntheta(L: int) -> int:
    '''Computes the number of theta samples for McEwen and Wiaux symmetric sampling.
    
    Note: Computes the number of samples in [0, pi], *not* over extended domain.

    Args:
        L (int): Harmonic band-limit.
    
    Returns:
        ntheta (int): Number of theta samples.   
    '''
    return L+1



def sampling_mw_ntheta(L: int) -> int:
    '''Computes the number of theta samples for McEwen and Wiaux sampling.

    Note: Computes the number of samples in (0, pi], *not* over extended domain.

    Args:
        L (int): Harmonic band-limit.
    
    Returns:
        ntheta (int): Number of theta samples.
    '''
    return L




#Functions from lines 41 - 110 of s2let_tiling.c



def bandlimit(kernel: str, lam: float, j: int) -> int:
    '''Computes the band-limit of a specific wavelet scale.

    Args:
        kernel (str): The wavelet type.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        j (int): Wavelet scale.
    
    Returns:
        Band-limit (int).
    '''
    if kernel.lower() == "s2dw":
        return math.ceil(lam ** (j+1))
    else:
        #This code currently does not support other wavelet types.
        return print("Invalid wavelet type.")



def L0(kernel: str, lam: float, j: int) -> int:
    '''Computes the minimum harmonic index supported by the given wavelet scale.

    Args:
        kernel (str): The wavelet type.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        j (int): Wavelet scale.
    
    Returns:
        el_min (int): The minimum harmonic multipole "ell" which is supported by a given wavelet scale.
    '''
    if kernel.lower() == "s2dw":
        return math.ceil(lam ** (j-1))
    else:
        #This code currently does not support other wavelet types
        return print("Invalid wavelet type.")



def j_max(L: int, lam: float) -> int:
    '''Computes needlet maximum level required to ensure exact reconstruction.

    Args:
        L (int): Upper harmonic band-limit.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
    
    Returns:
        j_max (int): The maximum wavelet scale used.
    '''
    return math.ceil(log(L) / log(lam))





#files originally from s2let_helper

def n_phi(sampling_scheme: str, L: int) -> int:
    '''Returns the number of phi samples.

    Args:
        sampling_scheme (str): Either 'mw' or 'mwss'.
        L (int): Upper harmonic band-limit.

    Returns:
        nphi (int): Number of phi samples.
    '''
    if sampling_scheme.lower() == "mwss":
        return sampling_mw_ss_nphi(L)
    elif sampling_scheme.lower() == "mw":
        return ampling_mw_nphi(L)


def n_theta(sampling_scheme: str, L: int) -> int:
    '''Returns the number of theta samples.

    Args:
        sampling_scheme (str): Either 'mw' or 'mwss'.
        L (int): Upper harmonic band-limit.

    Returns: 
        ntheta (int): Number of theta samples.
    '''
    if sampling_scheme.lower() == "mwss":
        return sampling_mw_ss_ntheta(L)
    elif sampling_scheme.lower() == "mw":
        return sampling_mw_ntheta(L)


def n_px(sampling_scheme: str, L: int) -> int:
    '''Returns the number of phi samples multiplied by the number of theta samples.
    
    Args:
        sampling_scheme (str): Either 'mw' or 'mwss'.
        L (int): Upper harmonic band-limit.
    
    Returns:
        npix (int): Total number of pixels.
    '''
    return n_phi(sampling_scheme, L) * n_theta(sampling_scheme, L)


def n_lm(L: int) -> int:
    '''Returns the square of the harmonic band-limit.
    
    Args:
        L (int): Upper harmonic band-limit.

    Returns:
        L_squared (int): Total number of harmonic coefficients.
    '''
    return L*L


def n_lm_scal(upsample: bool, L: int, kernel: str, J_min: int, lam: float) -> int:
    '''Computes the square of the band-limit, after determining what the value of the band-limit is.

    Args:
        upsample (bool): Boolean parameter which determines whether to store the scales at j_max resolution or its own resolution.
        L (int): Upper harmonic band-limit.
        kernel (str): The wavelet type.
        J_min (int): First wavelet scale to be used.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
    
    Returns:
        band_lim_squared (int): The square of the band-limit.
    '''
    if upsample == True:
        bandlimit = L
    else:
        bandlimit = min(bandlimit(kernel, J_min - 1, lam, L), L)

    return bandlimit * bandlimit


def n_lmn_wav(lam: float, L: int, J_min: int, upsample: bool, kernel: str, N: int, storage: str, reality: int) -> int:
    '''Computes the flmn sampling size.

    Calls upon functions originally defined in so3, which have been defined explicitly for the case needed here.
    
    Args:
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        L (int): Upper harmonic band-limit.
        J_min (int): First wavelet to be used.
        upsample (bool): Boolean parameter which determines whether to store the scales at j_max resolution or its own resolution.
        kernel (str): The wavelet type.
        N (int): Upper azimuthal band-limit.
        storage (str): The type of storage (padded or compact).
        reality (int): A non-zero value indicates the signal, f, is real.
 
    Returns:
        flmn_size (int): The length of the Wigner space coefficients f,l,m,n.
    '''

    J = j_max(lam, L)

    total = 0

    for i in range(J_min, J):
        if not upsample:
            bandlimit = min(bandlimit(kernel, i, lam, L), L)
            L = bandlimit
            N = min(N, bandlimit)

        if storage.lower() == "padded":
            if reality != 0:
                sampling_flmn_size = N*L*L
            else:
                sampling_flmn_size = (2*N-1)*L*L
        elif storage.lower() == "compact":
            # Both of these are based on the face that the sum over n*n from 1 to N-1 is (N-1)*N*(2N-1)/6.
            if reality != 0:
                sampling_flmn_size = N*(6*L*L-(N-1)*(2*N-1))/6
            else:
                sampling_flmn_size = (2*N-1)*(3*L*L-N*(N-1))/3
        else:
            print("Invalid storage method.")

        total += sampling_flmn_size

    return total


def n_gamma(N: int, steerable: int) -> int:
    '''Computes the number of gamma samples for a given sampling scheme
    
    Args:
        N (int): Upper orientational band-limit. Only flmn with n < N will be stored.
        steerable (int): A non-zero value indicates that the signal is steerable.

    Returns:
        ngamma (int): Number of gamma samples.
    '''
    if steerable != 0:
        return N
    else:
        return 2 * N - 1


def n_scal(upsample:bool, L: int, kernel: str, J_min: int, lam: float, sampling_scheme: str) -> int:
    '''Computes the number of phi samples multiplies by the number of theta samples.
    
    Args:
        upsample (bool): Boolean parameter which determines whether to store the scales at j_max resolution or its own resolution.
        L (int): Upper harmonic band-limit.
        kernel (str): The wavelet type.
        J_min (int): First wavelet to be used.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        sampling_scheme (str): Either 'mw' or 'mwss'.
    
    Returns:
       npix (int): Total number of pixels.
    '''
    if upsample == True:
        bandlimit = L
    else:
        bandlimit = min(bandlimit(kernel, J_min -1, lam, L), L)
    return n_phi(sampling_scheme, L) * n_theta(sampling_scheme, L)


def n_wav(lam: float, L: int, J_min: int, upsample: bool, kernel: str, sampling_scheme: str, steerable: int, N:int) -> int:
    '''Computes the sampling size of the signal, f.
    
    Calls upon functions originally defined in so3, which have been defined explicitly for the case needed here.

    Args:
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        L (int): Upper harmonic band-limit.
        J_min (int): First wavelet to be used.
        upsample (bool): Boolean parameter which determines whether to store the scales at j_max resolution or its own resolution.
        kernel (str): The wavelet type.
        sampling_scheme (str): Either 'mw' or 'mwss'.
        steerable (int): A non-zero value indicates that the signal is steerable.
        N (int): Upper orientational band-limit. Only flmn with n < N will be stored.

    Returns:
        f_size (int): The length of the Wigner space coefficients f,l,m,n, in pixel-space.
    '''
    J = j_max(lam, L)
    bandlimit = L

    total = 0

    for i in range(J_min, J):
        if not upsample:
            bandlimit = min(bandlimit(kernel, i, lam, L), L)
            L = bandlimit
        
            if sampling_scheme.lower() == "mw":
                sampling_nalpha = 2 * L - 1
                sampling_nbeta = L   
            else:
                sampling_nalpha = 2 * L
                sampling_nbeta = L + 1

            if steerable != 0:
                sampling_ngamma = N
            else:
                sampling_ngamma = 2 * N - 1

        sampling_f_size = sampling_nalpha * sampling_nbeta * sampling_ngamma
        total += sampling_f_size
    return total


def n_wav_j(upsample: bool, L: int, kernel: str, j: int, lam: float, sampling_scheme: str, steerable: int, N: int) -> int:
    '''Computes the sampling size of the signal, f.

    Calls upon functions originally defined in so3, which have been defined explicitly for the case needed here.
    
    Args:
        upsample (bool): Boolean parameter which determines whether to store the scales at j_max resolution or its own resolution.
        L (int): Upper harmonic band-limit.
        kernel (str): The wavelet type.
        j (int): Wavelet scale.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        sampling_scheme (str): Either 'mw' or 'mwss'.
        steerable (int): A non-zero value indicates that the signal is steerable.
        N (int): Upper orientational band-limit. Only flmn with n < N will be stored.

    Returns:
        f_size (int): The length of the Wigner space coefficients f,l,m,n, in pixel-space.
    '''
    if not upsample:
        L = min(bandlimit(kernel, j, lam, L ), L)
    
    if sampling_scheme.lower() == "mw":
        sampling_nalpha = 2 * L - 1
        sampling_nbeta = L
    else:
        sampling_nalpha = 2 * L
        sampling_nbeta = L + 1
    
    if steerable != 0:
        sampling_ngamma = N
    else:
        sampling_ngamma = 2 * N - 1 

    sampling_f_size = sampling_nalpha * sampling_nbeta * sampling_ngamma

    return sampling_f_size

