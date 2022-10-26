# This file translates functions which can be called to check dimensions of vectors from s2let (c).

from cmath import log
from functools import total_ordering
import math
from tkinter import N


#Defining functions originally called from ssht


def ssht_sampling_mw_ss_nphi(L: int) -> int:
    '''Computes the number of phi samples for McEwen and Wiaux symmetric sampling.
    
    Args:
        L (int): Upper harmonic band-limit.

    Returns:
        nphi: Number of phi samples.   
    '''
    return 2*L



def ssht_sampling_mw_nphi(L: int) -> int:
    '''Computes the number of phi samples for McEwen and Wiaux sampling.
    
    Args:
        L (int): Upper harmonic band-limit.

    Returns:
        nphi: Number of phi samples.
    '''
    return 2*L-1



def ssht_sampling_mw_ss_ntheta(L: int) -> int:
    '''Computes the number of theta samples for McEwen and Wiaux symmetric sampling.
    
    Note: Computes the number of samples in [0, pi], *not* over extended domain.

    Args:
        L (int): Harmonic band-limit.
    
    Returns:
        ntheta: Number of theta samples.   
    '''
    return L+1



def ssht_sampling_mw_ntheta(L: int) -> int:
    '''Computes the number of theta samples for McEwen and Wiaux sampling.

    Note: Computes the number of samples in (0, pi], *not* over extended domain.

    Args:
        L (int): Harmonic band-limit.
    
    Returns:
        ntheta: Number of theta samples.
    '''
    return L




#Functions from lines 41 - 110 of s2let_tiling.c



def s2let_bandlimit(s2let_kernel: str, lam: float, j: int) -> int:
    '''Computes the band-limit of a specific wavelet scale.

    Args:
        s2let_kernel (str): The wavelet type.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        j (int): Wavelet scale.
    
    Returns:
        Band-limit.
    '''
    if s2let_kernel == "S2DW":
        return math.ceil(lam ** (j+1))
    else:
        #This code currently does not support other wavelet types.
        return print("Invalid wavelet type.")



def s2let_L0(s2let_kernel: str, lam: float, j: int) -> int:
    '''Computes the minimum harmonic index supported by the given wavelet scale.

    Args:
        s2let_kernel (str): The wavelet type.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        j (int): Wavelet scale.
    
    Returns:
        el_min.
    '''
    if s2let_kernel == "S2DW":
        return math.ceil(lam ** (j-1))
    else:
        #This code currently does not support other wavelet types
        return print("Invalid wavelet type.")



def s2let_j_max(L: int, lam: float) -> int:
    '''Computes needlet maximum level required to ensure exact reconstruction.

    Args:
        L (int): Upper harmonic band-limit.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
    
    Returns:
        j_max
    '''
    return math.ceil(log(L) / log(lam))





#files originally from s2let_helper

def s2let_n_phi(sampling_scheme: str, L: int) -> int:
    '''Returns the number of phi samples.

    Args:
        sampling_scheme (str): Either S2LET_SAMPLING_MW or S2LET_SAMPLING_MW_SS.
        L (int): Upper harmonic band-limit.

    Returns:
        nphi: Number of phi samples.
    '''
    if sampling_scheme == "S2LET_SAMPLING_MW_SS":
        return ssht_sampling_mw_ss_nphi(L)
    else:
        return ssht_sampling_mw_nphi(L)


def s2let_n_theta(sampling_scheme: str, L: int) -> int:
    '''Returns the number of theta samples.

    Args:
        sampling_scheme (str): Either S2LET_SAMPLING_MW or S2LET_SAMPLING_MW_SS.
        L (int): Upper harmonic band-limit.

    Returns: 
        ntheta: Number of theta samples.
    '''
    if sampling_scheme == "S2LET_SAMPLING_MW_SS":
        return ssht_sampling_mw_ss_ntheta(L)
    else:
        return ssht_sampling_mw_ntheta(L)


def s2let_n_px(sampling_scheme: str, L: int) -> int:
    '''Returns the number of phi samples multiplied by the number of theta samples.
    
    Args:
        sampling_scheme (str): Either S2LET_SAMPLING_MW or S2LET_SAMPLING_MW_SS.
        L (int): Upper harmonic band-limit.
    
    Returns:
        nphi * ntheta.
    '''
    return s2let_n_phi(sampling_scheme, L) * s2let_n_theta(sampling_scheme, L)


def s2let_n_lm(L: int) -> int:
    '''Returns the square of the harmonic band-limit.
    
    Args:
        L (int): Upper harmonic band-limit.

    Returns:
        Square of L.
    '''
    return L*L


def s2let_n_lm_scal(upsample: bool, L: int, s2let_kernel: str, J_min: int, lam: float) -> int:
    '''Computes the square of the band-limit, after determining what the value of the band-limit is.

    Args:
        upsample (bool): Boolean parameter which determines whether to store the scales at j_max resolution or its own resolution.
        L (int): Upper harmonic band-limit.
        s2let_kernel (str): The wavelet type.
        J_min (int): First wavelet scale to be used.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
    
    Returns:
        The square of the band-limit.
    '''
    if bandlimit == upsample:
        bandlimit = L
    else:
        bandlimit = min(s2let_bandlimit(s2let_kernel, J_min - 1, lam, L), L)

    return bandlimit * bandlimit


def s2let_n_lmn_wav(lam: float, L: int, J_min: int, upsample: bool, s2let_kernel: str, N: int, storage: str, reality: int) -> int:
    '''Computes the flmn sampling size.

    Calls upon functions originally defined in so3, which have been defined explicitly for the case needed here.
    
    Args:
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        L (int): Upper harmonic band-limit.
        J_min (int): First wavelet to be used.
        upsample (bool): Boolean parameter which determines whether to store the scales at j_max resolution or its own resolution.
        N (int): Upper azimuthal band-limit.
        storage (str): The type of storage (padded or compact).
        reality (int): A non-zero value indicates the signal, f, is real.
 
    Returns:
        flmn_size.
    '''

    J = s2let_j_max(lam, L)

    total = 0

    for i in range(J_min, J):
        if not upsample:
            bandlimit = min(s2let_bandlimit(s2let_kernel, i, lam, L), L)
            L = bandlimit
            N = min(N, bandlimit)

        if storage == "SO3_STORAGE_PADDED":
            if reality != 0:
                so3_sampling_flmn_size = N*L*L
            else:
                so3_sampling_flmn_size = (2*N-1)*L*L
        if storage == "SO3_STORAGE_COMPACT":
            # Both of these are based on the face that the sum over n*n from 1 to N-1 is (N-1)*N*(2N-1)/6.
            if reality != 0:
                so3_sampling_flmn_size = N*(6*L*L-(N-1)*(2*N-1))/6
            else:
                so3_sampling_flmn_size = (2*N-1)*(3*L*L-N*(N-1))/3
        else:
            print("Invalid storage method.")

        total += so3_sampling_flmn_size

    return total


def s2let_n_gamma(N: int, steerable: int) -> int:
    '''Computes the number of gamma samples for a given sampling scheme
    
    Args:
        N (int): Upper orientational band-limit. Only flmn with n < N will be stored.
        steerable (int): A non-zero value indicates that the signal is steerable.

    Returns:
        ngamma: Number of gamma samples.
    '''
    if steerable != 0:
        return N
    else:
        return 2 * N - 1


def s2let_n_scal(upsample:bool, L: int, s2let_kernel: str, J_min: int, lam: float, sampling_scheme: str) -> int:
    '''Computes the number of phi samples multiplies by the number of theta samples.
    
    Args:
        upsample (bool): Boolean parameter which determines whether to store the scales at j_max resolution or its own resolution.
        L (int): Upper harmonic band-limit.
        s2let_kernel (str): The wavelet type.
        J_min (int): First wavelet to be used.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        sampling_scheme (str): Either S2LET_SAMPLING_MW or S2LET_SAMPLING_MW_SS.
    
    Returns:
        nphi * ntheta.
    '''
    if bandlimit == upsample:
        bandlimit = L
    else:
        bandlimit = min(s2let_bandlimit(s2let_kernel, J_min -1, lam, L), L)
    return s2let_n_phi(sampling_scheme, L) * s2let_n_theta(sampling_scheme, L)


def s2let_n_wav(lam: float, L: int, J_min: int, upsample: bool, s2let_kernel: str, sampling_scheme: str, steerable: int, N:int) -> int:
    '''Computes the sampling size of the signal, f.
    
    Calls upon functions originally defined in so3, which have been defined explicitly for the case needed here.

    Args:
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        L (int): Upper harmonic band-limit.
        J_min (int): First wavelet to be used.
        upsample (bool): Boolean parameter which determines whether to store the scales at j_max resolution or its own resolution.
        s2let_kernel (str): The wavelet type.
        sampling_scheme (str): Either S2LET_SAMPLING_MW or S2LET_SAMPLING_MW_SS.
        steerable (int): A non-zero value indicates that the signal is steerable.
        N (int): Upper orientational band-limit. Only flmn with n < N will be stored.

    Returns:
        f_size.
    '''
    J = s2let_j_max(lam, L)
    bandlimit = L

    total = 0

    for i in range(J_min, J):
        if not upsample:
            bandlimit = min(s2let_bandlimit(s2let_kernel, i, lam, L), L)
            L = bandlimit
        
            if sampling_scheme == "S2LET_SAMPLING_MW":
                so3_sampling_nalpha = 2 * L - 1
                so3_sampling_nbeta = L   
            else:
                so3_sampling_nalpha = 2 * L
                so3_sampling_nbeta = L + 1

            if steerable != 0:
                so3_sampling_ngamma = N
            else:
                so3_sampling_ngamma = 2 * N - 1

        so3_sampling_f_size = so3_sampling_nalpha * so3_sampling_nbeta * so3_sampling_ngamma
        total += so3_sampling_f_size
    return total


def s2let_n_wav_j(upsample: bool, L: int, s2let_kernel: str, j: int, lam: float, sampling_scheme: str, steerable: int, N: int) -> int:
    '''Computes the sampling size of the signal, f.

    Calls upon functions originally defined in so3, which have been defined explicitly for the case needed here.
    
    Args:
        upsample (bool): Boolean parameter which determines whether to store the scales at j_max resolution or its own resolution.
        L (int): Upper harmonic band-limit.
        s2let_kernel (str): The wavelet type.
        j (int): Wavelet scale.
        lam (float): Wavelet parameter which determines the scale factor between consecutive wavelet scales.
        sampling_scheme (str): Either S2LET_SAMPLING_MW or S2LET_SAMPLING_MW_SS.
        steerable (int): A non-zero value indicates that the signal is steerable.
        N (int): Upper orientational band-limit. Only flmn with n < N will be stored.

    Returns:
        f_size.
    '''
    if not upsample:
        L = min(s2let_bandlimit(s2let_kernel, j, lam, L ), L)
    
    if sampling_scheme == "S2LET_SAMPLING_MW":
        so3_sampling_nalpha = 2 * L - 1
        so3_sampling_nbeta = L
    else:
        so3_sampling_nalpha = 2 * L
        so3_sampling_nbeta = L + 1
    
    if steerable != 0:
        so3_sampling_ngamma = N
    else:
        so3_sampling_ngamma = 2 * N - 1

    so3_sampling_f_size = so3_sampling_nalpha * so3_sampling_nbeta * so3_sampling_ngamma

    return so3_sampling_f_size

