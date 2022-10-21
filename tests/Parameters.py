#Adding a utility function to take parameters and store them as a class.

class s2let_params:
    ''' A class with all parameters that are common to several functions of the API.'''
    def __init__(self, verbosity: int, reality: int, upsample: int, B: int, L: int, 
    J_min: int, N: int, spin: int, original_spin: int, s2let_sampling_t, ssht_dl_method):
        self.verbosity = verbosity
        '''Detail level for diagnostic console output in range [0,5].'''

        self.reality = reality
        '''A number to indicate whether the signal, f, is real or complex.
        
        A non-zero value indicates that the signal, f, is real. 
        Not all functions respect this value - instead there may be separate complex and real functions.
        See documentation of each function for details.
        '''

        self.upsample = upsample
        '''A value to indicate whether the signal is stored in a full-resolution format.
        
        A non-zero value indicates that the signal is stored in a full-resolution format, 
        where each wavelet scale is upsampled to use the same amount of pixels.
        A zero-value indicates that the signal uses only as many pixel as necessary for
        each wavelet scale's upper hamonic land-limit.
        This can lead to significant storage and time savings and is the default behaviour.
        '''

        self.B = B
        '''Wavelet parameter which determines the scale factor between consecutive wavelet scales.'''

        self.L = L
        '''Upper harmonic band-limit. Only flmn with l < L will be stored and considered.'''

        self.J_min = J_min
        '''First wavelet scale to be used.'''

        self.N = N
        '''Upper azimuthal band-limit. Only flmn with n < N will be stored.'''

        self.spin = spin
        '''Spin number of the signal f.'''

        self.original_spin = original_spin
        '''Indicates the spin number the wavelet was lowered from, if it was altered.
        
        If normalisation has value S2LET_WAV_NORM_SPIN_LOWERED this parameter indicates which
        spin number the wavelets were lowered from. Otherwise, it is ignored.'''

        self.s2let_sampling_t = s2let_sampling_t
        '''Sampling scheme to use for samples of the signal, f, as well as the wavelets.'''

        self.ssht_dl_method = ssht_dl_method
        '''Recursion method to use to compute the dl plane for Wigner functions.'''
