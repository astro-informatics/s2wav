:html_theme.sidebar_secondary.remove:

**************************
Utility Functions
**************************

.. list-table:: Bandlimiting functions.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.samples.L0_j`
     - Computes the minimum harmonic index supported by the given wavelet scale :math:`j`.
   * - :func:`~s2wav.samples.n_wav_scales`
     - Evalutes the total number of wavelet scales.
   * - :func:`~s2wav.samples.j_max`
     - Computes maximum wavelet scale required to ensure exact reconstruction.
   * - :func:`~s2wav.samples.LN_j`
     - Computes the harmonic bandlimit and directionality for scale :math:`j`.
   * - :func:`~s2wav.samples.scal_bandlimit`
     - Returns the harmominc bandlimit of the scaling coefficients.
   * - :func:`~s2wav.samples.wav_j_bandlimit`
     - Returns the harmominc bandlimit of the scaling coefficients.

.. list-table:: Shape functions.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.samples.f_scal`
     - Computes the shape of scaling coefficients in pixel-space.
   * - :func:`~s2wav.samples.f_wav_j`
     - Computes the shape of wavelet coefficients :math:`f^j` in pixel-space.
   * - :func:`~s2wav.samples.flmn_wav_j`
     - Returns the shape of wavelet coefficients :math:`f^j_{\ell m n}` in Wigner space.

.. list-table:: Array constructing and shape checking functions.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.samples.construct_f`
     - Defines a list of arrays corresponding to f_wav.
   * - :func:`~s2wav.samples.construct_flm`
     - Returns the shape of scaling coefficients in harmonic space.
   * - :func:`~s2wav.samples.construct_flmn`
     - Defines a list of arrays corresponding to flmn.
   * - :func:`~s2wav.samples.wavelet_shape_check`
     - Checks the shape of wavelet coefficients are correct.

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Utilities

   shapes