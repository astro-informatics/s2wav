:html_theme.sidebar_secondary.remove:

**************************
Utility Functions
**************************

.. list-table:: Sampling functions.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.utils.samples.nphi`
     - Computes the number of :math:`\phi`.
   * - :func:`~s2wav.utils.samples.ntheta`
     - Computes the number of :math:`\theta` samples.
   * - :func:`~s2wav.utils.samples.j_bandlimit`
     - Computes the band-limit of a specific wavelet scale.
   * - :func:`~s2wav.utils.samples.L0`
     - Computes the minimum harmonic index supported by the given wavelet scale :math:`j`.
   * - :func:`~s2wav.utils.samples.n_px`
     - Returns the number of spherical pixels for a given sampling scheme.
   * - :func:`~s2wav.utils.samples.n_lm`
     - Returns the number of harmonic coefficients at bandlimit L.
   * - :func:`~s2wav.utils.samples.n_lm_scal`
     - Computes the total number of harmonic coefficients for scaling kernels :math:`\Phi_{\el m}`
   * - :func:`~s2wav.utils.samples.n_lmn_wav`
     - Computes the total number of Wigner coefficients for directional wavelet kernels :math:`\Psi^j_{\el n}`.
   * - :func:`~s2wav.utils.samples.n_gamma`
     - Computes the number of :math:`\gamma` samples for a given sampling scheme
   * - :func:`~s2wav.utils.samples.n_scal`
     - Computes the number of pixel-space samples for scaling kernels :math:`\Phi`.
   * - :func:`~s2wav.utils.samples.n_wav`
     - Computes the total number of pixel-space samples for directional wavelet kernels :math:`\Psi`.
   * - :func:`~s2wav.utils.samples.n_wav_j`
     - Number of directional wavelet pixel-space coefficients for a specific scale :math:`j`.
   * - :func:`~s2wav.utils.shapes.n_wav_scales`
     - Evalutes the total number of wavelet scales.
   * - :func:`~s2wav.utils.samples.elm2ind`
     - Convert from spherical harmonic 2D indexing of :math:`(\ell,m)` to 1D index.
   * - :func:`~s2wav.utils.samples.elmn2ind`
     - Convert from Wigner space 3D indexing of :math:`(\ell,m, n)` to 1D index.

.. list-table:: Bandlimiting functions.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.utils.samples.j_max`
     - Computes maximum wavelet scale required to ensure exact reconstruction.
   * - :func:`~s2wav.utils.shapes.LN_j`
     - Computes the harmonic bandlimit and directionality for scale :math:`j`.
   * - :func:`~s2wav.utils.shapes.scal_bandlimit`
     - Returns the harmominc bandlimit of the scaling coefficients.
   * - :func:`~s2wav.utils.shapes.wav_j_bandlimit`
     - Returns the harmominc bandlimit of the scaling coefficients.

.. list-table:: Shape functions.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.utils.shapes.f_scal`
     - Computes the shape of scaling coefficients in pixel-space.
   * - :func:`~s2wav.utils.shapes.f_wav`
     - Computes the shape of wavelet coefficients in pixel-space.
   * - :func:`~s2wav.utils.shapes.f_wav_j`
     - Computes the shape of wavelet coefficients :math:`f^j` in pixel-space.
   * - :func:`~s2wav.utils.shapes.flmn_wav`
     - Returns the shape of wavelet coefficients in Wigner space.
   * - :func:`~s2wav.utils.shapes.flmn_wav_j`
     - Returns the shape of wavelet coefficients :math:`f^j_{\ell m n}` in Wigner space.

.. list-table:: Array constructing and shape checking functions.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.utils.shapes.construct_f`
     - Defines a list of arrays corresponding to f_wav.
   * - :func:`~s2wav.utils.shapes.construct_flm`
     - Returns the shape of scaling coefficients in harmonic space.
   * - :func:`~s2wav.utils.shapes.construct_flmn`
     - Defines a list of arrays corresponding to flmn.
   * - :func:`~s2wav.utils.shapes.wavelet_shape_check`
     - Checks the shape of wavelet coefficients are correct.

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Utilities

   math_functions
   samples
   shapes