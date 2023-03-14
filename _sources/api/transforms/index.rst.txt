:html_theme.sidebar_secondary.remove:

**************************
Wavelet Transforms
**************************

.. list-table:: Wavelet transforms
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.transforms.numpy_wavelets.synthesis_looped`
     - Loopy implementation of mapping from wavelet to pixel space.
   * - :func:`~s2wav.transforms.numpy_wavelets.synthesis`
     - Vectorised implementation of mapping from wavelet to pixel space.
   * - :func:`~s2wav.transforms.numpy_wavelets.analysis_looped`
     - Loopy implementation of mapping from pixel to wavelet space.
   * - :func:`~s2wav.transforms.numpy_wavelets.analysis`
     - Vectorised implementation of mapping from pixel to wavelet space.

   * - :func:`~s2wav.transforms.jax_wavelets.synthesis`
     - JAX implementation of mapping from wavelet to pixel space.
   * - :func:`~s2wav.transforms.jax_wavelets.analysis`
     - JAX implementation of mapping from pixel to wavelet space.
   * - :func:`~s2wav.transforms.jax_wavelets.flm_to_analysis`
     - JAX implementation of mapping from harmonic to wavelet space.
   * - :func:`~s2wav.transforms.jax_wavelets.generate_wigner_precomputes`
     - JAX function to generate precompute arrays for underlying Wigner transforms.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Wavelet transform

   numpy_wavelets
   jax_wavelets
   