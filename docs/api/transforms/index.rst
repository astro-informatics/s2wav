:html_theme.sidebar_secondary.remove:

**************************
Wavelet Transforms
**************************

.. list-table:: Numpy transforms
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.transforms.base.synthesis_looped`
     - Loopy implementation of mapping from wavelet to pixel space.
   * - :func:`~s2wav.transforms.base.synthesis`
     - Vectorised implementation of mapping from wavelet to pixel space.
   * - :func:`~s2wav.transforms.base.analysis_looped`
     - Loopy implementation of mapping from pixel to wavelet space.
   * - :func:`~s2wav.transforms.base.analysis`
     - Vectorised implementation of mapping from pixel to wavelet space.

.. list-table:: JAX transforms
   :widths: 25 25
   :header-rows: 1

   * - :func:`~s2wav.transforms.rec_wav_jax.synthesis`
     - JAX implementation of mapping from wavelet to pixel space (Recursive).
   * - :func:`~s2wav.transforms.rec_wav_jax.analysis`
     - JAX implementation of mapping from pixel to wavelet space (Recursive).
   * - :func:`~s2wav.transforms.rec_wav_jax.flm_to_analysis`
     - JAX implementation of mapping from harmonic to wavelet space (Recursive).
  
   * - :func:`~s2wav.transforms.pre_wav_jax.synthesis`
     - JAX implementation of mapping from wavelet to pixel space (fully precompute).
   * - :func:`~s2wav.transforms.pre_wav_jax.analysis`
     - JAX implementation of mapping from pixel to wavelet space (fully precompute).
   * - :func:`~s2wav.transforms.pre_wav_jax.flm_to_analysis`
     - JAX implementation of mapping from harmonic to wavelet space (fully precompute).
  
  .. list-table:: Matrices precomputations
   :widths: 25 25
   :header-rows: 1

   * - :func:`~s2wav.transforms.construct.generate_wigner_precomputes`
     - JAX function to generate precompute arrays for underlying Wigner transforms.
   * - :func:`~s2wav.transforms.construct.generate_full_precomputes`
     - JAX function to generate precompute arrays for fully precompute transforms.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Wavelet transform

   base
   construct
   rec_wav_jax
   pre_wav_jax
   