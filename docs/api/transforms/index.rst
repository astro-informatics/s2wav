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

   * - Function Name
     - Description
   * - :func:`~s2wav.transforms.wavelet.synthesis`
     - JAX implementation of mapping from wavelet to pixel space (Recursive).
   * - :func:`~s2wav.transforms.wavelet.analysis`
     - JAX implementation of mapping from pixel to wavelet space (Recursive).
   * - :func:`~s2wav.transforms.wavelet.flm_to_analysis`
     - JAX implementation of mapping from harmonic to wavelet coefficients only (Recursive).
   * - :func:`~s2wav.transforms.wavelet_precompute.synthesis`
     - JAX implementation of mapping from wavelet to pixel space (fully precompute).
   * - :func:`~s2wav.transforms.wavelet_precompute.analysis`
     - JAX implementation of mapping from pixel to wavelet space (fully precompute).
   * - :func:`~s2wav.transforms.wavelet_precompute.flm_to_analysis`
     - JAX implementation of mapping from harmonic to wavelet coefficients only (fully precompute).

.. list-table:: PyTorch transforms
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.transforms.wavelet_precompute_torch.synthesis`
     - PyTorch implementation of mapping from wavelet to pixel space (fully precompute).
   * - :func:`~s2wav.transforms.wavelet_precompute_torch.analysis`
     - PyTorch implementation of mapping from pixel to wavelet space (fully precompute).
   * - :func:`~s2wav.transforms.wavelet_precompute_torch.flm_to_analysis`
     - PyTorch implementation of mapping from harmonic to wavelet coefficients only (fully precompute).

.. list-table:: Matrices precomputations
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.transforms.construct.generate_wigner_precomputes`
     - JAX/PyTorch function to generate precompute arrays for underlying Wigner transforms.
   * - :func:`~s2wav.transforms.construct.generate_full_precomputes`
     - JAX/PyTorch function to generate precompute arrays for fully precompute transforms.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Wavelet transform

   base
   construct
   wavelet
   wavelet_precompute
   wavelet_precompute_torch
   