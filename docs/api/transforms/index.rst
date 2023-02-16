:html_theme.sidebar_secondary.remove:

**************************
Wavelet Transforms
**************************

.. list-table:: Wavelet transforms.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.transforms.analysis.analysis_transform_looped`
     - Loopy implementation of mapping from pixel to wavelet space.
   * - :func:`~s2wav.transforms.analysis.analysis_transform_vectorised`
     - Vectorised implementation of mapping from pixel to wavelet space.
   * - :func:`~s2wav.transforms.synthesis.synthesis_transform_looped`
     - Loopy implementation of mapping from wavelet to pixel space.
   * - :func:`~s2wav.transforms.synthesis.synthesis_transform_vectorised`
     - Vectorised implementation of mapping from wavelet to pixel space.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Wavelet transform

   analysis
   synthesis
   