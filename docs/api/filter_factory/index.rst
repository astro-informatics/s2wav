:html_theme.sidebar_secondary.remove:

**************************
Filter Factory 
**************************

.. list-table:: Filter generators.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.filter_factory.filters.filters_axisym`
     - Computes wavelet kernels :math:`\Psi^j_{\ell m}` and scaling kernel :math:`\Phi_{\ell m}` in harmonic space.
   * - :func:`~s2wav.filter_factory.filters.filters_directional`
     - Generates the harmonic coefficients for the directional tiling wavelets in harmonic space.
   * - :func:`~s2wav.filter_factory.filters.filters_axisym_vectorised`
     - Vectorised implementation of :func:`~s2wav.filter_factory.filters.filters_directional`.
   * - :func:`~s2wav.filter_factory.filters.filters_directional_vectorised`
     - Vectorised implementation of :func:`~s2wav.filter_factory.filters.filters_directional`.

.. list-table:: Wavelet kernel functions.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.filter_factory.kernels.tiling_integrand`
     - Tiling integrand for scale-discretised wavelets.
   * - :func:`~s2wav.filter_factory.kernels.part_scaling_fn`
     - Computes integral used to calculate smoothly decreasing function :math:`k_{\lambda}`.
   * - :func:`~s2wav.filter_factory.kernels.k_lam`
     - Compute function :math:`k_{\lambda}` used as a wavelet generating function.

.. list-table:: Wavelet tiling functions.
   :widths: 25 25
   :header-rows: 1

   * - Function Name
     - Description
   * - :func:`~s2wav.filter_factory.tiling.tiling_direction`
     - Generates the harmonic coefficients for the directionality component of the tiling functions.
   * - :func:`~s2wav.filter_factory.tiling.spin_normalization`
     - Computes the normalization factor for spin-lowered wavelets, which is :math:`\sqrt{\frac{(l+s)!}{(l-s)!}}`.
   * - :func:`~s2wav.filter_factory.tiling.spin_normalization_vectorised`
     - Vectorised version of :func:`~s2wav.filter_factory.tiling.spin_normalization`.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Wavelet generators

   filters
   tiling
   kernels