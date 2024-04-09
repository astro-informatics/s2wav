:html_theme.sidebar_secondary.remove:

**************************
Notebooks
**************************
A series of tutorial notebooks which go through the absolute base level application of 
``S2WAV`` apis. Post alpha release we will add examples for more involved applications, 
in the time being feel free to contact contributors for advice! At a high-level the 
``S2WAV`` package is structured such that the 2 primary transforms, the analysis and 
synthesis directional wavelet transforms, can easily be accessed.

Core usage |:rocket:|
-----------------
To import and use ``S2WAV``  is as simple follows: 

.. code-block:: Python 

    import s2wav 

    # Compute wavelet coefficients
    f_wav, f_scal = s2wav.analysis(f, L, N)

    # Map back to signal on the sphere 
    f = s2wav.synthesis(f_wav, f_scal, L, N)


C backend library support |:bulb:|
----------------------------------
``S2WAV`` also supports JAX frontend wrappers for the existing `SSHT <https://astro-informatics.github.io/ssht/>`_ 
spherical harmonic and Wigner transform C libraries which, though limited to CPU compute, are nevertheless very fast 
and memory efficient when e.g. GPU compute is not available. To call this operating mode simply run

.. code-block:: Python 

    import s2wav 

    # Compute wavelet coefficients
    f_wav, f_scal = s2wav.analysis(f, L, N, use_c_backend=True)

    # Map back to signal on the sphere 
    f = s2wav.synthesis(f_wav, f_scal, L, N, use_c_backend=True)

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Jupyter Notebooks

   numpy_transform/numpy_transforms.nblink
   jax_transform/jax_transforms.nblink
   jax_ssht_transform/jax_transforms.nblink
   torch_transform/torch_transforms.nblink
